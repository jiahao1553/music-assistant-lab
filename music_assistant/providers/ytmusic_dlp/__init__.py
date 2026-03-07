"""YouTube Music (Anonymous) provider for Music Assistant.

Uses yt-dlp to search, browse, and download audio from YouTube Music
without requiring user authentication. Audio is downloaded to a local
cache directory and played from disk.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shutil
from collections.abc import AsyncGenerator
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    EventType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    InvalidDataError,
    MediaNotFoundError,
    SetupFailedError,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    ItemMapping,
    MediaItemImage,
    Playlist,
    ProviderMapping,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.tags import async_parse_tags
from music_assistant.helpers.util import install_package, parse_title_and_version
from music_assistant.models.music_provider import MusicProvider

from .helpers import (
    get_artist_albums,
    get_artist_info,
    get_artist_top_tracks,
    get_playlist_info,
    get_song_radio,
    get_track_info,
    search_channels,
    search_playlists,
    search_ytmusic,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_CACHE_DIR = "cache_dir"
CONF_PRE_DOWNLOAD_COUNT = "pre_download_count"
DEFAULT_CACHE_DIR = "/tmp/ytmusic_cache"  # noqa: S108
YTM_DOMAIN = "https://music.youtube.com"

SUPPORTED_FEATURES = {
    ProviderFeature.SEARCH,
    ProviderFeature.BROWSE,
    ProviderFeature.LIBRARY_TRACKS,
    ProviderFeature.ARTIST_ALBUMS,
    ProviderFeature.ARTIST_TOPTRACKS,
    ProviderFeature.SIMILAR_TRACKS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return YoutubeMusicDlpProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider.

    :param instance_id: id of an existing provider instance (None if new instance setup).
    :param action: optional action key called from config entries UI.
    :param values: the (intermediate) raw values for config entries sent with the action.
    """
    if action == "clear_cache":
        cache_dir = Path(str((values or {}).get(CONF_CACHE_DIR, DEFAULT_CACHE_DIR)))
        if cache_dir.exists():
            await asyncio.to_thread(shutil.rmtree, cache_dir, ignore_errors=True)
        return (
            ConfigEntry(
                key="clear_cache_result",
                type=ConfigEntryType.LABEL,
                label=f"Cache cleared: {cache_dir}",
            ),
        )
    return (
        ConfigEntry(
            key=CONF_CACHE_DIR,
            type=ConfigEntryType.STRING,
            default_value=DEFAULT_CACHE_DIR,
            label="Cache Directory",
            required=True,
            description="Directory where downloaded audio files will be stored.",
        ),
        ConfigEntry(
            key=CONF_PRE_DOWNLOAD_COUNT,
            type=ConfigEntryType.INTEGER,
            default_value=2,
            label="Pre-download count",
            required=False,
            description="Number of upcoming queue tracks to pre-download in the background (0-5).",
            range=(0, 5),
        ),
        ConfigEntry(
            key="clear_cache",
            type=ConfigEntryType.ACTION,
            label="Clear cache",
            description="Delete all cached audio files from the cache directory.",
        ),
    )


class YoutubeMusicDlpProvider(MusicProvider):
    """Anonymous YouTube Music provider backed by yt-dlp."""

    _cache_dir: Path
    _pre_download_count: int
    _in_progress: set[str]
    _unsub_callbacks: list[Callable[[], None]]

    async def handle_async_init(self) -> None:
        """Set up the provider."""
        logging.getLogger("yt_dlp").setLevel(self.logger.level + 10)
        await install_package("yt-dlp[default]")
        try:
            await asyncio.to_thread(importlib.import_module, "yt_dlp")
        except ImportError as err:
            raise SetupFailedError("Package yt_dlp failed to install") from err
        self._cache_dir = Path(str(self.config.get_value(CONF_CACHE_DIR) or DEFAULT_CACHE_DIR))
        self._pre_download_count = int(self.config.get_value(CONF_PRE_DOWNLOAD_COUNT) or 2)  # type: ignore[arg-type]
        self._in_progress = set()
        self._unsub_callbacks = []
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._unsub_callbacks.append(
            self.mass.subscribe(self._on_queue_updated, EventType.QUEUE_UPDATED)
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload of the provider."""
        for unsub in self._unsub_callbacks:
            unsub()
        self._unsub_callbacks.clear()

    async def _on_queue_updated(self, event: MassEvent) -> None:
        """Start downloading the current track as soon as the queue index changes."""
        from music_assistant_models.player_queue import PlayerQueue  # noqa: PLC0415

        queue = event.data
        if not isinstance(queue, PlayerQueue) or not queue.current_item:
            return
        media_item = queue.current_item.media_item
        if not media_item:
            return
        video_id = next(
            (m.item_id for m in media_item.provider_mappings if m.provider_domain == self.domain),
            None,
        )
        if video_id and not self._find_cached_file(video_id):
            self.mass.create_task(self._ensure_downloaded(video_id))

    @use_cache(3600 * 24 * 7)
    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Perform search on music provider.

        :param search_query: Search query.
        :param media_types: A list of media_types to include. All types if None.
        :param limit: Number of items to return in the search (per type).
        """
        results = SearchResults()

        # Normalise: treat an empty list the same as all types
        active_types = set(media_types) if media_types else set(MediaType)

        want_tracks = MediaType.TRACK in active_types
        want_playlists = MediaType.PLAYLIST in active_types or MediaType.ALBUM in active_types
        want_artists = MediaType.ARTIST in active_types

        async def _empty() -> list[dict[str, Any]]:
            return []

        track_entries, playlist_entries, channel_entries = await asyncio.gather(
            search_ytmusic(search_query, self.logger.level, limit=limit)
            if want_tracks
            else _empty(),
            search_playlists(search_query, self.logger.level, limit=limit)
            if want_playlists
            else _empty(),
            search_channels(search_query, self.logger.level, limit=limit)
            if want_artists
            else _empty(),
        )

        tracks: list[Track | ItemMapping] = []
        for entry in track_entries:
            with suppress(InvalidDataError, KeyError, TypeError):
                if track := self._parse_track_from_entry(entry):
                    tracks.append(track)
        results.tracks = tracks

        playlists: list[Playlist | ItemMapping] = []
        for entry in playlist_entries:
            with suppress(InvalidDataError, KeyError, TypeError):
                playlists.append(self._parse_playlist_from_entry(entry))
        results.playlists = playlists

        artists: list[Artist | ItemMapping] = []
        for entry in channel_entries:
            with suppress(InvalidDataError, KeyError, TypeError):
                artists.append(self._parse_artist_from_entry(entry))
        results.artists = artists

        return results

    @use_cache(3600 * 24 * 7)
    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        info = await get_track_info(prov_track_id, self.logger.level)
        if info:
            track = self._parse_track_from_entry(info)
            if track:
                return track

        # yt-dlp info extraction failed (rate limit, sign-in required, etc.).
        # If we have a cached file for this track, build a Track from its ID3 tags
        # so that the track remains playable even when the network call fails.
        if local_path := self._find_cached_file(prov_track_id):
            with suppress(Exception):
                tags = await async_parse_tags(str(local_path), local_path.stat().st_size)
                name, version = parse_title_and_version(tags.title or prov_track_id)
                track = Track(
                    item_id=prov_track_id,
                    provider=self.instance_id,
                    name=name,
                    version=version,
                    duration=int(tags.duration or 0),
                    provider_mappings={
                        ProviderMapping(
                            item_id=prov_track_id,
                            provider_domain=self.domain,
                            provider_instance=self.instance_id,
                            url=f"{YTM_DOMAIN}/watch?v={prov_track_id}",
                            audio_format=AudioFormat(content_type=ContentType.MP3),
                        )
                    },
                )
                if tags.artists:
                    track.artists = UniqueList(
                        [
                            ItemMapping(
                                media_type=MediaType.ARTIST,
                                item_id=tags.artists[0],
                                provider=self.instance_id,
                                name=tags.artists[0],
                            )
                        ]
                    )
                self.logger.debug("Falling back to cached file tags for track %s", prov_track_id)
                return track

        msg = f"Track {prov_track_id} not found"
        raise MediaNotFoundError(msg)

    @use_cache(3600 * 24 * 30)
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get full artist details by id."""
        info = await get_artist_info(prov_artist_id, self.logger.level)
        if not info:
            msg = f"Artist {prov_artist_id} not found"
            raise MediaNotFoundError(msg)
        return self._parse_artist(info, prov_artist_id)

    @use_cache(3600 * 24 * 7)
    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Get a list of albums for the given artist."""
        raw = await get_artist_albums(prov_artist_id, self.logger.level)
        albums = []
        for entry in raw:
            with suppress(InvalidDataError, KeyError, TypeError):
                albums.append(self._parse_album_from_entry(entry))
        return albums

    @use_cache(3600 * 24 * 7)
    async def get_artist_toptracks(self, prov_artist_id: str) -> list[Track]:
        """Get a list of most popular tracks for the given artist."""
        raw = await get_artist_top_tracks(prov_artist_id, self.logger.level)
        tracks = []
        for entry in raw:
            with suppress(InvalidDataError, KeyError, TypeError):
                if track := self._parse_track_from_entry(entry):
                    tracks.append(track)
        return tracks

    @use_cache(3600 * 24 * 7)
    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full playlist details by id."""
        info = await get_playlist_info(prov_playlist_id, self.logger.level, limit=1)
        if not info:
            msg = f"Playlist {prov_playlist_id} not found"
            raise MediaNotFoundError(msg)
        return self._parse_playlist(info, prov_playlist_id)

    @use_cache(3600 * 3)
    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """Return playlist tracks for the given provider playlist id."""
        if page > 0:
            return []
        info = await get_playlist_info(prov_playlist_id, self.logger.level)
        if not info or "entries" not in info:
            return []
        tracks = []
        for index, entry in enumerate(info["entries"], 1):
            if not entry:
                continue
            with suppress(InvalidDataError, KeyError, TypeError):
                if track := self._parse_track_from_entry(entry):
                    track.position = index
                    tracks.append(track)
        return tracks

    @use_cache(3600 * 24)
    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """Retrieve a dynamic list of tracks based on the provided item."""
        raw = await get_song_radio(prov_track_id, self.logger.level, limit=limit)
        tracks = []
        for entry in raw:
            with suppress(InvalidDataError, KeyError, TypeError):
                if track := self._parse_track_from_entry(entry):
                    tracks.append(track)
        return tracks

    async def get_library_tracks(self) -> AsyncGenerator[Track, None]:
        """Yield all tracks currently present in the local cache directory."""
        tracks_dir = self._cache_dir / "tracks"
        if not tracks_dir.is_dir():
            return
        for fpath in tracks_dir.iterdir():
            if not fpath.is_file() or fpath.stat().st_size == 0:
                continue
            # filename format: "Song Title [video_id].mp3"
            stem = fpath.stem
            video_id = (
                stem[stem.rfind("[") + 1 : -1] if stem.endswith("]") and "[" in stem else stem
            )
            with suppress(Exception):
                tags = await async_parse_tags(str(fpath), fpath.stat().st_size)
                name, version = parse_title_and_version(tags.title or video_id)
                track = Track(
                    item_id=video_id,
                    provider=self.instance_id,
                    name=name,
                    version=version,
                    duration=int(tags.duration or 0),
                    provider_mappings={
                        ProviderMapping(
                            item_id=video_id,
                            provider_domain=self.domain,
                            provider_instance=self.instance_id,
                            url=f"{YTM_DOMAIN}/watch?v={video_id}",
                            audio_format=AudioFormat(content_type=ContentType.MP3),
                        )
                    },
                )
                if tags.artists:
                    track.artists = UniqueList(
                        [
                            ItemMapping(
                                media_type=MediaType.ARTIST,
                                item_id=tags.artists[0],
                                provider=self.instance_id,
                                name=tags.artists[0],
                            )
                        ]
                    )
                yield track

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return the content details for the given track when it will be streamed."""
        self.mass.create_task(self._pre_download_upcoming(item_id))

        # If the file is fully cached, return LOCAL_FILE for full seek support.
        # The streams controller calls get_stream_details BEFORE sending HTTP headers to
        # the player, so we must return immediately when the file is not ready. We use
        # CUSTOM in that case so the player receives 200 OK at once, and the actual
        # download wait happens inside get_audio_stream (after headers are sent).
        local_path = self._find_cached_file(item_id)
        if local_path is not None and item_id not in self._in_progress:
            tags = await async_parse_tags(str(local_path), local_path.stat().st_size)
            return StreamDetails(
                provider=self.instance_id,
                item_id=item_id,
                audio_format=AudioFormat(
                    content_type=ContentType.try_parse(tags.format or "mp3"),
                    sample_rate=tags.sample_rate,
                    bit_depth=tags.bits_per_sample,
                    channels=tags.channels,
                    bit_rate=tags.bit_rate,
                ),
                media_type=MediaType.TRACK,
                stream_type=StreamType.LOCAL_FILE,
                duration=int(tags.duration or 0),
                size=local_path.stat().st_size,
                path=str(local_path),
                can_seek=True,
                allow_seek=True,
            )

        # File not ready yet — return immediately; download happens in get_audio_stream.
        # Avoid any network call here: the streams controller calls get_stream_details
        # BEFORE sending HTTP headers to the player, so blocking on get_track() would
        # delay the 200 OK response and cause connection timeouts.
        #
        # Set expiration=0 so these CUSTOM streamdetails are immediately considered stale.
        # When _load_item re-evaluates streamdetails (e.g. on next-press), it will call
        # get_stream_details again, allowing the LOCAL_FILE path to be taken once the
        # download completes. Without this, the 10-minute default expiration causes
        # a cached file to still be streamed as CUSTOM (duration=0), breaking elapsed-time
        # tracking and causing "seek to time-since-next-pressed" when user presses play.
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(content_type=ContentType.MP3),
            media_type=MediaType.TRACK,
            stream_type=StreamType.CUSTOM,
            duration=0,
            can_seek=False,
            allow_seek=False,
            expiration=0,
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from cache, waiting for download to complete if necessary."""
        import aiofiles  # noqa: PLC0415

        local_path = await self._ensure_downloaded(streamdetails.item_id)
        async with aiofiles.open(str(local_path), "rb") as f:
            while chunk := await f.read(65536):
                yield chunk

    async def _ensure_downloaded(self, video_id: str) -> Path:
        """Download a track if it is not already cached, then return its path."""
        # Wait for any in-progress download to finish before checking the cache.
        # yt-dlp writes the final file incrementally via ffmpeg, so _find_cached_file
        # can return a partially-written file if we check before the download completes.
        while video_id in self._in_progress:
            await asyncio.sleep(0.5)

        if cached := self._find_cached_file(video_id):
            return cached

        self._in_progress.add(video_id)
        try:
            path = await asyncio.to_thread(self._download_track, video_id)
        finally:
            self._in_progress.discard(video_id)

        if not path or not path.exists():
            msg = f"Download failed for video {video_id}"
            raise MediaNotFoundError(msg)
        return path

    def _download_track(self, video_id: str) -> Path | None:
        """Run yt-dlp download synchronously (called in thread pool)."""
        import yt_dlp  # noqa: PLC0415

        out_dir = self._cache_dir / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        outtmpl = str(out_dir / "%(title)s [%(id)s].%(ext)s")

        ydl_opts: dict[str, Any] = {
            "quiet": self.logger.level > logging.DEBUG,
            "verbose": self.logger.level <= logging.DEBUG,
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                },
                {"key": "FFmpegMetadata", "add_metadata": True},
            ],
            "extractor_args": {
                "youtube": {
                    "skip": ["translated_subs", "dash"],
                    "player_client": ["tv_embedded", "ios", "mweb"],
                    "player_skip": ["webpage"],
                },
            },
        }

        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except yt_dlp.utils.DownloadError as err:
                self.logger.warning("yt-dlp download failed for %s: %s", video_id, err)
                return None

        return self._find_cached_file(video_id)

    def _find_cached_file(self, video_id: str) -> Path | None:
        """Return the cached file path for a video_id if it exists."""
        tracks_dir = self._cache_dir / "tracks"
        if not tracks_dir.is_dir():
            return None
        for fpath in tracks_dir.iterdir():
            if f"[{video_id}]" in fpath.stem and fpath.stat().st_size > 0:
                return fpath
        return None

    async def _pre_download_upcoming(self, current_video_id: str) -> None:
        """Download the next N tracks from every active queue in the background."""
        if self._pre_download_count <= 0:
            return
        for queue in self.mass.player_queues.all():
            current_index = queue.current_index
            if current_index is None:
                continue
            # Fetch the slice of items that come after the current one
            upcoming = self.mass.player_queues.items(
                queue.queue_id,
                limit=self._pre_download_count,
                offset=current_index + 1,
            )
            downloaded = 0
            for item in upcoming:
                if not item.media_item:
                    continue
                video_id = next(
                    (
                        m.item_id
                        for m in item.media_item.provider_mappings
                        if m.provider_domain == self.domain
                    ),
                    None,
                )
                if not video_id or self._find_cached_file(video_id):
                    continue
                self.logger.debug("Pre-downloading upcoming track %s", video_id)
                with suppress(Exception):
                    await self._ensure_downloaded(video_id)
                downloaded += 1
                if downloaded >= self._pre_download_count:
                    break

    def _parse_track_from_entry(self, entry: dict[str, Any]) -> Track | None:
        """Parse a yt-dlp flat/full entry into a Track."""
        video_id = entry.get("id") or entry.get("url")
        if not video_id:
            return None
        # Strip URL prefix if present
        if "/" in video_id:
            video_id = video_id.split("v=")[-1].split("/")[-1]

        title = entry.get("title") or entry.get("fulltitle") or "Unknown"
        name, version = parse_title_and_version(title)

        channel_id = entry.get("channel_id") or entry.get("uploader_id") or "unknown"
        channel_name = entry.get("channel") or entry.get("uploader") or "Unknown"

        track = Track(
            item_id=video_id,
            provider=self.instance_id,
            name=name,
            version=version,
            provider_mappings={
                ProviderMapping(
                    item_id=video_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"{YTM_DOMAIN}/watch?v={video_id}",
                    audio_format=AudioFormat(content_type=ContentType.MP3),
                )
            },
        )
        track.artists = UniqueList(
            [
                ItemMapping(
                    media_type=MediaType.ARTIST,
                    item_id=channel_id,
                    provider=self.instance_id,
                    name=channel_name,
                )
            ]
        )
        if not track.artists:
            msg = "Track is missing artists"
            raise InvalidDataError(msg)

        if thumbnails := entry.get("thumbnails"):
            track.metadata.images = UniqueList(self._parse_thumbnails(thumbnails))

        with suppress(TypeError, ValueError):
            duration = entry.get("duration")
            if duration is not None:
                track.duration = int(float(duration))

        return track

    def _parse_playlist_from_entry(self, entry: dict[str, Any]) -> Playlist:
        """Parse a flat yt-dlp search entry into a Playlist."""
        playlist_id = entry.get("id") or entry.get("url", "")
        if not playlist_id:
            msg = "Playlist entry has no id"
            raise InvalidDataError(msg)
        title = entry.get("title") or playlist_id
        playlist = Playlist(
            item_id=playlist_id,
            name=title,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=playlist_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"{YTM_DOMAIN}/playlist?list={playlist_id}",
                )
            },
        )
        if thumbnails := entry.get("thumbnails"):
            playlist.metadata.images = UniqueList(self._parse_thumbnails(thumbnails))
        playlist.owner = entry.get("uploader") or entry.get("channel") or self.name
        return playlist

    def _parse_artist_from_entry(self, entry: dict[str, Any]) -> Artist:
        """Parse a flat yt-dlp channel search entry into an Artist."""
        channel_id = entry.get("channel_id") or entry.get("id") or entry.get("url", "")
        if not channel_id:
            msg = "Channel entry has no id"
            raise InvalidDataError(msg)
        name = entry.get("channel") or entry.get("uploader") or entry.get("title") or channel_id
        artist = Artist(
            item_id=channel_id,
            name=name,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=channel_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"https://www.youtube.com/channel/{channel_id}",
                )
            },
        )
        if thumbnails := entry.get("thumbnails"):
            artist.metadata.images = UniqueList(self._parse_thumbnails(thumbnails))
        return artist

    def _parse_artist(self, info: dict[str, Any], channel_id: str) -> Artist:
        """Parse a yt-dlp channel extract_info result into an Artist."""
        name = info.get("channel") or info.get("uploader") or info.get("title") or channel_id
        artist = Artist(
            item_id=channel_id,
            name=name,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=channel_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"https://www.youtube.com/channel/{channel_id}",
                )
            },
        )
        if description := info.get("description"):
            artist.metadata.description = description
        if thumbnails := info.get("thumbnails"):
            artist.metadata.images = UniqueList(self._parse_thumbnails(thumbnails))
        return artist

    def _parse_album_from_entry(self, entry: dict[str, Any]) -> Album:
        """Parse a yt-dlp playlist entry into an Album."""
        album_id = entry.get("id") or entry.get("url", "")
        if not album_id:
            msg = "Album entry has no id"
            raise InvalidDataError(msg)
        title = entry.get("title") or "Unknown"
        name, version = parse_title_and_version(title)
        album = Album(
            item_id=album_id,
            name=name,
            version=version,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=album_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"{YTM_DOMAIN}/playlist?list={album_id}",
                )
            },
        )
        if thumbnails := entry.get("thumbnails"):
            album.metadata.images = UniqueList(self._parse_thumbnails(thumbnails))
        return album

    def _parse_playlist(self, info: dict[str, Any], playlist_id: str) -> Playlist:
        """Parse a yt-dlp playlist extract_info result into a Playlist."""
        title = info.get("title") or playlist_id
        playlist = Playlist(
            item_id=playlist_id,
            name=title,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=playlist_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"{YTM_DOMAIN}/playlist?list={playlist_id}",
                )
            },
        )
        if description := info.get("description"):
            playlist.metadata.description = description
        if thumbnails := info.get("thumbnails"):
            playlist.metadata.images = UniqueList(self._parse_thumbnails(thumbnails))
        if uploader := info.get("uploader") or info.get("channel"):
            playlist.owner = uploader
        else:
            playlist.owner = self.name
        return playlist

    def _parse_thumbnails(self, thumbnails: list[dict[str, Any]]) -> list[MediaItemImage]:
        """Parse yt-dlp thumbnails list to MediaItemImage list."""
        result: list[MediaItemImage] = []
        seen: set[str] = set()
        for thumb in sorted(thumbnails, key=lambda t: t.get("width", 0) or 0, reverse=True):
            url: str = thumb.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            width = thumb.get("width", 0) or 0
            height = thumb.get("height", 0) or 0
            image_type = (
                ImageType.LANDSCAPE if height > 0 and width / height > 2.0 else ImageType.THUMB
            )
            result.append(
                MediaItemImage(
                    type=image_type,
                    path=url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            )
        return result
