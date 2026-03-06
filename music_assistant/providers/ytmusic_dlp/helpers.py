"""Helper module for the anonymous YouTube Music provider.

Uses yt-dlp to extract metadata from YouTube Music without requiring user authentication.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote

from music_assistant.providers.ytmusic_dlp.constants import YTMRecommendationIcons

LOGGER = logging.getLogger(__name__)
YTM_DOMAIN = "https://music.youtube.com"

YTM_FEATURED_PLAYLISTS: dict[str, str] = {
    "Hot Hits": "RDCLAK5uy_kmPRjHDECIcuVwnKsx2Ns-1w0nhO-G2ds",
    "Chill Hits": "RDCLAK5uy_kzInc7TQmdaGUdjBRLFAP8uakeyiEFxbY",
    "Dance Hits": "RDCLAK5uy_mnY1-eO2b8yGXFAvREaxN15MxN0eTVRbs",
    "Top Hits": "RDCLAK5uy_nqyW4KnNEOQy9sVUj0L-CQmGHiRNjYBQE",
    "Feel Good Hits": "RDCLAK5uy_k9ElKEyMoUTmSLNdkCsrYxF4DVHbKYjXo",
}


def _get_base_ydl_opts(logger_level: int) -> dict[str, Any]:
    return {
        "quiet": logger_level > logging.DEBUG,
        "verbose": logger_level <= logging.DEBUG,
    }


async def ytdlp_extract_info(
    url: str, ydl_opts: dict[str, Any]
) -> dict[str, Any] | None:
    """Run yt-dlp extract_info in a thread and return the result."""
    import yt_dlp

    def _extract() -> dict[str, Any] | None:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                return ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError:
                return None

    return await asyncio.to_thread(_extract)


async def search_ytmusic(
    query: str, logger_level: int, limit: int = 20
) -> list[dict[str, Any]]:
    """Search YouTube Music and return a list of track dicts."""
    import yt_dlp

    opts = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "skip_download": True,
    }

    def _search() -> list[dict[str, Any]]:
        # ytsearch{N}:query is the canonical yt-dlp search extractor
        url = f"ytsearch{limit}:{query}"
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                result = ydl.extract_info(url, download=False)
                if result and "entries" in result:
                    return [e for e in result["entries"] if e]
            except yt_dlp.utils.DownloadError:
                pass
        return []

    return await asyncio.to_thread(_search)


async def search_playlists(
    query: str, logger_level: int, limit: int = 10
) -> list[dict[str, Any]]:
    """Search YouTube for playlists matching the query."""
    import yt_dlp

    # Do not use playlist_items here — YouTube search results pages are not playlists
    opts = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "skip_download": True,
    }

    def _search() -> list[dict[str, Any]]:
        # sp=EgIQAw%3D%3D filters YouTube search results to playlists only
        url = f"https://www.youtube.com/results?search_query={quote(query)}&sp=EgIQAw%3D%3D"
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                result = ydl.extract_info(url, download=False)
                if result and "entries" in result:
                    return [e for e in result["entries"] if e][:limit]
            except yt_dlp.utils.DownloadError:
                pass
        return []

    return await asyncio.to_thread(_search)


async def search_channels(
    query: str, logger_level: int, limit: int = 10
) -> list[dict[str, Any]]:
    """Search YouTube for channels (artists) matching the query."""
    import yt_dlp

    # Do not use playlist_items here — YouTube search results pages are not playlists
    opts = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "skip_download": True,
    }

    def _search() -> list[dict[str, Any]]:
        # sp=EgIQAg%3D%3D filters YouTube search results to channels only
        url = f"https://www.youtube.com/results?search_query={quote(query)}&sp=EgIQAg%3D%3D"
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                result = ydl.extract_info(url, download=False)
                if result and "entries" in result:
                    return [e for e in result["entries"] if e][:limit]
            except yt_dlp.utils.DownloadError:
                pass
        return []

    return await asyncio.to_thread(_search)


async def get_track_info(
    video_id: str, logger_level: int
) -> dict[str, Any] | None:
    """Fetch metadata for a single YouTube video/track."""
    opts = {
        **_get_base_ydl_opts(logger_level),
        "skip_download": True,
    }
    return await ytdlp_extract_info(
        f"https://www.youtube.com/watch?v={video_id}", opts
    )


async def get_playlist_info(
    playlist_id: str, logger_level: int, limit: int | None = None
) -> dict[str, Any] | None:
    """Fetch playlist metadata and track listing."""
    import yt_dlp

    opts: dict[str, Any] = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "skip_download": True,
    }
    if limit is not None:
        opts["playlist_items"] = f"1-{limit}"

    def _get() -> dict[str, Any] | None:
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                return ydl.extract_info(
                    f"https://www.youtube.com/playlist?list={playlist_id}",
                    download=False,
                )
            except yt_dlp.utils.DownloadError:
                return None

    return await asyncio.to_thread(_get)


async def get_artist_info(
    channel_id: str, logger_level: int
) -> dict[str, Any] | None:
    """Fetch basic info for a YouTube channel (artist)."""
    import yt_dlp

    opts = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "playlist_items": "0",
        "skip_download": True,
    }

    def _get() -> dict[str, Any] | None:
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                return ydl.extract_info(
                    f"https://www.youtube.com/channel/{channel_id}",
                    download=False,
                )
            except yt_dlp.utils.DownloadError:
                return None

    return await asyncio.to_thread(_get)


async def get_artist_top_tracks(
    channel_id: str, logger_level: int, limit: int = 20
) -> list[dict[str, Any]]:
    """Return top videos for an artist channel."""
    import yt_dlp

    opts = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "playlist_items": f"1-{limit}",
        "skip_download": True,
    }

    def _get() -> list[dict[str, Any]]:
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                result = ydl.extract_info(
                    f"https://www.youtube.com/channel/{channel_id}/videos",
                    download=False,
                )
                if result and "entries" in result:
                    return [e for e in result["entries"] if e]
            except yt_dlp.utils.DownloadError:
                pass
        return []

    return await asyncio.to_thread(_get)


async def get_artist_albums(
    channel_id: str, logger_level: int, limit: int = 20
) -> list[dict[str, Any]]:
    """Return albums for a YouTube channel by searching for playlists."""
    import yt_dlp

    opts = {
        **_get_base_ydl_opts(logger_level),
        "extract_flat": True,
        "playlist_items": f"1-{limit}",
        "skip_download": True,
    }

    def _get() -> list[dict[str, Any]]:
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                result = ydl.extract_info(
                    f"https://www.youtube.com/channel/{channel_id}/playlists",
                    download=False,
                )
                if result and "entries" in result:
                    return [e for e in result["entries"] if e]
            except yt_dlp.utils.DownloadError:
                pass
        return []

    return await asyncio.to_thread(_get)


async def get_song_radio(
    video_id: str, logger_level: int, limit: int = 25
) -> list[dict[str, Any]]:
    """Return auto-generated radio tracks for a given video."""
    result = await get_playlist_info(
        f"RDAMVM{video_id}", logger_level, limit=limit
    )
    if result and "entries" in result:
        return [e for e in result["entries"] if e]
    return []


async def get_recommendations(logger_level: int) -> list[dict[str, Any]]:
    """Return a list of featured playlist dicts as recommendations."""
    results = []
    for name, playlist_id in YTM_FEATURED_PLAYLISTS.items():
        info = await get_playlist_info(playlist_id, logger_level, limit=1)
        if info:
            results.append(
                {
                    "id": playlist_id,
                    "name": name,
                    "thumbnail": (info.get("thumbnails") or [{}])[-1].get("url"),
                }
            )
    return results


def determine_recommendation_icon(name: str) -> str:
    """Determine the icon for a recommendation based on its name."""
    query = name.lower()
    if "listen again" in query:
        return YTMRecommendationIcons.LISTEN_AGAIN
    if "continue" in query:
        return YTMRecommendationIcons.CONTINUE_WATCHING
    if "your mix" in query:
        return YTMRecommendationIcons.YOUR_MIX
    if "new" in query:
        return YTMRecommendationIcons.NEW_RELEASES
    if "recommended" in query:
        return YTMRecommendationIcons.RECOMMENDED
    return YTMRecommendationIcons.DEFAULT
