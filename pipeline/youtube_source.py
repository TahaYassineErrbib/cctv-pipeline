"""
Resolves a YouTube video/live-stream URL into a direct stream URL that
OpenCV's VideoCapture can open, using yt-dlp.

OpenCV cannot read youtube.com URLs directly — yt-dlp inspects the page
and extracts the actual underlying media URL (an .m3u8 or direct video
stream), which is what gets handed to cv2.VideoCapture.

Requires: pip install yt-dlp
"""

import config


def resolve_youtube_stream_url(youtube_url, quality=None):
    """
    Returns a direct, openable stream URL for the given YouTube URL.
    Raises a clear error if yt-dlp isn't installed or resolution fails.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )

    quality = quality or config.YOUTUBE_STREAM_QUALITY

    ydl_opts = {
        "format": quality,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

        if "url" in info:
            return info["url"]

        # some formats (e.g. live streams with separate audio/video) return
        # a 'formats' list instead of a single top-level 'url'
        if "formats" in info and info["formats"]:
            return info["formats"][-1]["url"]

        raise RuntimeError(f"Could not resolve a stream URL for: {youtube_url}")


def resolve_video_source(video_source):
    """
    Given config.VIDEO_SOURCE, returns what should actually be passed to
    cv2.VideoCapture — resolving a YouTube URL first if
    config.USE_YOUTUBE_STREAM is enabled, otherwise returning it unchanged
    (local file path or webcam index).
    """
    if config.USE_YOUTUBE_STREAM:
        print(f"Resolving YouTube stream URL for: {video_source}")
        resolved = resolve_youtube_stream_url(video_source)
        print("Resolved stream URL obtained.")
        return resolved

    return video_source
