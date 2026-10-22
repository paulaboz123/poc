"""
media.py - thin wrappers around ffmpeg, ffprobe and Tesseract.

Every external program is called here, with its errors turned into readable messages:
reading recordings (duration, frames, OCR) and building the narrated video.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import config


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int = 3600) -> str:
    """Run a command, return stdout, raise MediaError with the program's own error text."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        raise MediaError(f"'{cmd[0]}' not found. Run: uv run doctor.py") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"'{cmd[0]}' took longer than {timeout}s") from e
    if out.returncode != 0:
        raise MediaError(f"{Path(cmd[0]).name} failed: {(out.stderr or out.stdout).strip()[-500:]}")
    return out.stdout


def duration_s(video: Path) -> float:
    """Length of a video in seconds."""
    out = _run([config.FFPROBE, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video)], timeout=120)
    try:
        return float(out.strip())
    except ValueError as e:
        raise MediaError(f"Could not read the duration of {video.name}") from e


Crop = tuple[float, float, float, float]   # left, top, right, bottom as fractions of the frame


def parse_crop(text: str) -> Crop | None:
    """'0,0,0.78,1' -> (0.0, 0.0, 0.78, 1.0); '' -> None (whole frame)."""
    if not text or not text.strip():
        return None
    try:
        l, t, r, b = (float(x) for x in text.replace(";", ",").split(","))
    except ValueError as e:
        raise MediaError(f"Crop must be 'left,top,right,bottom' fractions, got {text!r}") from e
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
        raise MediaError(f"Crop values must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1: {text!r}")
    return l, t, r, b


def _crop_filter(crop: Crop | None) -> str:
    """ffmpeg filter that keeps only the crop area ('' when there is no crop)."""
    if not crop:
        return ""
    l, t, r, b = crop
    return f"crop=trunc(iw*{r - l:.4f}/2)*2:trunc(ih*{b - t:.4f}/2)*2:trunc(iw*{l:.4f}):trunc(ih*{t:.4f}),"


def sample_frames(video: Path, out_dir: Path, every_s: int, width: int, height: int,
                  crop: Crop | None = None) -> list[Path]:
    """Write one small greyscale JPG every `every_s` seconds. Frame n is at n * every_s."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video),
          "-vf", f"{_crop_filter(crop)}fps=1/{every_s},scale={width}:{height},format=gray",
          "-q:v", "4", str(out_dir / "f_%06d.jpg")])
    return sorted(out_dir.glob("f_*.jpg"))


def grab_frame(video: Path, at_s: float, out: Path, crop: Crop | None = None) -> Path:
    """Save one full-resolution frame at `at_s` seconds (only the cropped area, if given)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", _crop_filter(crop).rstrip(",")] if crop else []
    _run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at_s:.2f}",
          "-i", str(video), *vf, "-frames:v", "1", "-q:v", "2", str(out)], timeout=120)
    if not out.exists():
        raise MediaError(f"No frame at {at_s:.0f}s in {video.name} (past the end?)")
    return out


def ocr(image: Path) -> str:
    """Text on an image, with near-empty lines (OCR noise) removed."""
    raw = _run([config.TESSERACT, str(image), "stdout", "-l", config.OCR_LANG, "--psm", "3"],
               timeout=120)
    lines = (" ".join(line.split()) for line in raw.splitlines())
    return "\n".join(line for line in lines if sum(ch.isalnum() for ch in line) >= 3)


def tesseract_languages() -> list[str]:
    """Language packs installed for Tesseract, e.g. ['eng', 'osd']."""
    out = _run([config.TESSERACT, "--list-langs"], timeout=30)
    return [line.strip() for line in out.splitlines()[1:] if line.strip()]


def _concat_path(p: Path) -> str:
    """A path line for ffmpeg's concat list (forward slashes, quotes escaped)."""
    return "file '" + p.resolve().as_posix().replace("'", "'\\''") + "'"


def stills_to_video(images: list[Path], durations: list[float], audio: Path, out: Path,
                    subtitles: Path | None = None, fps: int = 25) -> Path:
    """One video from still images (each shown for its duration) plus one audio track, in a
    single encode, so picture and sound stay in sync. Subtitles are added as a track that
    players can switch on (and the .srt next to the video is picked up by most players)."""
    if len(images) != len(durations) or not images:
        raise MediaError("Need one duration per image")
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_name(out.stem + "_images.txt")
    lines = []
    for img, dur in zip(images, durations):
        lines += [_concat_path(img), f"duration {dur:.3f}"]
    lines.append(_concat_path(images[-1]))   # concat demuxer quirk: repeat the last image
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio)]
    maps = ["-map", "0:v", "-map", "1:a"]
    if subtitles:
        cmd += ["-i", str(subtitles)]
        maps += ["-map", "2:s", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    cmd += maps + ["-vf", f"fps={fps},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
                   "-tune", "stillimage", "-crf", "20", "-c:a", "aac", "-b:a", "128k",
                   "-shortest", "-movflags", "+faststart", str(out)]
    try:
        _run(cmd)
    finally:
        listing.unlink(missing_ok=True)
    return out


def to_mp3(wav: Path, out: Path) -> Path:
    """Compress the narration for listening (mp3; falls back to .m4a if ffmpeg has no mp3)."""
    try:
        _run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
              "-c:a", "libmp3lame", "-q:a", "4", str(out)])
        return out
    except MediaError:
        alt = out.with_suffix(".m4a")
        _run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
              "-c:a", "aac", "-b:a", "128k", str(alt)])
        return alt


def stream_durations(path: Path) -> dict[str, float]:
    """Length in seconds of each stream type in a media file, e.g. {'video': 85.2, 'audio': 85.3}."""
    out = _run([config.FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,duration",
                "-of", "csv=p=0", str(path)], timeout=60)
    result: dict[str, float] = {}
    for line in out.splitlines():
        kind, _, dur = line.partition(",")
        try:
            result.setdefault(kind.strip(), float(dur))
        except ValueError:
            continue
    return result
