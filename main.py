"""
CLIPFORGE Backend — FastAPI + FFmpeg AI Video Editor
Handles upload, analysis, auto-cutting, and export.
"""

import os
import uuid
import json
import shutil
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="CLIPFORGE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directory Setup ───────────────────────────────────────────────────────────
# Use /tmp on Render (ephemeral but writable on free tier)
BASE_DIR = Path("/tmp/clipforge")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (use Redis/DB in production)
jobs: dict = {}

# ─── Schemas ──────────────────────────────────────────────────────────────────
class CutSettings(BaseModel):
    mode: str = "smart"           # smart | beat | highlight | interview
    remove_silence: bool = True
    scene_detect: bool = True
    jump_cut: bool = False
    silence_threshold: float = 40  # dB (FFmpeg silencedetect)
    min_clip_length: float = 3.0   # seconds
    target_duration: float = 60.0  # seconds
    output_format: str = "mp4"
    aspect_ratio: str = "16:9"

class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | processing | done | error
    progress: int = 0
    message: str = ""
    cuts: list = []
    output_file: Optional[str] = None
    stats: dict = {}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def ffmpeg_available() -> bool:
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ffprobe_get_duration(filepath: str) -> float:
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            filepath
        ], capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        raise RuntimeError(f"ffprobe failed on {filepath}: {e}")


def detect_silence(filepath: str, noise_db: float = 40, min_duration: float = 0.5) -> list:
    """
    Use ffmpeg silencedetect to find silent segments.
    Returns list of (start, end) tuples.
    """
    result = subprocess.run([
        "ffmpeg", "-i", filepath,
        "-af", f"silencedetect=noise=-{noise_db}dB:d={min_duration}",
        "-f", "null", "-"
    ], capture_output=True, text=True, timeout=120)

    silent_ranges = []
    lines = result.stderr.split("\n")
    start = None
    for line in lines:
        if "silence_start" in line:
            try:
                start = float(line.split("silence_start: ")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "silence_end" in line and start is not None:
            try:
                end_part = line.split("silence_end: ")[1].split(" ")[0]
                end = float(end_part)
                silent_ranges.append((round(start, 2), round(end, 2)))
                start = None
            except (IndexError, ValueError):
                pass
    return silent_ranges


def detect_scenes(filepath: str, threshold: float = 0.4) -> list:
    """
    Use ffmpeg scene detection to find scene change timestamps.
    Returns list of timestamps.
    """
    result = subprocess.run([
        "ffmpeg", "-i", filepath,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ], capture_output=True, text=True, timeout=120)

    timestamps = []
    for line in result.stderr.split("\n"):
        if "pts_time" in line and "showinfo" in line:
            try:
                pts = line.split("pts_time:")[1].split(" ")[0]
                timestamps.append(round(float(pts), 2))
            except (IndexError, ValueError):
                pass
    return timestamps


def build_cut_list(
    duration: float,
    silent_ranges: list,
    scene_times: list,
    settings: CutSettings
) -> list:
    """
    Build the final list of segments to KEEP (inverse of cuts).
    Returns list of dicts with start, end, type, reason.
    """
    # Build list of ranges to remove
    cuts = []

    if settings.remove_silence:
        for s, e in silent_ranges:
            seg_len = e - s
            if seg_len >= settings.min_clip_length:
                cuts.append({
                    "start": s, "end": e,
                    "type": "SILENCE",
                    "reason": f"Silent pause ({seg_len:.1f}s)"
                })

    if settings.scene_detect and settings.mode in ("smart", "highlight"):
        for ts in scene_times:
            cuts.append({
                "start": round(ts - 0.1, 2),
                "end": round(ts + 0.1, 2),
                "type": "SCENE",
                "reason": "Scene transition"
            })

    # Deduplicate and sort
    cuts.sort(key=lambda x: x["start"])
    merged = []
    for c in cuts:
        if merged and c["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
        else:
            merged.append(c)

    return merged


def build_keep_segments(duration: float, cuts: list, min_len: float) -> list:
    """Invert cut list to produce keep segments."""
    keeps = []
    cursor = 0.0
    for c in cuts:
        if c["start"] > cursor + min_len:
            keeps.append((round(cursor, 3), round(c["start"], 3)))
        cursor = c["end"]
    if duration - cursor > min_len:
        keeps.append((round(cursor, 3), round(duration, 3)))
    return keeps


def get_crop_filter(aspect_ratio: str, filepath: str) -> str:
    """Return crop/scale ffmpeg filter string for target aspect ratio."""
    ratio_map = {
        "16:9": "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        "9:16": "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "1:1":  "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2",
        "4:3":  "scale=1440:1080:force_original_aspect_ratio=decrease,pad=1440:1080:(ow-iw)/2:(oh-ih)/2",
    }
    return ratio_map.get(aspect_ratio, ratio_map["16:9"])


# ─── Background Processing Task ───────────────────────────────────────────────
async def process_videos_task(job_id: str, file_paths: list, settings: CutSettings):
    """Main processing pipeline — runs in background."""
    job = jobs[job_id]

    try:
        job["status"] = "processing"
        job["progress"] = 5
        job["message"] = "Checking FFmpeg..."

        if not ffmpeg_available():
            raise RuntimeError(
                "FFmpeg is not installed on this server. "
                "Install it with: apt-get install ffmpeg"
            )

        all_cuts = []
        total_original_duration = 0.0
        total_kept_duration = 0.0
        concat_inputs = []

        job["progress"] = 10
        job["message"] = f"Analyzing {len(file_paths)} video(s)..."

        per_file_step = 70 // max(len(file_paths), 1)

        for idx, fp in enumerate(file_paths):
            file_label = Path(fp).name
            base_progress = 10 + idx * per_file_step

            job["progress"] = base_progress
            job["message"] = f"[{idx+1}/{len(file_paths)}] Reading: {file_label}"

            # Get duration
            duration = ffprobe_get_duration(fp)
            total_original_duration += duration

            job["progress"] = base_progress + int(per_file_step * 0.2)
            job["message"] = f"[{idx+1}/{len(file_paths)}] Detecting silence: {file_label}"

            # Detect silence
            silent = []
            if settings.remove_silence:
                silent = detect_silence(fp, settings.silence_threshold, 0.3)

            job["progress"] = base_progress + int(per_file_step * 0.5)
            job["message"] = f"[{idx+1}/{len(file_paths)}] Detecting scenes: {file_label}"

            # Detect scenes
            scenes = []
            if settings.scene_detect:
                scenes = detect_scenes(fp)

            job["progress"] = base_progress + int(per_file_step * 0.7)
            job["message"] = f"[{idx+1}/{len(file_paths)}] Building cut list: {file_label}"

            # Build cuts and keep-segments
            cuts = build_cut_list(duration, silent, scenes, settings)
            keeps = build_keep_segments(duration, cuts, settings.min_clip_length)

            for c in cuts:
                c["file"] = file_label
                all_cuts.append(c)

            if not keeps:
                # Fallback: keep entire video if nothing to cut
                keeps = [(0.0, duration)]

            # Trim each keep segment and write temp clip
            clip_paths = []
            for seg_i, (seg_s, seg_e) in enumerate(keeps):
                seg_duration = seg_e - seg_s
                if seg_duration < 0.1:
                    continue
                clip_out = UPLOAD_DIR / f"{job_id}_clip_{idx}_{seg_i}.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(seg_s),
                    "-i", fp,
                    "-t", str(seg_duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero",
                    str(clip_out)
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg trim failed for {file_label} segment {seg_i}: "
                        f"{stderr.decode()[-500:]}"
                    )
                clip_paths.append(str(clip_out))
                total_kept_duration += seg_duration

            concat_inputs.extend(clip_paths)

        job["progress"] = 82
        job["message"] = "Concatenating clips..."

        # Limit to target_duration if needed
        if settings.target_duration > 0 and total_kept_duration > settings.target_duration:
            # Simple trim: keep only clips up to target duration
            trimmed_inputs = []
            running = 0.0
            for cp in concat_inputs:
                d = ffprobe_get_duration(cp)
                if running + d <= settings.target_duration:
                    trimmed_inputs.append(cp)
                    running += d
                else:
                    remaining = settings.target_duration - running
                    if remaining > 0.5:
                        trimmed_cp = str(Path(cp).with_suffix("")) + "_trim.mp4"
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", cp,
                            "-t", str(remaining),
                            "-c", "copy",
                            trimmed_cp
                        ]
                        proc = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await proc.communicate()
                        trimmed_inputs.append(trimmed_cp)
                    break
            concat_inputs = trimmed_inputs

        # Write concat list file
        concat_list_path = UPLOAD_DIR / f"{job_id}_concat.txt"
        with open(concat_list_path, "w") as f:
            for cp in concat_inputs:
                f.write(f"file '{cp}'\n")

        # Final concat + optional crop
        ext = settings.output_format if settings.output_format in ("mp4", "webm", "mov") else "mp4"
        output_filename = f"{job_id}_output.{ext}"
        output_path = OUTPUT_DIR / output_filename

        crop_filter = get_crop_filter(settings.aspect_ratio, concat_inputs[0] if concat_inputs else "")

        job["progress"] = 88
        job["message"] = "Rendering final video..."

        if len(concat_inputs) == 1:
            # Single clip — just copy/transcode
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_inputs[0],
                "-vf", crop_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path)
            ]
        else:
            # Multi-clip concat
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_path),
                "-vf", crop_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path)
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Final render failed: {stderr.decode()[-600:]}")

        job["progress"] = 98
        job["message"] = "Finalizing..."

        # Cleanup temp clips
        for cp in concat_inputs:
            try:
                os.remove(cp)
            except Exception:
                pass
        try:
            os.remove(str(concat_list_path))
        except Exception:
            pass

        final_duration = ffprobe_get_duration(str(output_path))
        time_saved = max(0.0, total_original_duration - final_duration)

        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Complete!"
        job["output_file"] = output_filename
        job["cuts"] = all_cuts
        job["stats"] = {
            "cuts_count": len(all_cuts),
            "original_duration": round(total_original_duration, 1),
            "final_duration": round(final_duration, 1),
            "time_saved": round(time_saved, 1),
            "videos_processed": len(file_paths)
        }

    except Exception as exc:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = str(exc)
        # Cleanup uploads on failure
        for fp in file_paths:
            try:
                os.remove(fp)
            except Exception:
                pass


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "CLIPFORGE API", "status": "online", "ffmpeg": ffmpeg_available()}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg_available": ffmpeg_available(),
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR)
    }


@app.post("/upload")
async def upload_videos(files: List[UploadFile] = File(...)):
    """
    Upload one or more video files.
    Returns list of saved file IDs.
    """
    allowed_types = {
        "video/mp4", "video/quicktime", "video/x-msvideo",
        "video/x-matroska", "video/webm", "video/x-m4v", "video/avi"
    }
    MAX_SIZE_MB = 500
    saved = []

    for f in files:
        # Validate MIME type
        content_type = f.content_type or ""
        if not content_type.startswith("video/") and content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type} for file {f.filename}"
            )

        file_id = str(uuid.uuid4())
        suffix = Path(f.filename).suffix.lower() or ".mp4"
        dest = UPLOAD_DIR / f"{file_id}{suffix}"

        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := await f.read(1024 * 1024):  # 1MB chunks
                    size += len(chunk)
                    if size > MAX_SIZE_MB * 1024 * 1024:
                        out.close()
                        os.remove(dest)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File {f.filename} exceeds {MAX_SIZE_MB}MB limit"
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

        saved.append({
            "file_id": file_id,
            "filename": f.filename,
            "path": str(dest),
            "size_mb": round(size / 1024 / 1024, 2)
        })

    return {"uploaded": saved}


class ProcessRequest(BaseModel):
    file_ids: List[str]
    settings: CutSettings

@app.post("/process")
async def process_videos(
    background_tasks: BackgroundTasks,
    body: ProcessRequest,
):
    """
    Start an AI auto-cut processing job.
    Returns a job_id to poll for status.
    """
    file_ids = body.file_ids
    settings = body.settings
    if not file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")

    # Resolve file paths
    file_paths = []
    for fid in file_ids:
        # Find matching file (any extension)
        matches = list(UPLOAD_DIR.glob(f"{fid}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"File not found: {fid}")
        file_paths.append(str(matches[0]))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued...",
        "cuts": [],
        "output_file": None,
        "stats": {}
    }

    background_tasks.add_task(process_videos_task, job_id, file_paths, settings)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    """Poll processing job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{filename}")
def download_file(filename: str):
    """Download the processed output video."""
    # Security: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    media_type_map = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime"
    }
    media_type = media_type_map.get(file_path.suffix, "video/mp4")

    return FileResponse(
        path=str(file_path),
        filename=f"clipforge_output{file_path.suffix}",
        media_type=media_type
    )


@app.delete("/cleanup/{job_id}")
def cleanup_job(job_id: str):
    """Delete job files and remove from store."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.get("output_file"):
        try:
            os.remove(OUTPUT_DIR / job["output_file"])
        except Exception:
            pass

    del jobs[job_id]
    return {"deleted": job_id}
