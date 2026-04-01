"""
CLIPFORGE Backend — FastAPI + FFmpeg AI Video Editor
"""

import os, uuid, json, asyncio, subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="CLIPFORGE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,        # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Dirs (/tmp is always writable on Render free tier) ─────────────────────────
UPLOAD_DIR = Path("/tmp/clipforge/uploads")
OUTPUT_DIR = Path("/tmp/clipforge/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: dict = {}

# ── Schemas ───────────────────────────────────────────────────────────────────
class CutSettings(BaseModel):
    mode: str = "smart"
    remove_silence: bool = True
    scene_detect: bool = True
    jump_cut: bool = False
    silence_threshold: float = 40.0
    min_clip_length: float = 3.0
    target_duration: float = 60.0
    output_format: str = "mp4"
    aspect_ratio: str = "16:9"

class ProcessRequest(BaseModel):
    file_ids: List[str]
    settings: CutSettings

# ── Helpers ───────────────────────────────────────────────────────────────────
def ffmpeg_ok() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True, timeout=30
    )
    return float(json.loads(r.stdout)["format"]["duration"])

def detect_silence(path: str, noise_db: float, min_dur: float = 0.3) -> list:
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-af",
         f"silencedetect=noise=-{noise_db}dB:d={min_dur}", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120
    )
    ranges, start = [], None
    for line in r.stderr.split("\n"):
        if "silence_start" in line:
            try: start = float(line.split("silence_start: ")[1].strip())
            except: pass
        elif "silence_end" in line and start is not None:
            try:
                end = float(line.split("silence_end: ")[1].split(" ")[0])
                ranges.append((round(start, 2), round(end, 2)))
                start = None
            except: pass
    return ranges

def detect_scenes(path: str, threshold: float = 0.4) -> list:
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-vf",
         f"select='gt(scene,{threshold})',showinfo", "-vsync", "vfr", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120
    )
    times = []
    for line in r.stderr.split("\n"):
        if "pts_time" in line and "showinfo" in line:
            try: times.append(round(float(line.split("pts_time:")[1].split(" ")[0]), 2))
            except: pass
    return times

def build_cuts(duration, silences, scenes, settings):
    cuts = []
    if settings.remove_silence:
        for s, e in silences:
            if e - s >= settings.min_clip_length:
                cuts.append({"start": s, "end": e, "type": "SILENCE",
                             "reason": f"Silent pause ({e-s:.1f}s)"})
    if settings.scene_detect and settings.mode in ("smart", "highlight"):
        for ts in scenes:
            cuts.append({"start": round(ts-0.1,2), "end": round(ts+0.1,2),
                         "type": "SCENE", "reason": "Scene transition"})
    cuts.sort(key=lambda x: x["start"])
    merged = []
    for c in cuts:
        if merged and c["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
        else:
            merged.append(c)
    return merged

def invert_cuts(duration, cuts, min_len):
    keeps, cursor = [], 0.0
    for c in cuts:
        if c["start"] > cursor + min_len:
            keeps.append((round(cursor, 3), round(c["start"], 3)))
        cursor = c["end"]
    if duration - cursor > min_len:
        keeps.append((round(cursor, 3), round(duration, 3)))
    return keeps

def scale_filter(aspect_ratio: str) -> str:
    m = {
        "16:9": "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        "9:16": "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "1:1":  "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2",
        "4:3":  "scale=1440:1080:force_original_aspect_ratio=decrease,pad=1440:1080:(ow-iw)/2:(oh-ih)/2",
    }
    return m.get(aspect_ratio, m["16:9"])

# ── Background task ────────────────────────────────────────────────────────────
async def run_job(job_id: str, file_paths: list, settings: CutSettings):
    job = jobs[job_id]
    try:
        job.update(status="processing", progress=5, message="Checking FFmpeg...")
        if not ffmpeg_ok():
            raise RuntimeError("FFmpeg not found on server")

        all_cuts, concat_inputs = [], []
        total_orig, total_kept = 0.0, 0.0
        step = 70 // max(len(file_paths), 1)

        for idx, fp in enumerate(file_paths):
            label = Path(fp).name
            base = 10 + idx * step

            job.update(progress=base, message=f"[{idx+1}/{len(file_paths)}] Analysing: {label}")
            duration = get_duration(fp)
            total_orig += duration

            job.update(progress=base + int(step*0.25), message=f"[{idx+1}/{len(file_paths)}] Silence detection...")
            silences = detect_silence(fp, settings.silence_threshold) if settings.remove_silence else []

            job.update(progress=base + int(step*0.5), message=f"[{idx+1}/{len(file_paths)}] Scene detection...")
            scenes = detect_scenes(fp) if settings.scene_detect else []

            job.update(progress=base + int(step*0.7), message=f"[{idx+1}/{len(file_paths)}] Building cuts...")
            cuts = build_cuts(duration, silences, scenes, settings)
            keeps = invert_cuts(duration, cuts, settings.min_clip_length) or [(0.0, duration)]

            for c in cuts:
                c["file"] = label
                all_cuts.append(c)

            for si, (ss, se) in enumerate(keeps):
                seg_dur = se - ss
                if seg_dur < 0.1:
                    continue
                out = UPLOAD_DIR / f"{job_id}_c{idx}_{si}.mp4"
                cmd = ["ffmpeg", "-y", "-ss", str(ss), "-i", fp, "-t", str(seg_dur),
                       "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                       "-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero", str(out)]
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                _, err = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"FFmpeg trim failed: {err.decode()[-400:]}")
                concat_inputs.append(str(out))
                total_kept += seg_dur

        job.update(progress=82, message="Concatenating clips...")

        # Trim to target duration
        if settings.target_duration > 0 and total_kept > settings.target_duration:
            trimmed, running = [], 0.0
            for cp in concat_inputs:
                d = get_duration(cp)
                if running + d <= settings.target_duration:
                    trimmed.append(cp); running += d
                else:
                    rem = settings.target_duration - running
                    if rem > 0.5:
                        tp = cp.replace(".mp4", "_t.mp4")
                        cmd = ["ffmpeg", "-y", "-i", cp, "-t", str(rem), "-c", "copy", tp]
                        proc = await asyncio.create_subprocess_exec(
                            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                        await proc.communicate()
                        trimmed.append(tp)
                    break
            concat_inputs = trimmed

        ext = settings.output_format if settings.output_format in ("mp4","webm","mov") else "mp4"
        out_name = f"{job_id}_output.{ext}"
        out_path = OUTPUT_DIR / out_name

        job.update(progress=88, message="Rendering final video...")

        concat_file = UPLOAD_DIR / f"{job_id}_list.txt"
        with open(concat_file, "w") as f:
            for cp in concat_inputs:
                f.write(f"file '{cp}'\n")

        vf = scale_filter(settings.aspect_ratio)

        if len(concat_inputs) == 1:
            cmd = ["ffmpeg", "-y", "-i", concat_inputs[0],
                   "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                   "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out_path)]
        else:
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                   "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                   "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out_path)]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Final render failed: {err.decode()[-600:]}")

        # Cleanup temp clips
        for cp in concat_inputs:
            try: os.remove(cp)
            except: pass
        try: os.remove(str(concat_file))
        except: pass

        final_dur = get_duration(str(out_path))
        job.update(
            status="done", progress=100, message="Complete!",
            output_file=out_name,
            cuts=all_cuts,
            stats={
                "cuts_count": len(all_cuts),
                "original_duration": round(total_orig, 1),
                "final_duration": round(final_dur, 1),
                "time_saved": round(max(0.0, total_orig - final_dur), 1),
                "videos_processed": len(file_paths)
            }
        )
    except Exception as exc:
        job.update(status="error", progress=0, message=str(exc))
        for fp in file_paths:
            try: os.remove(fp)
            except: pass

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "CLIPFORGE API", "status": "online", "ffmpeg": ffmpeg_ok()}

@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg": ffmpeg_ok(),
            "upload_dir": str(UPLOAD_DIR), "output_dir": str(OUTPUT_DIR)}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        ct = f.content_type or ""
        if not ct.startswith("video/"):
            raise HTTPException(415, f"Not a video file: {f.filename} ({ct})")

        fid = str(uuid.uuid4())
        suffix = Path(f.filename or "video.mp4").suffix.lower() or ".mp4"
        dest = UPLOAD_DIR / f"{fid}{suffix}"
        size = 0
        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 500 * 1024 * 1024:
                        out.close()
                        os.remove(dest)
                        raise HTTPException(413, f"{f.filename} exceeds 500MB")
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Upload error: {e}")

        saved.append({"file_id": fid, "filename": f.filename,
                      "size_mb": round(size / 1024 / 1024, 2)})
    return {"uploaded": saved}

@app.post("/process")
async def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    file_paths = []
    for fid in req.file_ids:
        matches = list(UPLOAD_DIR.glob(f"{fid}.*"))
        if not matches:
            raise HTTPException(404, f"File not found: {fid}")
        file_paths.append(str(matches[0]))

    jid = str(uuid.uuid4())
    jobs[jid] = {"job_id": jid, "status": "queued", "progress": 0,
                 "message": "Queued...", "cuts": [], "output_file": None, "stats": {}}
    background_tasks.add_task(run_job, jid, file_paths, req.settings)
    return {"job_id": jid}

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/download/{filename}")
def download(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
    return FileResponse(str(path), filename=f"clipforge{path.suffix}",
                        media_type=types.get(path.suffix, "video/mp4"))

@app.delete("/cleanup/{job_id}")
def cleanup(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs.pop(job_id)
    if job.get("output_file"):
        try: os.remove(OUTPUT_DIR / job["output_file"])
        except: pass
    return {"deleted": job_id}
