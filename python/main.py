"""Vehicle tracking demo for the Arduino UNO Q (Arduino App Lab).

Pick a video from the ``sample_videos/`` folder in the web UI. Every frame is run
through your Edge Impulse YOLO26 model (a Linux ``.eim`` file) and the tracked
objects, with their persistent IDs, are streamed to the browser together with
the frame. The browser draws the boxes and shows "Vehicles in view".

No bricks are used: the app does not need a camera. Video files are fed to the
model through a small built-in client for the .eim file (no extra Python
packages needed), which also returns the object tracking output.
"""

import base64
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eim_client import EimRunner, Preprocessor, normalise_mode  # noqa: E402

# --------------------------------------------------------------------------
# Settings you may want to change
# --------------------------------------------------------------------------

# Name of your Edge Impulse model file. Copy the .eim into the app's models/
# folder and put its file name here.
MODEL_FILE = "modelfile.eim"

# Only count/draw these classes. An empty list means every class in the model,
# e.g. COUNT_LABELS = ["car", "truck"].
COUNT_LABELS: list = []

# Fallback resize mode if the model does not report one. It must match the
# "Resize mode" of the Image block in Edge Impulse Studio (Impulse design):
#   "squash" | "fit-shortest" | "fit-longest"
# If boxes look stretched or shifted, this is the setting to check.
RESIZE_MODE = "squash"

# Detection runs once per video and the result is saved (in .cache/results),
# then the video is replayed from the saved result at full speed. This is how
# many video frames per second are run through the model. None: every frame
# (best tracking, slowest). A number such as 10 analyses every 3rd frame of a
# 30 fps video: about 3x faster; boxes in between are interpolated. Tracking
# can get worse for fast-moving vehicles, so raise it again if IDs flicker.
ANALYSIS_FPS = None

# Replay never sends more frames per second than this to the browser.
REPLAY_MAX_FPS = 30

# Minimum confidence for a detection (0..1). The model has its own value from
# Studio (see the "thresholds" line in the app log); this overrides it while
# the app runs. None keeps the model's value.
MIN_SCORE = 0.4

# At the first playback of a video the app tries the model on a few frames with
# different image preparations (resize mode, colour order) and uses the one
# that finds vehicles. The result is logged. False: always use the settings
# the model reports.
AUTO_PROBE = True

STREAM_WIDTH = 800  # saved/replayed frames are scaled down to this width
JPEG_QUALITY = 75
THUMB_WIDTH = 240

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = APP_DIR / "sample_videos"
MODELS_DIR = APP_DIR / "models"
CACHE_VERSION = 2


def _cache_root():
    """Folder for saved detection results (falls back to the temp folder)."""
    for base in (APP_DIR / ".cache" / "results", Path(tempfile.gettempdir()) / "vehicle-tracking-results"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write-test"
            probe.write_text("x")
            probe.unlink()
            return base
        except OSError:
            continue
    raise RuntimeError("no writable folder for saved results")

ui = WebUI()


def find_model():
    """Return the path of the model file, or None if it is not in models/."""
    path = MODELS_DIR / MODEL_FILE
    return path if path.is_file() else None


def ensure_executable(path):
    """EIM files are Linux executables; make sure the exec bit is set."""
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Sample videos
# --------------------------------------------------------------------------

_video_info_cache = {}


def _jpeg_b64(image, quality):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _fps_of(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and math.isfinite(fps) and fps > 1 else 30.0


def describe_video(path):
    stat = path.stat()
    key = (path.name, stat.st_mtime_ns, stat.st_size)
    if key in _video_info_cache:
        return _video_info_cache[key]

    info = {"name": path.name, "thumb": None, "duration": None}
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                info["duration"] = round(total / _fps_of(cap), 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 10)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if ok:
                scale = THUMB_WIDTH / frame.shape[1]
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                info["thumb"] = _jpeg_b64(small, 70)
    finally:
        cap.release()

    _video_info_cache[key] = info
    return info


def list_videos(model_path=None):
    VIDEOS_DIR.mkdir(exist_ok=True)
    paths = [
        p
        for p in sorted(VIDEOS_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith(".")
    ]
    videos = []
    for path in paths:
        info = dict(describe_video(path))
        meta = None
        if model_path is not None:
            try:
                meta = _load_meta(_cache_dir(path, model_path))
            except OSError:
                meta = None
        info["processed"] = meta is not None
        info["metrics"] = meta["metrics"] if meta else None
        videos.append(info)
    return videos


def resolve_video(name):
    """Map a name from the browser to a file inside sample_videos/ (nothing else)."""
    candidate = VIDEOS_DIR / Path(str(name)).name
    if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
        return candidate
    return None


# --------------------------------------------------------------------------
# Playback session
# --------------------------------------------------------------------------

_lock = threading.Lock()
_thread = None
_stop_event = None


def send_status(state, **extra):
    ui.send_message("status", {"state": state, **extra})


def send_library():
    model = find_model()
    ui.send_message(
        "library",
        {
            "videos": list_videos(model),
            "model_found": model is not None,
            "model_file": MODEL_FILE,
        },
    )


def _extract_tracks(result, preprocessor):
    """Boxes of one inference result, ready for the browser.

    The tracking is done inside the model file: with object tracking enabled in
    Edge Impulse Studio the result holds ``object_tracking``, the confirmed
    tracks with their IDs. Without it the plain detections are shown, no IDs.
    """
    tracked = result.get("object_tracking")
    items = tracked if tracked is not None else (result.get("bounding_boxes") or [])

    tracks = []
    for item in items:
        label = item.get("label", "")
        if COUNT_LABELS and label not in COUNT_LABELS:
            continue
        object_id = item.get("object_id", item.get("id")) if tracked is not None else None
        x, y, w, h = preprocessor.to_frame(item["x"], item["y"], item["width"], item["height"])
        tracks.append(
            {
                "id": object_id,
                "label": label,
                "x": round(x, 4),
                "y": round(y, 4),
                "w": round(w, 4),
                "h": round(h, 4),
            }
        )
    return tracks


def _object_detection_block(model_info):
    params = model_info.get("model_parameters", {})
    for source in (model_info.get("thresholds"), params.get("thresholds")):
        for block in source or []:
            if block.get("type") == "object_detection":
                return block.get("id")
    return None


def _start_runner(model_path, override=None, min_score="default"):
    """Start the model; return (runner, preprocessor, mode, model_info).

    override: optional (resize_mode, bgr) that replaces what the model reports.
    """
    if min_score == "default":
        min_score = MIN_SCORE
    runner = EimRunner(model_path, on_output=lambda line: print(f"[eim] {line}", flush=True))
    try:
        model_info = runner.init()
        params = model_info.get("model_parameters", {})
        mode = normalise_mode(params.get("image_resize_mode")) or RESIZE_MODE
        bgr = False
        if override:
            mode, bgr = override
        preprocessor = Preprocessor(
            params["image_input_width"],
            params["image_input_height"],
            params.get("image_channel_count", 3),
            mode,
            bgr,
        )
        block = _object_detection_block(model_info)
        if min_score is not None and block is not None:
            try:
                runner.set_threshold(block, min_score=min_score)
            except Exception as exc:
                print(f"[vehicle-tracking] could not set min_score: {exc}", flush=True)
    except Exception:
        runner.stop()
        raise
    print(f"[vehicle-tracking] model parameters: {json.dumps(params)[:1500]}", flush=True)
    print(
        "[vehicle-tracking] model: "
        f"input={preprocessor.width}x{preprocessor.height}x{preprocessor.channels} "
        f"resize={params.get('image_resize_mode')!r}->{mode} order={'BGR' if bgr else 'RGB'} "
        f"type={params.get('model_type')!r} labels={params.get('labels')!r} "
        f"object_tracking={params.get('has_object_tracking', 'n/a')} "
        f"min_score={min_score} thresholds={model_info.get('thresholds', params.get('thresholds', 'n/a'))}",
        flush=True,
    )
    return runner, preprocessor, mode, model_info


_probe_cache = {}


def _probe(model_path, video_path, stop_event):
    """Find the image preparation that makes the model see vehicles in this video.

    Runs a few frames through the model with every combination of resize mode
    and colour order, at a very low threshold, and logs what each one finds.
    Returns (override or None, notice or None).
    """
    key = (str(model_path), video_path.name, video_path.stat().st_mtime)
    if key in _probe_cache:
        return _probe_cache[key]

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for fraction in (0.2, 0.5):
            if total:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fraction))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    if not frames:
        return None, None

    runner = None
    try:
        runner, pre, declared, info = _start_runner(model_path, min_score=None)
        block = _object_detection_block(info)
        if block is not None:
            try:
                runner.set_threshold(block, min_score=0.05)
            except Exception as exc:
                print(f"[vehicle-tracking] probe: could not lower min_score: {exc}", flush=True)
        modes = [declared] + [m for m in ("squash", "fit-shortest", "fit-longest") if m != declared]
        variants = [(m, False) for m in modes] + [(m, True) for m in modes]
        threshold = MIN_SCORE if MIN_SCORE is not None else 0.5
        rows = []
        for mode, bgr in variants:
            if stop_event.is_set():
                return None, None
            probe_pre = Preprocessor(pre.width, pre.height, pre.channels, mode, bgr)
            good, top = 0, 0.0
            for frame in frames:
                boxes = runner.classify(probe_pre.prepare(frame)).get("result", {}).get("bounding_boxes") or []
                good += sum(1 for b in boxes if b.get("value", 0) >= threshold)
                top = max([top] + [b.get("value", 0) for b in boxes])
            rows.append(((mode, bgr), good, top))
            print(
                f"[vehicle-tracking] probe {mode:12s} {'BGR' if bgr else 'RGB'}: "
                f"{good} detections >= {threshold}, best score {top:.2f}",
                flush=True,
            )
    finally:
        if runner is not None:
            runner.stop()

    base = rows[0]
    best = max(rows, key=lambda r: (r[1], r[2]))
    if best[1] == 0:
        top = max(r[2] for r in rows)
        notice = (
            "The model found no vehicles in test frames of this video with any image preparation"
            + (f" (best score {top:.2f}, below the minimum score {threshold})." if top else ".")
            + " Check that this video resembles the images the model was trained on."
        )
        result = (None, notice)
    elif best[1] > base[1]:
        print(f"[vehicle-tracking] probe: using resize={best[0][0]}, {'BGR' if best[0][1] else 'RGB'} "
              f"(the model's own setting found {base[1]})", flush=True)
        result = (best[0], None)
    else:
        result = (None, None)
    _probe_cache[key] = result
    return result


def _log_result(index, result, tracks):
    """Short line in the app log for the first frames, then every 100th."""
    if index > 3 and index % 100:
        return
    boxes = result.get("bounding_boxes") or []
    tracked = result.get("object_tracking")
    sample = (tracked or boxes or [None])[0]
    print(
        f"[vehicle-tracking] frame {index}: result keys={sorted(result)} "
        f"bounding_boxes={len(boxes)} "
        f"object_tracking={'n/a' if tracked is None else len(tracked)} "
        f"shown={len(tracks)} sample={sample}",
        flush=True,
    )


# --------------------------------------------------------------------------
# Saved results: detect once, replay as often as you like
# --------------------------------------------------------------------------


def _cache_dir(video_path, model_path):
    st, ms = video_path.stat(), model_path.stat()
    raw = "|".join(
        str(x)
        for x in (
            CACHE_VERSION, video_path.name, st.st_size, st.st_mtime_ns,
            model_path.name, ms.st_size, ms.st_mtime_ns,
            ANALYSIS_FPS, MIN_SCORE, sorted(COUNT_LABELS), STREAM_WIDTH,
        )
    )
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", video_path.stem)[:40]
    return _cache_root() / f"{slug}-{hashlib.sha1(raw.encode()).hexdigest()[:12]}"


def _load_meta(cache_dir):
    try:
        meta = json.loads((cache_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    return meta if meta.get("version") == CACHE_VERSION else None


def _lerp(a, b, t):
    return {k: round(a[k] + (b[k] - a[k]) * t, 4) for k in ("x", "y", "w", "h")}


def _timeline(analysed, n_frames):
    """Boxes for every frame from the analysed ones: {frame index: tracks}.

    Frames that were not analysed take the boxes of the nearest analysed
    frame, moved towards the other neighbour for vehicles with the same ID.
    """
    keys = sorted(analysed)
    timeline = {}
    pos = 0
    for i in range(n_frames):
        while pos + 1 < len(keys) and keys[pos + 1] <= i:
            pos += 1
        a = keys[pos]
        b = keys[pos + 1] if pos + 1 < len(keys) else None
        if i == a or b is None:
            timeline[i] = [dict(t) for t in analysed[a]]
            continue
        t = (i - a) / (b - a)
        base, other = (analysed[a], analysed[b]) if t < 0.5 else (analysed[b], analysed[a])
        other_by_id = {x["id"]: x for x in other if x["id"] is not None}
        out = []
        for track in base:
            track = dict(track)
            mate = other_by_id.get(track["id"]) if track["id"] is not None else None
            if mate is not None:
                pa, pb = (track, mate) if base is analysed[a] else (mate, track)
                track.update(_lerp(pa, pb, t))
            out.append(track)
        timeline[i] = out
    return timeline


def _summarise(infer_ms, cycle_ms, elapsed, stride):
    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    return {
        "analysed": len(infer_ms),
        "stride": stride,
        "infer_avg_ms": avg(infer_ms),
        "infer_min_ms": round(min(infer_ms), 1) if infer_ms else None,
        "infer_max_ms": round(max(infer_ms), 1) if infer_ms else None,
        "cycle_avg_ms": avg(cycle_ms),
        "det_fps": round(len(infer_ms) / elapsed, 2) if elapsed > 0 else None,
        "elapsed_s": round(elapsed, 1),
    }


def _process(video_path, model_path, cache_dir, stop_event, override, notice):
    """Run the model over the whole video and save frames + boxes.

    Streams every analysed frame to the browser as it goes, with live
    inference-time and speed numbers. Returns the saved metadata, or None if
    it was stopped or failed (an error status has been sent in that case).
    """
    tmp = cache_dir.with_name(cache_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "frames").mkdir(parents=True)

    runner = None
    cap = None
    done = False
    try:
        try:
            runner, pre, mode, info = _start_runner(model_path, override)
        except Exception as exc:
            send_status("error", message=f"Could not start the model: {exc}")
            return None
        if stop_event.is_set():
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            send_status("error", message=f"Could not open {video_path.name}. Try an H.264 .mp4.")
            return None
        fps = _fps_of(cap)
        est_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, round(fps / ANALYSIS_FPS)) if ANALYSIS_FPS else 1
        model_info = {
            "name": info.get("project", {}).get("name") or model_path.name,
            "input": f"{pre.width}x{pre.height}",
            "resize": mode,
        }
        send_status("processing", video=video_path.name, total=est_total, model=model_info, notice=notice)

        analysed, infer_ms, cycle_ms = {}, [], []
        started = time.monotonic()
        last_done = started
        size = None
        index = 0

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            small = frame
            if frame.shape[1] > STREAM_WIDTH:
                scale = STREAM_WIDTH / frame.shape[1]
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError("could not encode a frame")
            jpeg = buf.tobytes()
            (tmp / "frames" / f"{index:06d}.jpg").write_bytes(jpeg)
            size = (small.shape[1], small.shape[0])

            if index % stride == 0:
                features = pre.prepare(frame)
                t0 = time.perf_counter()
                response = runner.classify(features)
                round_trip = (time.perf_counter() - t0) * 1000
                if stop_event.is_set():
                    return None

                result = response.get("result", {})
                tracks = _extract_tracks(result, pre)
                _log_result(index + 1, result, tracks)
                analysed[index] = tracks

                timing = response.get("timing") or {}
                model_ms = sum(timing.get(k, 0) for k in ("dsp", "classification", "anomaly"))
                infer_ms.append(float(model_ms) if model_ms else round_trip)
                now = time.monotonic()
                cycle_ms.append((now - last_done) * 1000)
                last_done = now
                elapsed = now - started
                seen = {t["id"] for tr in analysed.values() for t in tr if t["id"] is not None}
                speed = (index + 1) / elapsed if elapsed > 0 else 0
                ui.send_message(
                    "frame",
                    {
                        "phase": "processing",
                        "image": base64.b64encode(jpeg).decode("ascii"),
                        "tracks": tracks,
                        "count": len(tracks),
                        "unique": len(seen),
                        "frame": index + 1,
                        "total": est_total,
                        "round": 1,
                        "metrics": {
                            "infer_ms": round(infer_ms[-1]),
                            "infer_avg_ms": round(sum(infer_ms) / len(infer_ms)),
                            "round_trip_ms": round(round_trip),
                            "det_fps": round(len(infer_ms) / elapsed, 2),
                            "analysed": len(infer_ms),
                            "eta_s": round((est_total - index - 1) / speed) if est_total and speed else None,
                        },
                    },
                )
            index += 1

        if stop_event.is_set() or not analysed:
            if not stop_event.is_set():
                send_status("error", message=f"No frames could be read from {video_path.name}.")
            return None

        n_frames = index
        timeline = _timeline(analysed, n_frames)
        seen, frames_meta = set(), []
        for i in range(n_frames):
            seen.update(t["id"] for t in timeline[i] if t["id"] is not None)
            frames_meta.append({"tracks": timeline[i], "unique": len(seen)})

        meta = {
            "version": CACHE_VERSION,
            "video": video_path.name,
            "fps": fps,
            "total": n_frames,
            "size": size,
            "model": model_info,
            "notice": notice,
            "metrics": {**_summarise(infer_ms, cycle_ms, time.monotonic() - started, stride), "unique": len(seen)},
            "frames": frames_meta,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
        same_video = re.compile(re.escape(cache_dir.name.rsplit("-", 1)[0]) + r"-[0-9a-f]{12}")
        for old in cache_dir.parent.iterdir():
            if same_video.fullmatch(old.name):  # earlier results for this video
                shutil.rmtree(old, ignore_errors=True)
        tmp.rename(cache_dir)
        done = True
        print(
            f"[vehicle-tracking] processed {video_path.name}: {len(infer_ms)} frames analysed, "
            f"inference avg {meta['metrics']['infer_avg_ms']} ms, {meta['metrics']['det_fps']} fps, "
            f"{meta['metrics']['unique']} unique vehicles",
            flush=True,
        )
        return meta
    except Exception as exc:
        if not stop_event.is_set():
            send_status("error", message=f"Detection failed: {exc}")
        return None
    finally:
        if cap is not None:
            cap.release()
        if runner is not None:
            runner.stop()
        if not done:
            shutil.rmtree(tmp, ignore_errors=True)


def _replay(meta, cache_dir, loop, stop_event, cached):
    """Play the saved frames with their boxes at the video's own speed."""
    n = meta["total"]
    fps = meta["fps"]
    step = max(1, math.ceil(fps / REPLAY_MAX_FPS))
    frames_dir = cache_dir / "frames"
    round_no = 0
    unique = 0

    while not stop_event.is_set():
        round_no += 1
        send_status(
            "replaying",
            video=meta["video"],
            loop=loop,
            round=round_no,
            cached=cached,
            model=meta["model"],
            notice=meta.get("notice"),
            metrics=meta["metrics"],
        )
        started = time.monotonic()
        smoothed, last_t = 0.0, None
        for i in range(0, n, step):
            wait = started + i / fps - time.monotonic()
            if wait > 0 and stop_event.wait(wait):
                return
            if stop_event.is_set():
                return
            try:
                jpeg = (frames_dir / f"{i:06d}.jpg").read_bytes()
            except OSError:
                send_status("error", message="Saved frames are missing. Run detection again.")
                return
            entry = meta["frames"][i]
            unique = entry["unique"]
            now = time.perf_counter()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                smoothed = inst if smoothed == 0 else 0.8 * smoothed + 0.2 * inst
            last_t = now
            ui.send_message(
                "frame",
                {
                    "phase": "replay",
                    "image": base64.b64encode(jpeg).decode("ascii"),
                    "tracks": entry["tracks"],
                    "count": len(entry["tracks"]),
                    "unique": unique,
                    "frame": i + 1,
                    "total": n,
                    "round": round_no,
                    "fps": round(smoothed, 1),
                    "metrics": meta["metrics"],
                },
            )
        if not loop:
            send_status("finished", video=meta["video"], unique=unique, metrics=meta["metrics"])
            return


def _job(video_path, loop, reprocess, stop_event):
    """Worker thread: detect once (if not saved yet), then replay."""
    model_path = find_model()
    if model_path is None:
        send_status("error", message=f"Model not found: models/{MODEL_FILE}. Copy it there or change MODEL_FILE in python/main.py.")
        return
    try:
        cache_dir = _cache_dir(video_path, model_path)
        meta = None if reprocess else _load_meta(cache_dir)
        cached = meta is not None
        if meta is None:
            send_status("loading", video=video_path.name)
            ensure_executable(model_path)
            override, notice = None, None
            if AUTO_PROBE:
                send_status(
                    "loading",
                    video=video_path.name,
                    note="Checking how the model sees this video. This takes a moment the first time.",
                )
                try:
                    override, notice = _probe(model_path, video_path, stop_event)
                except Exception as exc:
                    print(f"[vehicle-tracking] probe failed: {exc}", flush=True)
                if stop_event.is_set():
                    return
                send_status("loading", video=video_path.name)
            meta = _process(video_path, model_path, cache_dir, stop_event, override, notice)
            if meta is None:
                return
        _replay(meta, cache_dir, loop, stop_event, cached)
    except Exception as exc:
        if not stop_event.is_set():
            send_status("error", message=f"Playback failed: {exc}")


def _stop_locked():
    global _thread, _stop_event
    if _stop_event is not None:
        _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=15)
    _thread = None
    _stop_event = None


def start_playback(name, loop, reprocess=False):
    global _thread, _stop_event
    video = resolve_video(name)
    with _lock:
        _stop_locked()
        if video is None:
            send_status("error", message=f"Video not found in sample_videos/: {name}")
            return
        _stop_event = threading.Event()
        _thread = threading.Thread(target=_job, args=(video, loop, reprocess, _stop_event), daemon=True)
        _thread.start()


def stop_playback():
    with _lock:
        _stop_locked()
    send_status("stopped")


# --------------------------------------------------------------------------
# Web UI wiring
# --------------------------------------------------------------------------


def _in_thread(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()


def on_connect(sid, *_):
    _in_thread(send_library)


def on_list(sid, data=None):
    _in_thread(send_library)


def on_play(sid, data=None):
    data = data or {}
    _in_thread(
        start_playback,
        data.get("video"),
        bool(data.get("loop", True)),
        bool(data.get("reprocess", False)),
    )


def on_stop(sid, data=None):
    _in_thread(stop_playback)


ui.on_connect(on_connect)
ui.on_message("list", on_list)
ui.on_message("play", on_play)
ui.on_message("stop", on_stop)

print(f"[vehicle-tracking] videos: {VIDEOS_DIR}  model: {MODELS_DIR / MODEL_FILE}")

App.run()
