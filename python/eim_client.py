"""Client for Edge Impulse Linux models (.eim) and image preparation.

Used by the app (main.py). It speaks the same protocol as
the Edge Impulse Linux Python SDK (edge_impulse_linux/runner.py): the .eim is
started with a Unix socket path as its only argument, requests are plain JSON
with an incrementing "id", and every response ends with a NUL byte. No extra
Python packages are needed.
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np


class EimRunner:
    """Minimal client for an Edge Impulse Linux model (.eim).

    It speaks the same protocol as the Edge Impulse Linux Python SDK, so no
    extra Python package is needed: the .eim is started with a Unix socket
    path, requests are plain JSON and every response ends with a NUL byte.
    """

    def __init__(self, model_path, start_timeout=60, on_output=None, max_output_lines=200):
        self._model_path = str(model_path)
        self._on_output = on_output
        self._max_output_lines = max_output_lines
        self._start_timeout = start_timeout
        self._proc = None
        self._sock = None
        self._tempdir = None
        self._ix = 0
        self._buffer = b""

    def _pump_output(self, proc):
        lines = 0
        for line in proc.stdout:
            lines += 1
            if self._on_output is not None and lines <= self._max_output_lines:
                self._on_output(line.rstrip())

    def init(self):
        self._tempdir = tempfile.mkdtemp(prefix="eim-")
        socket_path = os.path.join(self._tempdir, "runner.sock")
        self._proc = subprocess.Popen(
            [self._model_path, socket_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        # Whatever the model prints (warnings, errors) is passed on to
        # ``on_output``, otherwise it would be lost.
        threading.Thread(target=self._pump_output, args=(self._proc,), daemon=True).start()

        deadline = time.monotonic() + self._start_timeout
        while not os.path.exists(socket_path):
            if self._proc.poll() is not None:
                raise RuntimeError(f"the model exited immediately (code {self._proc.returncode})")
            if time.monotonic() > deadline:
                raise RuntimeError("the model did not start in time")
            time.sleep(0.05)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        return self._send({"hello": 1})

    def classify(self, features):
        return self._send({"classify": features})

    def set_threshold(self, block_id, **values):
        """Change a threshold of the model, e.g. set_threshold(4, min_score=0.3)."""
        if self._sock is None:
            raise RuntimeError("the model is not running")
        self._sock.settimeout(10)
        try:
            self._send({"set_threshold": {"id": block_id, **values}})
        except socket.timeout:
            raise RuntimeError("the model did not answer set_threshold") from None
        finally:
            if self._sock is not None:
                self._sock.settimeout(None)

    def stop(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._tempdir is not None:
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def _send(self, msg):
        if self._sock is None:
            raise RuntimeError("the model is not running")
        self._ix += 1
        msg["id"] = self._ix
        self._sock.sendall(json.dumps(msg, separators=(",", ":")).encode("utf-8"))
        while True:
            resp = self._read_message()
            if resp.get("id") == self._ix:
                break
        if not resp.get("success", False):
            raise RuntimeError(resp.get("error") or "the model reported an error")
        resp.pop("id", None)
        resp.pop("success", None)
        return resp

    def _read_message(self):
        while b"\0" not in self._buffer:
            chunk = self._sock.recv(1 << 16)
            if not chunk:
                raise RuntimeError("the model closed the connection")
            self._buffer += chunk
        raw, self._buffer = self._buffer.split(b"\0", 1)
        return json.loads(raw.decode("utf-8"))


def normalise_mode(value):
    text = str(value or "").lower().replace("_", "-").replace(" ", "-")
    if "squash" in text:
        return "squash"
    if "shortest" in text:
        return "fit-shortest"
    if "longest" in text:
        return "fit-longest"
    return None


class Preprocessor:
    """Resizes frames to the model input and maps boxes back to the frame.

    Box coordinates returned by the model are pixels in the model input
    (``width`` x ``height``). ``to_frame`` converts them to 0..1 fractions of
    the original frame, undoing whichever resize mode was used.
    """

    def __init__(self, width, height, channels, mode, bgr=False):
        self.bgr = bool(bgr)  # True: send blue-green-red instead of red-green-blue
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.mode = mode
        self._layouts = {}
        self._layout = None

    def _layout_for(self, frame_w, frame_h):
        key = (frame_w, frame_h)
        if key in self._layouts:
            return self._layouts[key]
        w, h = self.width, self.height
        if self.mode == "squash":
            layout = (0, 0, w, h)
        elif self.mode == "fit-shortest":
            scale = max(w / frame_w, h / frame_h)
            rw = max(w, round(frame_w * scale))
            rh = max(h, round(frame_h * scale))
            layout = (-((rw - w) // 2), -((rh - h) // 2), rw, rh)
        else:  # fit-longest: letterbox
            scale = min(w / frame_w, h / frame_h)
            rw = min(w, max(1, round(frame_w * scale)))
            rh = min(h, max(1, round(frame_h * scale)))
            layout = ((w - rw) // 2, (h - rh) // 2, rw, rh)
        self._layouts[key] = layout
        return layout

    def prepare(self, frame_bgr):
        """Return the flat feature list the EIM expects for this frame."""
        frame_h, frame_w = frame_bgr.shape[:2]
        ox, oy, rw, rh = self._layout_for(frame_w, frame_h)
        self._layout = (ox, oy, rw, rh)
        w, h = self.width, self.height

        if self.mode == "squash":
            img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
            if self.mode == "fit-shortest":
                img = resized[-oy : -oy + h, -ox : -ox + w]
            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)
                img[oy : oy + rh, ox : ox + rw] = resized

        if self.channels == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint32)
            packed = (gray << 16) | (gray << 8) | gray
        else:
            rgb = (img if self.bgr else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype(np.uint32)
            packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        return packed.reshape(-1).tolist()

    def to_frame(self, x, y, width, height):
        """Model-space pixel box -> (x, y, w, h) as fractions of the frame."""
        ox, oy, rw, rh = self._layout
        x0 = max(0.0, (x - ox) / rw)
        y0 = max(0.0, (y - oy) / rh)
        x1 = min(1.0, (x + width - ox) / rw)
        y1 = min(1.0, (y + height - oy) / rh)
        return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)
