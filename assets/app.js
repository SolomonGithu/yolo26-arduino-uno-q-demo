(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("view");
  const ctx = canvas.getContext("2d");
  const els = {
    banner: $("banner"),
    videos: $("videos"),
    reprocess: $("reprocess"),
    loop: $("loop"),
    refresh: $("refresh"),
    overlay: $("overlay"),
    count: $("count"),
    unique: $("unique"),
    infer: $("infer"),
    stop: $("stop"),
  };

  const state = { videos: [], selected: null, modelFound: true, active: false };

  function setBanner(...parts) {
    els.banner.replaceChildren();
    if (!parts.length) {
      els.banner.hidden = true;
      return;
    }
    for (const part of parts) {
      if (typeof part === "string") {
        els.banner.append(part);
      } else {
        const code = document.createElement("code");
        code.textContent = part.code;
        els.banner.append(code);
      }
    }
    els.banner.dataset.kind = "error";
    els.banner.hidden = false;
  }

  function setOverlay(text, { clear = false } = {}) {
    if (!text) {
      els.overlay.hidden = true;
      return;
    }
    els.overlay.textContent = text;
    els.overlay.classList.toggle("is-clear", clear);
    els.overlay.hidden = false;
  }

  function formatDuration(seconds) {
    if (seconds == null) return "";
    const s = Math.round(seconds);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function resetReadout() {
    els.count.textContent = "0";
    els.unique.textContent = "0";
    els.infer.textContent = "–";
  }

  function clearCanvas() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function halt(message) {
    state.active = false;
    latest = null;
    els.stop.disabled = true;
    resetReadout();
    clearCanvas();
    setOverlay(message, { clear: true });
  }

  if (!window.io) {
    setBanner(
      "Could not load the socket.io client. Put socket.io.min.js in ",
      { code: "assets/libs/" },
      " or connect this computer to the internet."
    );
    setOverlay("Not connected");
    return;
  }

  const socket = io();

  socket.on("connect", () => {
    socket.emit("list", {});
    if (els.stop.disabled) setOverlay(state.selected ? "" : "Pick a video to start");
  });

  socket.on("disconnect", () => {
    els.stop.disabled = true;
    setOverlay("Lost the connection to the board. Reconnecting…");
  });

  socket.on("library", (data) => {
    state.videos = data.videos || [];
    state.modelFound = !!data.model_found;
    renderVideos();

    if (state.modelFound) {
      setBanner();
    } else {
      setBanner(
        "Model not found. Copy ",
        { code: data.model_file || "your .eim file" },
        " into the app's ",
        { code: "models/" },
        " folder, or set ",
        { code: "MODEL_FILE" },
        " in python/main.py."
      );
    }
  });

  socket.on("status", (s) => {
    switch (s.state) {
      case "loading":
        state.active = true;
        els.stop.disabled = false;
        setOverlay(s.note || "Starting the model…");
        break;
      case "processing":
        if (state.active) els.stop.disabled = false;
        break;
      case "replaying":
        if (!state.active) break;
        els.stop.disabled = false;
        showReplay(s.metrics);
        break;
      case "finished":
        state.active = false;
        els.stop.disabled = true;
        setOverlay(
          `Finished. ${s.unique} unique vehicle${s.unique === 1 ? "" : "s"} tracked. Pick a video to play it again.`,
          { clear: true }
        );
        break;
      case "stopped":
        if (!state.active) halt("Stopped. Pick a video to play again.");
        break;
      case "error":
        state.active = false;
        els.stop.disabled = true;
        setOverlay(s.message);
        break;
    }
  });

  function showReplay(m) {
    if (m && m.infer_avg_ms != null) els.infer.textContent = `avg ${Math.round(m.infer_avg_ms)} ms`;
  }

  let latest = null;
  let drawing = false;

  socket.on("frame", (data) => {
    if (!state.active) return;
    latest = data;
    if (!drawing) drawLatest();
  });

  function drawLatest() {
    const data = latest;
    latest = null;
    if (!data) return;
    drawing = true;

    const img = new Image();
    img.onload = () => {
      if (state.active) paint(data, img);
      drawing = false;
      if (latest) drawLatest();
    };
    img.onerror = () => {
      drawing = false;
    };
    img.src = "data:image/jpeg;base64," + data.image;
  }

  function colorFor(id) {
    if (id == null) return "#f2b632";
    return `hsl(${(id * 137.508) % 360} 85% 60%)`;
  }

  function paint(data, img) {
    if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    }
    const W = canvas.width;
    const H = canvas.height;
    ctx.drawImage(img, 0, 0, W, H);

    const lineWidth = Math.max(2, W / 320);
    const fontSize = Math.max(12, Math.round(W / 44));
    ctx.lineWidth = lineWidth;
    ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";

    for (const t of data.tracks) {
      const x = t.x * W;
      const y = t.y * H;
      const color = colorFor(t.id);
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, t.w * W, t.h * H);

      const text = t.id == null ? t.label : `${t.label} #${t.id}`;
      const tagW = ctx.measureText(text).width + 10;
      const tagH = fontSize + 6;
      const tagY = y - tagH >= 0 ? y - tagH : y;
      ctx.fillStyle = color;
      ctx.fillRect(x, tagY, tagW, tagH);
      ctx.fillStyle = "#101418";
      ctx.fillText(text, x + 5, tagY + tagH / 2 + 1);
    }

    if (!els.overlay.hidden) setOverlay("");

    els.count.textContent = String(data.count);
    els.unique.textContent = String(data.unique);

    if (data.phase === "processing" && data.metrics) {
      const m = data.metrics;
      els.infer.textContent = `${m.infer_ms} ms (avg ${m.infer_avg_ms})`;
    }
  }

  function renderVideos() {
    els.videos.replaceChildren();

    if (!state.videos.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.append("No videos found. Copy .mp4 files into the app's ");
      const code = document.createElement("code");
      code.textContent = "sample_videos/";
      li.append(code, " folder, then choose Refresh list.");
      els.videos.append(li);
      return;
    }

    for (const video of state.videos) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "video-card";
      button.setAttribute("aria-pressed", String(video.name === state.selected));
      button.disabled = !state.modelFound;

      const thumb = document.createElement("img");
      thumb.className = "thumb";
      thumb.alt = "";
      if (video.thumb) thumb.src = "data:image/jpeg;base64," + video.thumb;

      const meta = document.createElement("span");
      meta.className = "meta";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = video.name;
      name.title = video.name;
      const len = document.createElement("span");
      len.className = "len";
      len.textContent = formatDuration(video.duration);
      if (video.processed) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = "Detected";
        badge.title = "Detection is saved. Replay starts right away.";
        len.append(badge);
      }
      meta.append(name, len);

      button.append(thumb, meta);
      button.addEventListener("click", () => play(video.name));
      li.append(button);
      els.videos.append(li);
    }
  }

  function play(name) {
    state.selected = name;
    renderVideos();
    resetReadout();
    clearCanvas();
    latest = null;
    state.active = true;
    setBanner();
    els.stop.disabled = false;
    setOverlay("Starting the model…");
    socket.emit("play", {
      video: name,
      loop: els.loop.checked,
      reprocess: els.reprocess.checked,
    });
  }

  els.stop.addEventListener("click", () => {
    halt("Stopping…");
    socket.emit("stop", {});
  });
  els.refresh.addEventListener("click", () => socket.emit("list", {}));

  clearCanvas();
})();
