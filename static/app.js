// Front-end logic. The camera runs in the BROWSER and is shown live at full speed.
// Small frames are sent to the Python server, which returns only face boxes + statuses;
// the browser draws the ID and status tags on a transparent canvas over the video.
const $ = id => document.getElementById(id);
const COLORS = {"MASK ON":"#1fa755","NO MASK":"#ef4444","MASK BELOW NOSE":"#f59e0b",
                "MASK ON CHIN":"#f59e0b","CHECK MASK":"#d4a017"};
const DARK_TEXT = new Set(["#f59e0b", "#d4a017"]);
const SID = Math.random().toString(36).slice(2) + Date.now().toString(36);
const SEND_W = 320, OVERLAY_W = 800, IN_FLIGHT = 2;
const WARN_ICON = '<svg viewBox="0 0 24 24" width="34" height="34"><path fill="#fff" d="M12 2 1 21h22L12 2Zm1 14h-2v-2h2v2Zm0-4h-2V8h2v4Z"/></svg>';
const OK_ICON = '<svg viewBox="0 0 24 24" width="34" height="34"><path fill="#fff" d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Zm-2 15-5-5 1.4-1.4L10 14.2l7.6-7.6L19 8Z"/></svg>';
let stream = null, running = false, mode = null, boxes = [], frameW = SEND_W, frameH = 240;

function showView(live) {
  $("home").hidden = live;
  $("live").hidden = !live;
}

function render(d) {                         // banner + stats box
  const banner = $("banner");
  if (!d.faces) {
    banner.className = "banner idle"; banner.textContent = "No faces detected";
  } else if (d.masked === d.faces) {
    banner.className = "banner ok"; banner.innerHTML = OK_ICON + "<span>ACCESS GRANTED</span>";
  } else {
    banner.className = "banner bad"; banner.innerHTML = WARN_ICON + "<span>PLEASE WEAR A MASK PROPERLY</span>";
  }
  $("count").textContent = d.faces;
  $("masked").textContent = d.masked;
  $("attn").textContent = d.faces - d.masked;
  $("meta").textContent = `${d.compliance}% compliance` + (d.fps !== undefined ? ` \u00b7 ${d.fps} updates/sec` : "");
}

function renderStages(st) {                  // "Image Operations / Image Segmentation" panels
  if (!st) return;
  const cards = list => list.map(s =>
    `<div class="step"><img src="${s.img}" alt="${s.name}"><span>${s.name}</span></div>`).join("");
  $("ops").innerHTML = cards(st.ops);
  $("seg").innerHTML = st.seg.length ? cards(st.seg)
      : '<p class="muted">No face found, so there is nothing to segment.</p>';
}

async function post(url, body) {
  const r = await fetch(url, body);
  return r.json();
}

function toBlob(canvas) {
  return new Promise(res => canvas.toBlob(res, "image/jpeg", 0.6));
}

function tag(ctx, text, x, y, fill, h = 30) {   // filled label with white/dark text
  const w = ctx.measureText(text).width + 18;
  ctx.fillStyle = fill; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = DARK_TEXT.has(fill) ? "#1b1b1b" : "#fff";
  ctx.fillText(text, x + 9, y + h / 2 + 1);
  return w;
}

function drawOverlay() {                     // boxes are in the (mirrored) sent-frame coordinates
  const ov = $("overlay"), ctx = ov.getContext("2d");
  ov.width = OVERLAY_W; ov.height = Math.round(OVERLAY_W * frameH / frameW);
  ctx.clearRect(0, 0, ov.width, ov.height);
  const k = ov.width / frameW;
  ctx.lineWidth = 4; ctx.font = "bold 19px 'Segoe UI', Arial, sans-serif"; ctx.textBaseline = "middle";
  for (const b of boxes) {
    const [x, y, w, h] = b.box.map(v => v * k);
    const col = COLORS[b.status] || "#64748b";
    ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
    tag(ctx, `ID ${b.id}`, x - 2, Math.max(y - 30, 0), col);
    const sw = ctx.measureText(b.status).width + 18;
    tag(ctx, b.status, Math.min(Math.max(x + w / 2 - sw / 2, 0), ov.width - sw),
        Math.min(y + h - 2, ov.height - 30), col);
  }
}

let prevT = performance.now(), fps = 0, lastStages = 0;

async function worker(offsetMs) {            // several workers overlap network waits
  const video = $("cam"), canvas = $("grab"), ctx = canvas.getContext("2d");
  await new Promise(r => setTimeout(r, offsetMs));
  while (running) {
    const w = SEND_W, h = Math.round(w * video.videoHeight / video.videoWidth) || 240;
    canvas.width = w; canvas.height = h;
    ctx.save(); ctx.translate(w, 0); ctx.scale(-1, 1);       // mirrored, like the video on screen
    ctx.drawImage(video, 0, 0, w, h); ctx.restore();
    try {
      const form = new FormData();
      form.append("image", await toBlob(canvas), "frame.jpg");
      form.append("sid", SID);
      if (performance.now() - lastStages > 1000) {            // pipeline thumbnails once a second
        form.append("stages", "1"); lastStages = performance.now();
      }
      const res = await post("/api/detect", {method: "POST", body: form});
      if (!running) break;
      if (res.ok) {
        boxes = res.boxes; frameW = res.w; frameH = res.h;
        drawOverlay();
        const now = performance.now();
        fps = 0.8 * fps + 0.2 * (1000 / (now - prevT)); prevT = now;
        res.fps = fps.toFixed(1);
        render(res);
        renderStages(res.stages);
      }
    } catch (e) {
      $("meta").textContent = "Lost connection to the server, retrying...";
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; $("cam").srcObject = null;
}

$("btnStart").onclick = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("msg").textContent = "This browser cannot use the camera (it needs HTTPS)."; return;
  }
  $("msg").style.color = "var(--muted)"; $("msg").textContent = "Asking for camera permission...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
  } catch (e) {
    $("msg").style.color = "var(--bad)";
    $("msg").textContent = "Camera blocked. Click the camera icon in the address bar and allow it."; return;
  }
  const video = $("cam");
  video.srcObject = stream;
  await video.play();
  mode = "camera"; running = true; boxes = [];
  video.hidden = false; $("photo").hidden = true;
  $("liveTitle").textContent = "Live Webcam Feed";
  $("stopLabel").textContent = "Stop Camera";
  $("btnSnap").hidden = false;
  $("msg").textContent = "";
  showView(true);
  for (let i = 0; i < IN_FLIGHT; i++) worker(i * 150);
};

$("btnStop").onclick = () => {                // also works as "Back" after a photo
  stopCamera(); mode = null; boxes = [];
  $("ops").innerHTML = '<p class="muted">Waiting for the first frame...</p>';
  $("seg").innerHTML = '<p class="muted">Appears when a face is detected.</p>';
  showView(false);
};

$("btnSnap").onclick = () => {                // video + labels merged into one JPEG
  const ov = $("overlay"), video = $("cam");
  const c = document.createElement("canvas"); c.width = ov.width; c.height = ov.height;
  const ctx = c.getContext("2d");
  ctx.save(); ctx.translate(c.width, 0); ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, c.width, c.height); ctx.restore();
  ctx.drawImage(ov, 0, 0);
  const a = document.createElement("a");
  a.href = c.toDataURL("image/jpeg", 0.9); a.download = `mask_snapshot_${Date.now()}.jpg`; a.click();
};

$("file").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  $("msg").style.color = "var(--muted)"; $("msg").textContent = "Analysing photo...";
  const form = new FormData(); form.append("image", file);
  const res = await post("/api/analyze", {method: "POST", body: form});
  e.target.value = "";
  if (!res.ok) { $("msg").style.color = "var(--bad)"; $("msg").textContent = res.error; return; }
  stopCamera(); mode = "photo";
  const photo = $("photo");
  photo.onload = () => { boxes = res.boxes; frameW = res.w; frameH = res.h; drawOverlay(); };
  photo.src = res.image; photo.hidden = false; $("cam").hidden = true;
  $("liveTitle").textContent = "Photo Analysis";
  $("stopLabel").textContent = "Back";
  $("btnSnap").hidden = true;
  $("msg").textContent = "";
  showView(true);
  render(res); renderStages(res.stages);
};