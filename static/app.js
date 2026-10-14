// Front-end logic. The camera runs in the BROWSER and is shown live at full speed.
// Small frames are sent to the Python server, which returns only face boxes + statuses;
// the browser draws those on a transparent canvas over the live video.
const $ = id => document.getElementById(id);
const COLORS = {"MASK ON":"#1a9e4a","NO MASK":"#d83a3a","MASK BELOW NOSE":"#e07b00",
                "MASK ON CHIN":"#e07b00","CHECK MASK":"#c9a400"};
const SID = Math.random().toString(36).slice(2) + Date.now().toString(36);
const SEND_W = 320, OVERLAY_W = 640, IN_FLIGHT = 2;
let stream = null, running = false, boxes = [], frameW = SEND_W, frameH = 240;

function show(which) {                       // which: "feed" | "photo" | null
  $("stage").hidden = which !== "feed";
  $("photo").hidden = which !== "photo";
  $("placeholder").hidden = which !== null;
}

function render(d) {                         // update the side panel from JSON
  const banner = $("banner");
  banner.textContent = d.banner;
  banner.className = "banner " + (d.banner === "ACCESS GRANTED" ? "ok" :
                                  d.faces ? "bad" : "");
  $("bar").style.width = d.compliance + "%";
  $("comp").textContent = d.compliance;
  $("count").textContent = d.faces;
  if (d.fps !== undefined) $("fps").textContent = d.fps;
  $("people").innerHTML = d.people.map(p =>
    `<div class="person"><b>Person ${p.id}</b>` +
    `<span class="chip" style="background:${COLORS[p.status] || "#666"}">${p.status}</span></div>`
  ).join("");
}

async function post(url, body) {
  const r = await fetch(url, body);
  return r.json();
}

function toBlob(canvas) {
  return new Promise(res => canvas.toBlob(res, "image/jpeg", 0.6));
}

function drawOverlay() {                     // boxes are in the (mirrored) sent-frame coordinates
  const ov = $("overlay"), ctx = ov.getContext("2d");
  ov.width = OVERLAY_W; ov.height = Math.round(OVERLAY_W * frameH / frameW);
  ctx.clearRect(0, 0, ov.width, ov.height);
  const k = ov.width / frameW;
  ctx.lineWidth = 3; ctx.font = "bold 18px sans-serif"; ctx.textBaseline = "bottom";
  for (const b of boxes) {
    const [x, y, w, h] = b.box.map(v => v * k);
    const col = COLORS[b.status] || "#888";
    const label = `Person ${b.id}: ${b.status}`;
    ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(0,0,0,.65)";
    ctx.fillRect(x, Math.max(y - 24, 0), ctx.measureText(label).width + 10, 24);
    ctx.fillStyle = col; ctx.fillText(label, x + 5, Math.max(y, 24) - 3);
  }
}

let prevT = performance.now(), fps = 0, lastStages = 0;

function renderStages(st) {                  // fills the "Image Operations / Segmentation" panels
  if (!st) return;
  const cards = list => list.map(s =>
    `<div class="step"><img src="${s.img}" alt="${s.name}"><span>${s.name}</span></div>`).join("");
  $("ops").innerHTML = cards(st.ops);
  $("seg").innerHTML = st.seg.length ? cards(st.seg)
      : '<p class="muted">No face found, so there is nothing to segment.</p>';
}

async function worker(offsetMs) {            // several workers overlap network waits
  const video = $("cam"), canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
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
      $("msg").textContent = "Lost connection to the server, retrying...";
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

$("btnStart").onclick = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("msg").textContent = "This browser cannot use the camera (it needs HTTPS)."; return;
  }
  $("msg").textContent = "Asking for camera permission...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
  } catch (e) {
    $("msg").textContent = "Camera blocked. Click the camera icon in the address bar and allow it."; return;
  }
  const video = $("cam");
  video.srcObject = stream;
  await video.play();
  running = true; boxes = [];
  show("feed");
  $("btnStart").disabled = true; $("btnStop").disabled = false; $("btnSnap").disabled = false;
  $("msg").textContent = "Camera running. Try covering your nose and mouth with a mask.";
  for (let i = 0; i < IN_FLIGHT; i++) worker(i * 150);
};

$("btnStop").onclick = () => {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; boxes = []; $("cam").srcObject = null; show(null);
  $("btnStart").disabled = false; $("btnStop").disabled = true; $("btnSnap").disabled = true;
  $("msg").textContent = "Camera stopped.";
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
  $("msg").textContent = "Analysing photo...";
  const form = new FormData(); form.append("image", file);
  const res = await post("/api/analyze", {method: "POST", body: form});
  if (!res.ok) { $("msg").textContent = res.error; return; }
  $("photo").src = res.image; show("photo"); render(res); renderStages(res.stages);
  $("msg").textContent = res.faces ? "Photo analysed." : "No face found - try a clearer, front-facing photo.";
  e.target.value = "";
};