// Front-end logic. The camera now runs in the BROWSER (getUserMedia); each frame is
// sent to the Python server, which runs the OpenCV pipeline and returns the annotated frame.
const $ = id => document.getElementById(id);
const COLORS = {"MASK ON":"#1a9e4a","NO MASK":"#d83a3a","MASK BELOW NOSE":"#e07b00",
                "MASK ON CHIN":"#e07b00","CHECK MASK":"#c9a400"};
const SID = Math.random().toString(36).slice(2) + Date.now().toString(36);
let stream = null, running = false, lastImage = null;

function show(which) {                       // which: "feed" | "photo" | null
  $("feed").hidden = which !== "feed";
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

async function loop() {
  const video = $("cam"), canvas = $("grab"), ctx = canvas.getContext("2d");
  let prev = performance.now(), fps = 0;
  while (running) {
    const w = 320, h = Math.round(w * video.videoHeight / video.videoWidth) || 360;
    canvas.width = w; canvas.height = h;
    ctx.save(); ctx.translate(w, 0); ctx.scale(-1, 1);       // mirror, like the desktop version
    ctx.drawImage(video, 0, 0, w, h); ctx.restore();
    try {
      const form = new FormData();
      form.append("image", await toBlob(canvas), "frame.jpg");
      form.append("sid", SID);
      const res = await post("/api/frame", {method: "POST", body: form});
      if (!running) break;
      if (res.ok) {
        lastImage = res.image;
        $("feed").src = res.image;
        const now = performance.now();
        fps = 0.8 * fps + 0.2 * (1000 / (now - prev)); prev = now;
        res.fps = fps.toFixed(1);
        render(res);
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
  running = true;
  show("feed");
  $("btnStart").disabled = true; $("btnStop").disabled = false; $("btnSnap").disabled = false;
  $("msg").textContent = "Camera running. Try covering your nose and mouth with a mask.";
  loop();
};

$("btnStop").onclick = () => {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; $("feed").removeAttribute("src"); show(null);
  $("btnStart").disabled = false; $("btnStop").disabled = true; $("btnSnap").disabled = true;
  $("msg").textContent = "Camera stopped.";
};

$("btnSnap").onclick = () => {                // saves the latest annotated frame
  if (!lastImage) return;
  const a = document.createElement("a");
  a.href = lastImage; a.download = `mask_snapshot_${Date.now()}.jpg`; a.click();
};

$("file").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  $("msg").textContent = "Analysing photo...";
  const form = new FormData(); form.append("image", file);
  const res = await post("/api/analyze", {method: "POST", body: form});
  if (!res.ok) { $("msg").textContent = res.error; return; }
  $("photo").src = res.image; show("photo"); render(res);
  $("msg").textContent = res.faces ? "Photo analysed." : "No face found - try a clearer, front-facing photo.";
  e.target.value = "";
};