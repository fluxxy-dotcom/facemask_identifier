// Front-end logic: talks to the Python (Flask) server with fetch().
const $ = id => document.getElementById(id);
const COLORS = {"MASK ON":"#1a9e4a","NO MASK":"#d83a3a","MASK BELOW NOSE":"#e07b00",
                "MASK ON CHIN":"#e07b00","CHECK MASK":"#c9a400"};
let poll = null;

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

$("btnStart").onclick = async () => {
  $("msg").textContent = "Starting camera (loading models)...";
  const res = await post("/api/camera", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({action: "start"})});
  if (!res.ok) { $("msg").textContent = res.error; return; }
  $("feed").src = "/video_feed?" + Date.now();          // MJPEG stream from Python
  show("feed");
  $("btnStart").disabled = true; $("btnStop").disabled = false; $("btnSnap").disabled = false;
  $("msg").textContent = "Camera running. Try covering your nose and mouth with a mask.";
  poll = setInterval(async () => render(await (await fetch("/api/stats")).json()), 500);
};

$("btnStop").onclick = async () => {
  await post("/api/camera", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({action: "stop"})});
  clearInterval(poll); $("feed").removeAttribute("src"); show(null);
  $("btnStart").disabled = false; $("btnStop").disabled = true; $("btnSnap").disabled = true;
  $("msg").textContent = "Camera stopped.";
};

$("btnSnap").onclick = () => { window.location = "/api/snapshot"; };   // downloads a JPEG

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
