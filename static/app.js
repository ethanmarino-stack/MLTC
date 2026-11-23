/* Browser capture and display only. All ML predictions and alert rules run in Python. */
"use strict";
const $ = id => document.getElementById(id);
let socket, mediaStream, sampleTimer, pendingFrame = false, running = false, starting = false;
let mode = "debug", activeMode = "debug", generation = 0, stopping = false;
let intentionalClose = false;
let uploadedTokens = [];
const feeds = {debug: [], demo: []};
const demoCameras = ["north", "creek", "feeder", "gate"];
const demoCameraNames = {north:"North Trail", creek:"Creek Crossing", feeder:"Feeder Area", gate:"Main Gate"};
const demoPlayback = new DemoPlayback($, demoCameraNames, showError);

function showError(message) { $("error").textContent = message; $("error").hidden = !message; }
function timeOfDay(value) { return new Date(value).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", second:"2-digit"}); }
function controls() {
  for (const name of ["debug", "demo"]) {
    if (!$(name + "-start")) continue;
    const canConnect = socket?.readyState === WebSocket.OPEN || (intentionalClose && socket?.readyState === WebSocket.CLOSED);
    $(name + "-start").disabled = running || starting || stopping || !canConnect;
    $(name + "-stop").disabled = (!running && !starting) || stopping;
    $(name + "-settings").disabled = running || starting || stopping;
  }
  document.querySelectorAll(".camera-inputs").forEach(group => group.disabled = running || starting || stopping);
}
function stopMedia() {
  demoPlayback.clear();
  clearInterval(sampleTimer); sampleTimer = null; pendingFrame = false;
  if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
  mediaStream = null; $("local-preview").srcObject = null;
}
async function discardUploads() {
  const tokens = uploadedTokens; uploadedTokens = [];
  await Promise.allSettled(tokens.map(token => fetch("/api/uploads/" + token, {method:"DELETE"})));
}
function stop() {
  generation++; stopMedia();
  stopping = socket?.readyState === WebSocket.OPEN;
  if (stopping) socket.send(JSON.stringify({action:"stop"}));
  else { intentionalClose = true; socket?.close(); discardUploads(); }
  running = false; starting = false; controls();
  $("debug-status").textContent = "Stopped";
  demoCameras.forEach(id => { $(id + "-status").textContent = "Stopped"; });
}
function switchTab(name) {
  if (mode !== name && (running || starting)) stop();
  if (mode !== name) demoPlayback.clear();
  mode = name;
  $("selected-model").textContent = $(name + "-model").value.trim() || "No model selected";
  for (const tab of ["debug", "demo"]) {
    const selected = tab === name;
    $(tab).hidden = !selected;
    $("tab-" + tab).setAttribute("aria-selected", selected);
    $("tab-" + tab).tabIndex = selected ? 0 : -1;
  }
}
for (const tab of ["debug", "demo"]) {
  $("tab-" + tab).onclick = () => switchTab(tab);
  $("tab-" + tab).onkeydown = event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault(); const next = ["Home", "ArrowLeft"].includes(event.key) ? "debug" : "demo";
      switchTab(next); $("tab-" + next).focus();
    }
  };
}
function updateSource() {
  stopMedia();
  $("camera-picker").hidden = $("debug-source").value !== "webcam";
  $("debug-upload-wrap").hidden = $("debug-source").value !== "video";
  $("preview-details").hidden = $("debug-source").value === "video";
}
$("debug-source").onchange = updateSource;
async function refreshCameras() {
  try {
    if (!navigator.mediaDevices) throw new Error("Open http://127.0.0.1:8000 in a supported browser.");
    const selected = $("webcam").value;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "videoinput");
    $("webcam").replaceChildren(new Option("Default webcam", ""));
    devices.forEach((device, index) => $("webcam").add(new Option(device.label || "Webcam " + (index + 1), device.deviceId)));
    if ([...$("webcam").options].some(option => option.value === selected)) $("webcam").value = selected;
  } catch (error) { showError(error.message); }
}
$("refresh-cameras").onclick = refreshCameras;

async function upload(file) {
  if (!file) throw new Error("Select a saved video first.");
  if (file.size > 200 * 1024 * 1024) throw new Error("Video exceeds 200 MB. Use a local Demo path for larger files.");
  const form = new FormData(); form.append("file", file);
  const response = await fetch("/api/uploads", {method:"POST", body:form});
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Video upload failed.");
  uploadedTokens.push(result.upload_id);
  return result;
}
function settings(name, cameras) {
  return {action:"start", mode:name, model:$(name + "-model").value.trim(),
    confidence:Number($(name + "-confidence").value)/100, rate:Number($(name + "-rate").value),
    repeats:Number($(name + "-repeats").value), cooldown:Number($(name + "-cooldown").value),
    loop:$(name + "-loop")?.checked || false,
    display_mode:name === "demo" ? $("demo-display-mode").value : "detection", cameras};
}
function syncDemoDisplayMode() {
  if (!running && !starting && !stopping) {
    demoPlayback.configure($("demo-display-mode").value, $("demo-loop").checked, Number($("demo-rate").value));
  }
}
async function startDebug() {
  const runId = ++generation; starting = true; activeMode = "debug"; controls(); showError("");
  try {
    const kind = $("debug-source").value;
    let camera = {id:"debug", kind};
    if (kind === "video") {
      const result = await upload($("debug-upload").files[0]);
      camera = {...camera, upload_id:result.upload_id, name:result.name};
    } else {
      if (!navigator.mediaDevices) throw new Error("Browser capture requires localhost and a supported browser.");
      // Request screen permission directly from the Start click, as browsers require.
      const acquired = kind === "screen"
        ? await navigator.mediaDevices.getDisplayMedia({video:true, audio:false})
        : await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960}, height:{ideal:540},
            ...($("webcam").value ? {deviceId:{exact:$("webcam").value}} : {})}, audio:false});
      if (runId !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
      mediaStream = acquired; $("local-preview").srcObject = acquired;
      $("preview-details").open = true;
      acquired.getTracks().forEach(track => track.addEventListener("ended", stop, {once:true}));
      await $("local-preview").play();
      camera.name = $("webcam").selectedOptions[0]?.textContent || "Webcam";
      if (kind === "webcam") refreshCameras();
    }
    if (runId !== generation) { stopMedia(); await discardUploads(); return; }
    await ensureConnection();
    if (runId !== generation) { stopMedia(); await discardUploads(); return; }
    socket.send(JSON.stringify(settings("debug", [camera])));
    $("debug-status").textContent = "Loading model…";
  } catch (error) {
    stopMedia(); await discardUploads(); starting = false; controls();
    showError(error.name === "NotAllowedError" ? "Permission was declined or sharing was cancelled. Press Start to try again." : error.message);
  }
}

async function startDemo() {
  const runId = ++generation; starting = true; activeMode = "demo"; controls(); showError("");
  try {
    demoPlayback.configure($("demo-display-mode").value, $("demo-loop").checked, Number($("demo-rate").value));
    const cameras = [];
    for (const id of demoCameras) {
      const file = $(id + "-upload").files[0], path = $(id + "-path").value.trim();
      if (file) {
        demoPlayback.selectUpload(id, file);
        const result = await upload(file);
        cameras.push({id, kind:"video", upload_id:result.upload_id, name:result.name});
      } else if (path) {
        cameras.push({id, kind:"video", path, name:path.split(/[\\/]/).pop()});
      }
      if (runId !== generation) { await discardUploads(); return; }
    }
    if (!cameras.length) throw new Error("Select a recording or trusted path for at least one camera.");
    await ensureConnection();
    if (runId !== generation) { await discardUploads(); return; }
    const request = settings("demo", cameras);
    console.debug("[Demo] start", {display_mode:request.display_mode, cameras:cameras.map(camera => camera.id)});
    socket.send(JSON.stringify(request));
  } catch (error) {
    demoPlayback.clear(); await discardUploads(); starting = false; controls(); showError(error.message);
  }
}

async function readyDemoPlayback(message) {
  const runId = generation;
  console.debug("[Demo] prepare", {display_mode:message.display_mode,
    cameras:message.cameras.map(camera => ({id:camera.id, media_registered:!!camera.media_url}))});
  await demoPlayback.prepare(message.cameras);
  console.debug("[Demo] prepared");
  if (runId === generation && running && socket?.readyState === WebSocket.OPEN) {
    console.debug("[Demo] play");
    socket.send(JSON.stringify({action:"play"}));
  }
}
function sampleFrame() {
  // Wait for the previous response. Slow inference drops sampling opportunities
  // rather than building a browser/network queue of increasingly old frames.
  if (pendingFrame || !running || !mediaStream || socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount) return;
  const video = $("local-preview");
  if (!video.videoWidth) return;
  const canvas = $("capture-canvas"), runId = generation;
  const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  pendingFrame = true;
  canvas.toBlob(blob => {
    if (runId !== generation) return;
    if (blob && blob.size <= 500000 && socket.readyState === WebSocket.OPEN) socket.send(blob);
    else { pendingFrame = false; showError("Could not encode a small enough browser frame."); }
  }, "image/jpeg", .72);
}
function addEvents(name, events) {
  feeds[name] = [...events].reverse().concat(feeds[name]).slice(0, 20);
  const list = $(name + "-feed"); list.replaceChildren();
  for (const event of feeds[name]) {
    const item = document.createElement("li"), title = document.createElement("strong");
    title.textContent = event.title; item.append(title);
    const detail = document.createElement("span");
    detail.textContent = `${event.camera_name} · ${event.source} · ${event.footage_time ? "Footage " + event.footage_time : timeOfDay(event.processing_time)} · Confidence ${Math.round(event.confidence*100)}%`;
    item.append(detail);
    const processed = document.createElement("span"); processed.textContent = "Processed locally at " + timeOfDay(event.processing_time); item.append(processed);
    if (event.snapshot_url) { const link = document.createElement("a"); link.href = event.snapshot_url; link.target = "_blank"; link.rel = "noopener"; link.textContent = "View snapshot ↗"; item.append(link); }
    list.append(item);
  }
  $(name + "-feed-empty").hidden = feeds[name].length > 0;
}
function receive(message) {
  if (message.type === "connected") { $("system-status").textContent = "System ready"; controls(); }
  if (message.type === "started") {
    if (stopping) return;
    if (message.mode === "demo") {
      const requested = demoPlayback.live ? "live" : "detection";
      console.debug("[Demo] started", {requested_mode:requested, returned_mode:message.display_mode});
      if (message.display_mode !== requested) {
        // Never silently wait for frames from a different/older handshake.
        stop();
        showError("Demo display-mode mismatch. Fully restart the server, then hard-refresh the dashboard (Ctrl+Shift+R).");
        return;
      }
    }
    intentionalClose = false; starting = false; running = true; activeMode = message.mode;
    $("selected-model").textContent = message.model; $("system-status").textContent = "Monitoring";
    controls();
    for (const camera of message.cameras) {
      $(camera.id + "-status").textContent = message.mode !== "demo" || camera.error ? camera.status
        : (message.display_mode === "live" ? "Preparing playback…" : "Starting detection…");
    }
    const failures = message.cameras.filter(camera => camera.error);
    if (failures.length) showError(failures.map(camera => camera.status).join(" "));
    if (message.mode === "debug" && mediaStream) {
      sampleTimer = setInterval(sampleFrame, 1000/Number($("debug-rate").value)); sampleFrame();
    }
    if (message.mode === "demo" && message.display_mode === "live") {
      readyDemoPlayback(message).catch(() => showError("Could not prepare browser playback. Stop and try Detection view."));
    }
  }
  if (message.type === "playing" && running && activeMode === "demo") {
    console.debug("[Demo] playing");
    demoPlayback.playTogether(message.starts_in_ms);
  }
  if (message.type === "frame") {
    pendingFrame = false;
    if (!running) return;
    $(message.camera_id + "-status").textContent = message.source;
    if (message.camera_id === "debug") {
      const image = $("debug-frame"); image.src = "data:image/jpeg;base64," + message.image;
      image.hidden = false; $("debug-empty").hidden = true;
      $("debug-fps").textContent = message.fps ? message.fps.toFixed(1) + " FPS" : "Warming up";
      $("debug-inference").textContent = message.inference_ms + " ms";
      $("debug-processing").textContent = timeOfDay(message.processing_time);
      $("debug-footage").textContent = message.footage_time || "N/A · live source";
      $("debug-detections").textContent = message.detections.map(box => `${box.label} ${Math.round(box.confidence*100)}%`).join(" · ") || "No detections above the confidence threshold";
    } else {
      demoPlayback.renderFrame(message);
      $(message.camera_id + "-footage").textContent = "Footage " + message.footage_time;
      $(message.camera_id + "-detections").textContent = Object.entries(message.animals).map(([label, score]) => `${label} ${Math.round(score*100)}%`).join(" · ") || "No animals detected";
      $(message.camera_id + "-status").textContent = `${message.status} · Detection ${message.fps.toFixed(1)} FPS · ${message.inference_ms} ms`;
    }
    addEvents(activeMode, message.events);
  }
  if (message.type === "frame_dropped") pendingFrame = false;
  if (message.type === "error") {
    pendingFrame = false; showError(message.message);
    if (message.camera_id) {
      demoPlayback.release(message.camera_id);
      $(message.camera_id + "-status").textContent = "Source error";
    }
    else { starting = false; if (!running) stopMedia(); controls(); }
  }
  if (message.type === "camera_status") {
    demoPlayback.release(message.camera_id);
    $(message.camera_id + "-status").textContent = message.status;
  }
  if (["finished", "stopped"].includes(message.type)) {
    stopMedia(); running = false; starting = false; stopping = message.type === "finished";
    controls(); $("system-status").textContent = "System ready";
    if (message.type === "stopped") $("debug-status").textContent = "Stopped";
    if (message.type === "stopped") {
      discardUploads(); intentionalClose = true; socket.close(1000, "Stopped"); controls();
    }
    if (message.type === "finished") socket.send(JSON.stringify({action:"stop"}));
  }
}
function connect() {
  let resolveReady, rejectReady;
  const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  socket.onmessage = event => {
    const message = JSON.parse(event.data); receive(message);
    if (message.type === "connected") resolveReady();
  };
  socket.onclose = () => {
    stopMedia(); running = false; starting = false; stopping = false; controls();
    rejectReady(new Error("Connection closed."));
    $("system-status").textContent = intentionalClose ? "System ready" : "Disconnected";
    if (!intentionalClose) { generation++; showError("Connection closed. Reload to reconnect. Use one dashboard tab."); }
  };
  socket.onerror = () => {
    generation++; stopMedia(); socket.close();
    rejectReady(new Error("Cannot reach the local Python server."));
    showError("Cannot reach the local Python server. Check the terminal and reload.");
  };
  return ready;
}
async function ensureConnection() {
  if (socket?.readyState !== WebSocket.OPEN) await connect();
}
$("debug-start").onclick = startDebug;
$("debug-stop").onclick = stop;
$("demo-start").onclick = startDemo;
$("demo-stop").onclick = stop;
$("demo-clear").onclick = () => { feeds.demo = []; addEvents("demo", []); };
$("debug-clear").onclick = () => { feeds.debug = []; addEvents("debug", []); };
$("debug-confidence").oninput = () => $("debug-confidence-value").textContent = $("debug-confidence").value + "%";
$("demo-confidence").oninput = () => $("demo-confidence-value").textContent = $("demo-confidence").value + "%";
$("demo-display-mode").onchange = syncDemoDisplayMode;
for (const id of demoCameras) {
  $(id + "-upload").onchange = () => demoPlayback.release(id);
  $(id + "-path").onchange = () => demoPlayback.release(id);
}
for (const name of ["debug", "demo"]) $(name + "-model").oninput = () => $("selected-model").textContent = $(name + "-model").value.trim() || "No model selected";
window.addEventListener("pagehide", () => { stopMedia(); socket?.close(); });
updateSource();
// Browsers may restore a previous select value without firing `change`.
// Initialize the visible stages from the actual restored value before Start.
syncDemoDisplayMode();
window.addEventListener("pageshow", syncDemoDisplayMode);
connect().catch(() => {});
