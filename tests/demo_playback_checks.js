/* CPU-only JavaScript checks with fake DOM/video/socket objects. No page is opened.
   A development JS runner evaluates this file, demo-playback.js, then app.js in
   the same isolated context and calls runDemoPlaybackChecks(). No app dependency. */
const mockNodes = new Map(), mockTimers = new Map(), mockRevoked = [];
let mockTimerId = 0, mockURLId = 0;
class MockElement {
  constructor(id = "") {
    this.id = id; this.hidden = true; this.children = []; this.listeners = new Map();
    this.value = ""; this.checked = false; this.files = []; this.src = "";
    this.readyState = 2; this.duration = 30; this.currentTime = 0; this.paused = true;
  }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  removeEventListener(name) { this.listeners.delete(name); }
  pause() { this.paused = true; }
  load() {} // Deliberately does not open or decode media.
  play() {
    if (this.rejectPlay) return Promise.reject(new Error("Mock autoplay rejected"));
    this.paused = false; return Promise.resolve();
  }
}
function mockElement(id) {
  if (!mockNodes.has(id)) mockNodes.set(id, new MockElement(id));
  return mockNodes.get(id);
}
globalThis.document = {
  getElementById: mockElement,
  createElement: tag => new MockElement(tag),
  querySelectorAll: () => ["north", "creek", "feeder", "gate"].map(id => mockElement(id + "-inputs"))
};
const mockWindowEvents = new Map();
globalThis.window = {addEventListener: (name, callback) => mockWindowEvents.set(name, callback)};
globalThis.location = {protocol:"http:", host:"mock.invalid"};
globalThis.URL = {createObjectURL: () => "blob:mock-" + (++mockURLId), revokeObjectURL: url => mockRevoked.push(url)};
globalThis.setTimeout = callback => { const id = ++mockTimerId; mockTimers.set(id, callback); return id; };
globalThis.clearTimeout = id => mockTimers.delete(id);
globalThis.setInterval = globalThis.setTimeout;
globalThis.clearInterval = globalThis.clearTimeout;
globalThis.fetch = async () => ({ok:true, json:async () => ({})});
globalThis.FormData = class { append() {} };
class MockSocket {
  static OPEN = 1;
  static CLOSED = 3;
  constructor() { this.readyState = 1; this.sent = []; this.bufferedAmount = 0; }
  send(value) { this.sent.push(value); }
  close() { this.readyState = 3; this.onclose?.(); }
}
globalThis.WebSocket = MockSocket;
mockElement("debug-source").value = "webcam";
mockElement("demo-display-mode").value = "live";
mockElement("demo-loop").checked = true;
mockElement("demo-rate").value = "1";

async function runDemoPlaybackChecks() {
  const passed = [];
  function assert(condition, message) { if (!condition) throw new Error(message); }
  function flushStart() {
    const id = demoPlayback.startTimer, callback = mockTimers.get(id);
    mockTimers.delete(id); callback();
  }
  function frame(events = []) {
    return {type:"frame", camera_id:"north", status:"Replaying", source:"REPLAY north.mp4",
      image:"unexpected-image-must-not-display", display_mode:"live", playback_seconds:2.5,
      fps:1, inference_ms:10, footage_time:"00:02.50", animals:{cow:.9}, events};
  }

  demoPlayback.configure("live", true, 1);
  demoPlayback.selectUpload("north", {});
  demoPlayback.playTogether(250); flushStart();
  running = true; activeMode = "demo";
  receive(frame([{title:"Cow spotted", camera_name:"North Trail", source:"REPLAY north.mp4",
    footage_time:"00:02.50", processing_time:"2026-11-11T12:00:00-06:00", confidence:.9,
    snapshot_url:"/snapshots/fixture.jpg"}]));
  assert(mockElement("north-frame").hidden, "Live view displayed an annotated image");
  assert(!mockElement("north-video").hidden && !mockElement("north-video").paused, "Raw video did not start");
  assert(mockElement("demo-feed").children.length === 1, "Live view lost its detection event");
  assert(mockElement("north-status").textContent.includes("Detection 1.0 FPS"), "Misleading FPS label");
  assert(mockElement("north-video").loop, "Loop setting was not applied");
  passed.push("Live view renders raw video while detection events and metadata continue");

  const video = mockElement("north-video");
  video.currentTime = 2; demoPlayback.renderFrame({...frame(), playback_seconds:2.7});
  assert(video.currentTime === 2, "Minor drift caused a seek");
  demoPlayback.renderFrame({...frame(), playback_seconds:5});
  assert(video.currentTime === 5, "Large drift was not corrected");
  video.currentTime = 29.8; demoPlayback.renderFrame({...frame(), playback_seconds:.1});
  assert(video.currentTime === 29.8, "Loop boundary caused an unnecessary seek");
  const urlsBefore = mockURLId;
  for (let i=0; i<10; i++) receive(frame());
  assert(mockURLId === urlsBefore && demoPlayback.entries.size === 1 && mockTimers.size === 0,
    "Loop/frame updates created duplicate URLs or timers");
  passed.push("Drift correction and loops do not duplicate playback resources");

  demoPlayback.configure("detection", false, 1);
  demoPlayback.renderFrame({...frame(), image:"jpeg-frame", display_mode:"detection"});
  assert(!mockElement("north-frame").hidden, "Detection view lost annotated frames");
  assert(mockElement("north-video").hidden, "Detection view also displayed raw video");
  assert(mockElement("north-frame").src === "data:image/jpeg;base64,jpeg-frame", "JPEG source changed");
  passed.push("Detection view retains the annotated JPEG fallback exclusively");

  demoPlayback.configure("live", true, 1);
  demoPlayback.selectUpload("north", {});
  const first = demoPlayback.entries.get("north").url;
  demoPlayback.selectUpload("north", {});
  assert(mockRevoked.filter(url => url === first).length === 1, "Replacement URL was not revoked once");
  demoPlayback.selectUpload("creek", {});
  mockElement("creek-video").rejectPlay = true;
  demoPlayback.playTogether(0); flushStart(); await Promise.resolve();
  assert(!mockElement("north-video").paused, "A failed camera stopped an unrelated video");
  assert(mockElement("creek-playback-note").textContent.includes("Creek Crossing"), "Playback error lacks camera name");
  assert(mockElement("creek-playback-note").textContent.includes("Detection view"), "No fallback guidance");
  passed.push("Playback failure is isolated and offers Detection view fallback");

  controls(); assert(mockElement("demo-settings").disabled, "Settings changed during a run");
  starting = true; running = false; controls();
  assert(mockElement("demo-settings").disabled, "Settings changed during startup");
  starting = false; stopping = true; controls();
  assert(mockElement("demo-settings").disabled, "Settings changed during shutdown");
  stopping = false; running = true;
  stop();
  assert(demoPlayback.entries.size === 0 && mockTimers.size === 0, "Stop retained media resources");
  assert(mockElement("north-video").src === undefined, "Stop retained a video source");
  passed.push("Busy settings lock, replacement and Stop cleanup work");

  demoPlayback.configure("live", false, 1);
  demoPlayback.selectUpload("north", {});
  mockElement("north-video").readyState = 0;
  const readiness = demoPlayback.prepare([{id:"north", error:false}]);
  demoPlayback.clear(); await readiness;
  assert(mockTimers.size === 0, "Cancelled loading timer leaked");
  mockElement("north-video").readyState = 2;
  demoPlayback.selectUpload("north", {}); mode = "demo"; running = false;
  switchTab("debug"); assert(demoPlayback.entries.size === 0, "Tab change retained object URL");
  demoPlayback.selectUpload("north", {}); socket.onerror();
  assert(demoPlayback.entries.size === 0, "Socket failure retained object URL");
  demoPlayback.selectUpload("north", {}); mockWindowEvents.get("pagehide")();
  assert(demoPlayback.entries.size === 0 && mockTimers.size === 0, "Page exit retained resources");
  assert(mockRevoked.length === mockURLId && new Set(mockRevoked).size === mockRevoked.length,
    "Object URLs were leaked or revoked more than once");
  passed.push("Loading cancellation, tab change, socket failure and page exit release every URL");
  return passed;
}

async function runDemoStartupChecks() {
  const passed = [];
  function assert(condition, message) { if (!condition) throw new Error(message); }
  async function settle() { for (let i=0; i<8; i++) await Promise.resolve(); }
  function reset(displayMode, sourceType = "path") {
    demoPlayback.clear(); mockTimers.clear();
    running = false; starting = false; stopping = false; mode = "demo"; generation++;
    socket = new MockSocket();
    for (const id of demoCameras) {
      mockElement(id + "-path").value = ""; mockElement(id + "-upload").files = [];
      mockElement(id + "-video").readyState = 2;
      mockElement(id + "-video").rejectPlay = false;
    }
    mockElement("demo-display-mode").value = displayMode;
    mockElement("demo-model").value = "mock-model";
    mockElement("demo-confidence").value = "50";
    mockElement("demo-repeats").value = "2";
    mockElement("demo-cooldown").value = "30";
    if (sourceType === "path") mockElement("north-path").value = "fixture.mp4";
    else mockElement("north-upload").files = [{size:100}];
    globalThis.fetch = async () => ({ok:true, json:async () => ({upload_id:"mock-upload", name:"fixture.mp4"})});
    showError("");
  }
  function started(displayMode, sourceType) {
    return {type:"started", mode:"demo", display_mode:displayMode, model:"mock-model",
      cameras:[{id:"north", status:"Ready", error:false,
        media_url:displayMode === "live" && sourceType === "path" ? "/api/media/mock-token" : null}]};
  }
  function frame(displayMode) {
    return {type:"frame", camera_id:"north", display_mode:displayMode, status:"Replaying",
      source:"REPLAY fixture.mp4", image:displayMode === "detection" ? "jpeg-frame" : null,
      animals:{cow:.9}, fps:1, inference_ms:10, playback_seconds:2.5, footage_time:"00:02.50", events:[]};
  }

  reset("detection");
  mockWindowEvents.get("pageshow")();
  assert(mockElement("north-view-badge").textContent === "DETECTION VIEW", "Restored selector differs from badges");
  mockElement("demo-display-mode").value = "live"; mockElement("demo-display-mode").onchange();
  assert(mockElement("north-view-badge").textContent === "LIVE VIEW", "Stopped selector does not update immediately");
  running = true; mockElement("demo-display-mode").value = "detection";
  mockElement("demo-display-mode").onchange();
  assert(demoPlayback.live, "A running session changed its display mode");
  passed.push("Restored and stopped selectors synchronize badges; running mode stays locked");

  for (const displayMode of ["detection", "live"]) {
    for (const sourceType of ["path", "upload"]) {
      reset(displayMode, sourceType);
      await startDemo();
      const request = JSON.parse(socket.sent[0]);
      assert(request.display_mode === displayMode, "Start omitted/misreported the requested mode");
      receive(started(displayMode, sourceType)); await settle();
      const actions = socket.sent.map(value => JSON.parse(value).action);
      assert(JSON.stringify(actions) === JSON.stringify(displayMode === "live" ? ["start", "play"] : ["start"]), "Wrong play handshake");
      if (displayMode === "live") {
        receive({type:"playing", starts_in_ms:250});
        const timer = demoPlayback.startTimer, callback = mockTimers.get(timer);
        mockTimers.delete(timer); callback();
      }
      receive(frame(displayMode));
      assert(mockElement("north-empty").hidden, "Camera stayed on Waiting for a recording");
      assert(mockElement("north-frame").hidden === (displayMode === "live"), "Wrong image display mode");
      assert(mockElement("north-video").hidden === (displayMode === "detection"), "Wrong raw video display mode");
      assert(mockElement("north-footage").textContent.includes("00:02.50"), "Footage timestamp did not update");
      assert(mockElement("north-view-badge").textContent === (displayMode === "live" ? "LIVE VIEW" : "DETECTION VIEW"), "Mode badge mismatch");
      passed.push(`${displayMode}/${sourceType}: complete Start and message flow makes feed visible`);
    }
  }

  for (const returned of [undefined, "live"]) {
    reset("detection"); await startDemo(); receive(started(returned, "path"));
    assert(!running && !starting, "Mixed-version started message left a running/Ready session");
    assert(JSON.parse(socket.sent.at(-1)).action === "stop", "Mode mismatch did not clean up server session");
    assert(mockElement("error").textContent.includes("hard-refresh"), "No mismatch recovery instructions");
  }
  passed.push("Missing/wrong server display mode fails visibly instead of silently staying Ready");

  reset("detection");
  const get = demoPlayback.get;
  demoPlayback.get = id => id === "north-view-badge" ? null : get(id);
  try {
    await startDemo();
    assert(!starting && !running && !mockElement("error").hidden, "Missing DOM left Start disabled with no error");
  } finally { demoPlayback.get = get; }
  passed.push("Missing/mismatched DOM errors during configuration are caught before sending Start");

  reset("live"); await startDemo();
  mockElement("north-video").readyState = 0;
  receive(started("live", "path")); await settle();
  assert(socket.sent.length === 1, "Live started before browser preparation");
  for (const [id, callback] of [...mockTimers]) { mockTimers.delete(id); callback(); }
  await settle();
  assert(JSON.parse(socket.sent.at(-1)).action === "play", "Failed/timeout media preparation deadlocked detection");
  assert(mockElement("north-playback-note").textContent.includes("North Trail"), "Media timeout lacks camera-specific error");
  passed.push("Media loading failure resolves preparation and still starts Python via play");
  demoPlayback.clear(); mockTimers.clear();
  return passed;
}
