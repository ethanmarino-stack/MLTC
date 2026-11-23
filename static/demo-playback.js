/* Raw browser playback only. Python owns detection, event timestamps and alerts. */
"use strict";
class DemoPlayback {
  constructor(getElement, cameraNames, onError) {
    this.get = getElement;
    this.names = cameraNames;
    this.onError = onError;
    this.entries = new Map();
    this.startTimer = null;
    this.live = true;
    this.loop = false;
  }

  configure(displayMode, loop, rate) {
    this.clear();
    this.live = displayMode === "live";
    this.loop = loop;
    for (const id of Object.keys(this.names)) {
      this.get(id + "-frame").hidden = true;
      this.get(id + "-empty").hidden = false;
      this.get(id + "-view-badge").textContent = this.live ? "LIVE VIEW" : "DETECTION VIEW";
      this.get(id + "-playback-note").textContent = this.live
        ? `Smooth playback · Detection sampling ${rate} FPS`
        : `Annotated frames · Detection sampling ${rate} FPS`;
    }
  }

  selectUpload(id, file) {
    if (this.live) this.setSource(id, URL.createObjectURL(file), true);
  }

  setSource(id, url, objectURL = false) {
    this.release(id);
    const video = this.get(id + "-video");
    const entry = {video, url, objectURL, failed:false, cancelReady:null};
    this.entries.set(id, entry);
    video.muted = true; video.autoplay = false; video.playsInline = true;
    video.loop = this.loop;
    // Playback starts together only after Python acknowledges the shared epoch.
    video.onerror = () => this.playbackError(id, entry);
    video.onended = () => {
      if (!this.loop) this.get(id + "-playback-note").textContent = "Replay ended · Waiting for final detections";
    };
    video.src = url;
    video.load();
  }

  playbackError(id, entry) {
    if (this.entries.get(id) !== entry || entry.failed) return;
    entry.failed = true;
    entry.video.pause(); entry.video.hidden = true;
    this.get(id + "-empty").hidden = false;
    const message = `${this.names[id]}: browser playback unavailable. Python detection continues. Stop, then select Detection view as a fallback.`;
    this.get(id + "-playback-note").textContent = message;
    this.onError(message); // A failed codec or autoplay affects this panel only.
  }

  waitReady(id, entry) {
    if (entry.failed || entry.video.readyState >= 2) return Promise.resolve();
    return new Promise(resolve => {
      const finish = failed => {
        clearTimeout(timeout);
        entry.video.removeEventListener("loadeddata", loaded);
        entry.video.removeEventListener("error", error);
        entry.cancelReady = null;
        if (failed) this.playbackError(id, entry);
        resolve();
      };
      const loaded = () => finish(false), error = () => finish(true);
      const timeout = setTimeout(() => finish(true), 8000);
      entry.cancelReady = () => finish(false);
      entry.video.addEventListener("loadeddata", loaded, {once:true});
      entry.video.addEventListener("error", error, {once:true});
    });
  }

  async prepare(cameras) {
    const ready = [];
    for (const camera of cameras) {
      if (camera.error) { this.release(camera.id); continue; }
      if (camera.media_url) this.setSource(camera.id, camera.media_url);
      const entry = this.entries.get(camera.id);
      if (entry) ready.push(this.waitReady(camera.id, entry));
      else {
        const message = `${this.names[camera.id]}: no playable media reference. Restart the server and hard-refresh, or stop and use Detection view.`;
        this.get(camera.id + "-playback-note").textContent = message;
        this.onError(message);
      }
    }
    // Failed raw feeds do not prevent the other feeds, or Python, from starting.
    await Promise.all(ready);
  }

  playTogether(delayMs) {
    clearTimeout(this.startTimer);
    this.startTimer = setTimeout(() => {
      this.startTimer = null;
      for (const [id, entry] of this.entries) {
        if (entry.failed) continue;
        entry.video.currentTime = 0;
        entry.video.playbackRate = 1;
        entry.video.autoplay = true; // Enabled only for the run initiated by Start.
        entry.video.hidden = false;
        this.get(id + "-empty").hidden = true;
        // Issue all play calls in the same callback, without awaiting each camera.
        entry.video.play().catch(() => this.playbackError(id, entry));
      }
    }, delayMs);
  }

  renderFrame(message) {
    const id = message.camera_id, image = this.get(id + "-frame");
    if (!this.live) {
      this.get(id + "-video").hidden = true;
      if (message.image) {
        image.src = "data:image/jpeg;base64," + message.image;
        image.hidden = false; this.get(id + "-empty").hidden = true;
      }
      return;
    }
    image.hidden = true; // Even an unexpected JPEG can never overlay Live view.
    const entry = this.entries.get(id);
    if (!entry || entry.failed || entry.video.paused || entry.video.readyState < 2) return;
    const target = message.playback_seconds, video = entry.video;
    if (!Number.isFinite(target)) return;
    let drift = Math.abs(video.currentTime - target);
    if (this.loop && Number.isFinite(video.duration) && video.duration > 0) {
      drift = Math.min(drift, Math.abs(video.duration - drift));
    }
    // Use the current Python playback clock, not an old inference frame's time.
    // Small differences are left alone so detection ticks do not make video jump.
    if (drift > 1 && Number.isFinite(video.duration) && video.duration > 0) {
      video.currentTime = Math.min(target, Math.max(0, video.duration - .05));
    }
  }

  release(id) {
    const entry = this.entries.get(id);
    if (!entry) return;
    entry.cancelReady?.();
    entry.video.onerror = null; entry.video.onended = null;
    entry.video.pause(); entry.video.hidden = true;
    entry.video.removeAttribute("src"); entry.video.load();
    if (entry.objectURL) URL.revokeObjectURL(entry.url);
    this.entries.delete(id);
  }

  clear() {
    clearTimeout(this.startTimer); this.startTimer = null;
    for (const id of [...this.entries.keys()]) this.release(id);
  }
}
