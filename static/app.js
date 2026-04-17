const form = document.querySelector("#startForm");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const videoFeed = document.querySelector("#videoFeed");
const videoStage = document.querySelector(".video-stage");
const uploadField = document.querySelector(".upload-field");
const liveField = document.querySelector(".live-field");
const sourceRadios = document.querySelectorAll("input[name='source_type']");
const alertsList = document.querySelector("#alertsList");

let sessionId = null;
let metricsTimer = null;

sourceRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    const mode = document.querySelector("input[name='source_type']:checked").value;
    uploadField.classList.toggle("hidden", mode !== "upload");
    liveField.classList.toggle("hidden", mode !== "live");
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  startBtn.disabled = true;
  startBtn.textContent = "Starting...";

  try {
    if (sessionId) {
      await stopSession();
    }

    const response = await fetch("/api/start", {
      method: "POST",
      body: new FormData(form),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Could not start analysis.");
    }

    sessionId = payload.session_id;
    videoFeed.src = `/video_feed/${sessionId}?t=${Date.now()}`;
    videoStage.classList.add("active");
    stopBtn.disabled = false;
    metricsTimer = window.setInterval(refreshMetrics, 700);
    await refreshMetrics();
  } catch (error) {
    alert(error.message);
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = "Start analysis";
  }
});

stopBtn.addEventListener("click", async () => {
  await stopSession();
});

async function refreshMetrics() {
  if (!sessionId) {
    return;
  }
  const response = await fetch(`/api/metrics/${sessionId}`);
  if (!response.ok) {
    return;
  }
  const metrics = await response.json();
  document.querySelector("#vehicles").textContent = metrics.vehicles;
  document.querySelector("#totalCount").textContent = metrics.total_count;
  document.querySelector("#avgSpeed").textContent = metrics.avg_speed_kmh.toFixed(1);
  document.querySelector("#overspeeding").textContent = metrics.overspeeding;
  document.querySelector("#wrongWay").textContent = metrics.wrong_way;
  document.querySelector("#uturns").textContent = metrics.illegal_uturns;
  document.querySelector("#congestionLevel").textContent = metrics.congestion_level;
  document.querySelector("#overallConfidence").textContent = metrics.overall_confidence_pct.toFixed(1);
  document.querySelector("#alertConfidence").textContent = metrics.alert_confidence_pct.toFixed(1);

  alertsList.innerHTML = "";
  if (!metrics.alerts || metrics.alerts.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No alerts yet.";
    alertsList.appendChild(item);
  } else {
    metrics.alerts.forEach((alertText) => {
      const item = document.createElement("li");
      item.textContent = alertText;
      alertsList.appendChild(item);
    });
  }
}

async function stopSession() {
  if (metricsTimer) {
    window.clearInterval(metricsTimer);
    metricsTimer = null;
  }

  if (sessionId) {
    await fetch(`/api/stop/${sessionId}`, { method: "POST" });
  }

  sessionId = null;
  videoFeed.removeAttribute("src");
  videoStage.classList.remove("active");
  stopBtn.disabled = true;
}
