// RollCall Interactive Photo Workbench (Zero-CDN Modular ES6 Client)

const state = {
  session: null,
  identities: [],
  selectedFaceIndex: null,
  activeFilter: "all",
  zoom: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  didDrag: false,
  loadedImage: null,
};

const el = (id) => document.getElementById(id);

function showToast(message) {
  const area = el("toast-area");
  const node = document.createElement("div");
  node.className = "toast";
  node.textContent = message;
  area.appendChild(node);
  setTimeout(() => node.remove(), 3500);
}

function setLoading(visible, text = "Processing...") {
  el("loading-text").textContent = text;
  el("loading-overlay").classList.toggle("hidden", !visible);
}

function openModal(id) {
  el(id).classList.remove("hidden");
}

function closeModal(id) {
  el(id).classList.add("hidden");
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("rollcall_theme", theme);
  el("theme-icon").innerHTML = theme === "dark" ? "&#9728;" : "&#9790;";
}

async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    el("enrolled-pill-text").textContent = `${data.enrolled_count} Enrolled`;
    if (!data.model_ready || (data.self_check && !data.self_check.ok)) {
      const checkDetails = data.self_check
        ? Object.entries(data.self_check.checks)
            .filter(([, c]) => !c.ok)
            .map(([name, c]) => `${name}: ${c.detail}`)
            .join(" | ")
        : "";
      el("model-alert").textContent =
        checkDetails || data.model_error || "Startup self-check failed. Run scripts/setup_models.py.";
      el("model-alert").classList.remove("hidden");
    } else {
      el("model-alert").classList.add("hidden");
    }
  } catch (err) {
    el("model-alert").textContent = "Cannot connect to local RollCall server on 127.0.0.1.";
    el("model-alert").classList.remove("hidden");
  }
}

async function refreshIdentities() {
  const res = await fetch("/api/identities");
  const data = await res.json();
  state.identities = data.identities || [];
  el("enrolled-pill-text").textContent = `${state.identities.length} Enrolled`;
  el("roster-count").textContent = String(state.identities.length);

  const chips = el("roster-chips");
  chips.innerHTML = "";
  if (state.identities.length === 0) {
    chips.innerHTML = '<span class="photo-meta">No identities enrolled yet.</span>';
  } else {
    for (const person of state.identities) {
      const chip = document.createElement("span");
      chip.className = "roster-chip";
      chip.textContent =
        person.photo_count > 1
          ? `${person.display_name} (${person.photo_count})`
          : person.display_name;
      chips.appendChild(chip);
    }
  }

  // Update assign dropdown
  const select = el("assign-select");
  select.innerHTML = '<option value="">Assign to enrolled person...</option>';
  for (const person of state.identities) {
    const opt = document.createElement("option");
    opt.value = person.identity_id;
    opt.textContent = person.display_name;
    select.appendChild(opt);
  }
}

function renderEnrollmentSummary(summary) {
  const box = el("enroll-summary");
  box.classList.remove("hidden");
  const warnCount = (summary.warnings || []).length;
  const errCount = (summary.errors || []).length;
  el("warning-dot").classList.toggle("hidden", warnCount === 0 && errCount === 0);

  let html = `
    <strong>Enrollment Complete:</strong>
    ${summary.enrolled_photos} photo(s) enrolled across <strong>${summary.unique_identities}</strong> identities
    (Cache hits: ${summary.cache_hits}, RAM: ${summary.memory_rss_mb} MB).
  `;

  if (warnCount > 0) {
    html += `<div class="enroll-warn-block"><strong>Warnings (${warnCount}):</strong><ul>`;
    for (const w of summary.warnings) {
      html += `<li>${escapeHtml(w.filename)}: ${escapeHtml(w.reason)}</li>`;
    }
    html += `</ul></div>`;
  }
  if (errCount > 0) {
    html += `<div class="enroll-err-block"><strong>Skipped (${errCount}):</strong><ul>`;
    for (const e of summary.errors) {
      html += `<li>${escapeHtml(e.filename)}: ${escapeHtml(e.reason)}</li>`;
    }
    html += `</ul></div>`;
  }
  box.innerHTML = html;

  renderMultiFaceChoices(summary.multiface_pending || []);
}

function renderMultiFaceChoices(items) {
  const wrap = el("multiface-section");
  wrap.innerHTML = "";
  if (!items.length) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  const heading = document.createElement("h3");
  heading.className = "multiface-heading";
  heading.textContent = `Multi-Face Reference Photos (${items.length}) — Select the target person:`;
  wrap.appendChild(heading);

  for (const item of items) {
    const card = document.createElement("div");
    card.className = "multiface-card";
    card.innerHTML = `<div><strong>${escapeHtml(item.display_name)}</strong> <span class="photo-meta">(${escapeHtml(item.filename)} — ${item.face_count} faces)</span></div>`;
    const grid = document.createElement("div");
    grid.className = "multiface-grid";

    for (const cand of item.candidates) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "candidate-btn";
      btn.innerHTML = `<img src="${cand.crop_b64}" alt="Face option ${cand.candidate_index + 1}" /><span>Face #${cand.candidate_index + 1}</span>`;
      btn.addEventListener("click", async () => {
        await resolveMultiFace(item.token, cand.candidate_index, card);
      });
      grid.appendChild(btn);
    }

    const skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "btn btn-sm btn-ghost";
    skipBtn.textContent = "Skip Photo";
    skipBtn.addEventListener("click", async () => {
      await resolveMultiFace(item.token, null, card);
    });
    grid.appendChild(skipBtn);

    card.appendChild(grid);
    wrap.appendChild(card);
  }
}

async function resolveMultiFace(token, candidateIndex, cardElement) {
  const res = await fetch("/api/enroll/resolve-multiface", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, candidate_index: candidateIndex }),
  });
  if (res.ok) {
    cardElement.remove();
    if (!el("multiface-section").querySelector(".multiface-card")) {
      el("multiface-section").classList.add("hidden");
    }
    await refreshIdentities();
    showToast(candidateIndex === null ? "Skipped multi-face photo." : "Selected face enrolled.");
  }
}

function pollEnrollJob(jobId) {
  const progWrap = el("enroll-progress-wrap");
  const poll = setInterval(async () => {
    const stRes = await fetch(`/api/enroll/${jobId}`);
    const st = await stRes.json();
    const pct = st.total > 0 ? Math.round((st.processed / st.total) * 100) : 100;
    el("enroll-progress-bar").value = pct;
    el("enroll-progress-pct").textContent = `${pct}%`;
    el("enroll-progress-label").textContent =
      st.status === "completed" ? "Enrollment complete" : `Processing ${st.current_file || "..."}`;

    if (st.status === "completed") {
      clearInterval(poll);
      renderEnrollmentSummary(st.summary);
      await refreshIdentities();
      showToast(`Enrolled ${st.summary.unique_identities} identities.`);
    } else if (st.status === "failed") {
      clearInterval(poll);
      progWrap.classList.add("hidden");
      showToast(st.error || "Enrollment failed.");
    }
  }, 200);
}

async function handleFolderSelect(fileList) {
  const validFiles = Array.from(fileList).filter((f) => !f.name.startsWith("."));
  if (!validFiles.length) return;

  const form = new FormData();
  for (const file of validFiles) {
    form.append("files", file, file.name);
  }
  form.append("clear_existing", el("chk-clear-existing").checked ? "true" : "false");
  form.append("persist_cache", el("chk-persist-cache").checked ? "true" : "false");

  const progWrap = el("enroll-progress-wrap");
  progWrap.classList.remove("hidden");
  el("enroll-progress-bar").value = 5;
  el("enroll-progress-label").textContent = `Uploading ${validFiles.length} file(s)...`;
  el("enroll-progress-pct").textContent = "5%";

  const res = await fetch("/api/enroll", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    showToast(err.detail || "Enrollment failed.");
    progWrap.classList.add("hidden");
    return;
  }

  const { job_id } = await res.json();
  pollEnrollJob(job_id);
}

async function handleFolderPathEnroll() {
  const folderPath = el("folder-path-input").value.trim();
  if (!folderPath) {
    showToast("Paste a local folder path first.");
    return;
  }

  const progWrap = el("enroll-progress-wrap");
  progWrap.classList.remove("hidden");
  el("enroll-progress-bar").value = 5;
  el("enroll-progress-label").textContent = "Reading local folder...";
  el("enroll-progress-pct").textContent = "5%";

  const res = await fetch("/api/enroll/from-path", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      folder_path: folderPath,
      clear_existing: el("chk-clear-existing").checked,
      persist_cache: el("chk-persist-cache").checked,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    showToast(err.detail || "Could not load folder path.");
    progWrap.classList.add("hidden");
    return;
  }

  const { job_id } = await res.json();
  pollEnrollJob(job_id);
}

async function handleGroupUpload(file) {
  if (!file) return;
  setLoading(true, `Analyzing ${file.name}...`);
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const res = await fetch("/api/identify", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Could not process group photo.");
      return;
    }
    loadSessionIntoUI(data);
  } finally {
    setLoading(false);
  }
}

async function runDemoMode() {
  setLoading(true, "Running Demo Mode through local YuNet + SFace pipeline...");
  try {
    const res = await fetch("/api/demo", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Demo Mode data unavailable.");
      return;
    }
    if (data.demo_enrollment) {
      renderEnrollmentSummary(data.demo_enrollment);
    }
    await refreshIdentities();
    loadSessionIntoUI(data);
    showToast("Demo Mode finished using the live recognition pipeline.");
  } finally {
    setLoading(false);
  }
}

function loadSessionIntoUI(data) {
  state.session = data;
  state.selectedFaceIndex = data.faces.length ? data.faces[0].face_index : null;
  state.activeFilter = "all";
  updateFilterChips();

  el("photo-title").textContent = data.filename;
  const timeTxt = data.processing_time_ms ? `${data.processing_time_ms} ms` : "";
  el("photo-meta").textContent = `${data.width}×${data.height}px ${timeTxt ? "• " + timeTxt : ""} • RAM: ${data.memory_rss_mb} MB`;

  ["btn-zoom-out", "btn-zoom-in", "btn-zoom-reset", "btn-export-png", "btn-export-csv"].forEach(
    (id) => { el(id).disabled = false; }
  );

  if (data.image_b64) {
    const img = new Image();
    img.onload = () => {
      state.loadedImage = img;
      resetZoom();
      drawCanvas();
    };
    img.src = data.image_b64;
  } else {
    drawCanvas();
  }

  updateSummaryAndList();
  renderExplainabilityPanel();
}

function updateSummaryAndList() {
  const s = state.session?.summary || {
    total_detected: 0,
    confirmed: 0,
    likely: 0,
    unknown: 0,
    absent_count: 0,
  };
  el("stat-detected").textContent = String(s.total_detected);
  el("stat-confirmed").textContent = String(s.confirmed);
  el("stat-likely").textContent = String(s.likely);
  el("stat-unknown").textContent = String(s.unknown);
  el("stat-absent").textContent = String(s.absent_count);

  const list = el("results-list");
  list.innerHTML = "";
  if (!state.session) {
    list.innerHTML = '<div class="empty-list-msg">Upload a group photo to see detected faces and attendance status.</div>';
    return;
  }

  if (state.activeFilter === "absent") {
    const absent = state.session.absent_identities || [];
    if (!absent.length) {
      list.innerHTML = '<div class="empty-list-msg">Everyone enrolled is confirmed present!</div>';
      return;
    }
    for (const person of absent) {
      const row = document.createElement("div");
      row.className = "face-card";
      row.innerHTML = `
        <div class="face-card-main">
          <span class="face-card-name">${escapeHtml(person.display_name)}</span>
          <span class="face-card-sub">${person.pending_review ? "Possible match pending review" : "Not confirmed in group photo"}</span>
        </div>
        <span class="status-badge ${person.pending_review ? "badge-Likely" : "badge-Absent"}">
          ${person.pending_review ? "Needs Review" : "Absent"}
        </span>
      `;
      list.appendChild(row);
    }
    return;
  }

  const filtered = state.session.faces.filter(
    (f) => state.activeFilter === "all" || f.status === state.activeFilter
  );

  if (!filtered.length) {
    list.innerHTML = '<div class="empty-list-msg">No faces match this filter.</div>';
    return;
  }

  for (const face of filtered) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "face-card" + (face.face_index === state.selectedFaceIndex ? " selected" : "");
    btn.setAttribute("role", "listitem");

    const badgeLabel = face.status === "Likely" ? "Needs Review" : face.status;
    const simTxt = face.similarity !== null ? `Sim: ${face.similarity.toFixed(3)}` : "Quality gate";
    btn.innerHTML = `
      <img class="face-thumb" src="${face.crop_b64}" alt="Face #${face.face_index + 1}" />
      <div class="face-card-main">
        <span class="face-card-name">#${face.face_index + 1} ${escapeHtml(face.display_name)}</span>
        <span class="face-card-sub">${simTxt} &bull; ${face.quality.width_px}×${face.quality.height_px}px</span>
      </div>
      <span class="status-badge badge-${face.status}">${badgeLabel}</span>
    `;
    btn.addEventListener("click", () => selectFace(face.face_index));
    list.appendChild(btn);
  }
}

function selectFace(faceIndex) {
  state.selectedFaceIndex = faceIndex;
  updateSummaryAndList();
  renderExplainabilityPanel();
  drawCanvas();
}

function renderExplainabilityPanel() {
  const panel = el("explain-panel");
  if (!state.session || state.selectedFaceIndex === null) {
    panel.classList.add("hidden");
    return;
  }
  const face = state.session.faces.find((f) => f.face_index === state.selectedFaceIndex);
  if (!face) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  el("explain-face-num").textContent = `Face #${face.face_index + 1}`;
  el("explain-title").textContent = face.display_name;

  const badge = el("explain-badge");
  badge.className = `status-badge badge-${face.status}`;
  badge.textContent = face.status === "Likely" ? "Likely (Needs Review)" : face.status;

  el("explain-group-crop").src = face.crop_b64;

  const refImg = el("explain-ref-crop");
  const refPh = el("explain-ref-placeholder");
  if (face.matched_photo_id) {
    refImg.src = `/api/reference-crop/${face.matched_photo_id}`;
    refImg.classList.remove("hidden");
    refPh.classList.add("hidden");
    el("explain-ref-label").textContent = "Matched Reference";
  } else {
    refImg.classList.add("hidden");
    refPh.classList.remove("hidden");
    el("explain-ref-label").textContent = "No Reference Match";
  }

  el("explain-sim").textContent = face.similarity !== null ? face.similarity.toFixed(4) : "N/A";
  el("explain-margin").textContent =
    face.runner_up_margin !== null ? `+${face.runner_up_margin.toFixed(4)}` : "N/A";
  el("explain-size").textContent = `${face.quality.width_px} × ${face.quality.height_px} px`;
  el("explain-quality").textContent = `Blur ${face.quality.blur_laplacian} • Det ${face.quality.detector_score}`;

  el("explain-runnerup").textContent = face.runner_up_name
    ? `Runner-up candidate: ${face.runner_up_name} (${face.runner_up_similarity?.toFixed(4) ?? "—"})`
    : "No runner-up candidate above floor.";
  el("explain-reason").textContent = face.decision_reason;

  el("btn-confirm-match").classList.toggle("hidden", face.status !== "Likely");
}

async function submitCorrection(payload) {
  if (!state.session || state.selectedFaceIndex === null) return;
  const res = await fetch(`/api/sessions/${state.session.session_id}/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ face_index: state.selectedFaceIndex, ...payload }),
  });
  const data = await res.json();
  if (!res.ok) {
    showToast(data.detail || "Could not update identification.");
    return;
  }
  state.session = { ...state.session, ...data };
  await refreshIdentities();
  updateSummaryAndList();
  renderExplainabilityPanel();
  drawCanvas();
  showToast("Session identification updated.");
}

// Canvas Rendering, Zoom & Hit Testing
function getCanvasMetrics() {
  const canvas = el("photo-canvas");
  const img = state.loadedImage;
  if (!img) return null;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
  }
  const baseScale = Math.min(canvas.width / img.width, canvas.height / img.height) * 0.94;
  const scale = baseScale * state.zoom;
  const drawW = img.width * scale;
  const drawH = img.height * scale;
  const offsetX = (canvas.width - drawW) / 2 + state.panX;
  const offsetY = (canvas.height - drawH) / 2 + state.panY;
  return { canvas, img, scale, offsetX, offsetY, drawW, drawH };
}

function resetZoom() {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  el("zoom-level").textContent = "100%";
  drawCanvas();
}

function drawCanvas() {
  const hasImg = Boolean(state.loadedImage);
  el("empty-dropzone").classList.toggle("hidden", hasImg);
  el("photo-canvas").classList.toggle("hidden", !hasImg);
  if (!hasImg) return;

  const m = getCanvasMetrics();
  if (!m) return;
  const ctx = m.canvas.getContext("2d");
  ctx.clearRect(0, 0, m.canvas.width, m.canvas.height);
  ctx.drawImage(m.img, m.offsetX, m.offsetY, m.drawW, m.drawH);

  const palette = {
    Confirmed: "#10b981",
    Likely: "#f59e0b",
    unknown: "#ef4444",
  };

  for (const face of state.session.faces) {
    const [bx, by, bw, bh] = face.bbox;
    const x = m.offsetX + bx * m.scale;
    const y = m.offsetY + by * m.scale;
    const w = bw * m.scale;
    const h = bh * m.scale;
    const isSelected = face.face_index === state.selectedFaceIndex;
    const color = palette[face.status] || "#94a3b8";

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 3.5 : 2;
    if (face.status === "Likely") ctx.setLineDash([6, 4]);
    if (face.status === "unknown") ctx.setLineDash([3, 3]);
    ctx.strokeRect(x, y, w, h);

    if (isSelected) {
      ctx.setLineDash([]);
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 4, y - 4, w + 8, h + 8);
    }

    const prefix = face.status === "Confirmed" ? "✓" : face.status === "Likely" ? "?" : "!";
    const label = `${prefix} #${face.face_index + 1} ${face.display_name}`;
    ctx.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
    const textW = ctx.measureText(label).width;
    const tagH = 20;
    const tagY = Math.max(0, y - tagH - 2);

    ctx.fillStyle = color;
    ctx.fillRect(x, tagY, textW + 12, tagH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 6, tagY + 14);
    ctx.restore();
  }
}

function updateFilterChips() {
  document.querySelectorAll(".filter-chip").forEach((btn) => {
    const active = btn.dataset.filter === state.activeFilter;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function bindEvents() {
  applyTheme(localStorage.getItem("rollcall_theme") || "light");
  el("btn-theme").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  el("btn-open-enroll").addEventListener("click", () => openModal("enroll-modal"));
  el("btn-empty-enroll").addEventListener("click", () => openModal("enroll-modal"));
  el("btn-wipe").addEventListener("click", () => openModal("wipe-modal"));

  document.querySelectorAll("[data-close]").forEach((node) => {
    node.addEventListener("click", () => closeModal(node.dataset.close));
  });

  el("folder-input").addEventListener("change", (e) => {
    handleFolderSelect(e.target.files);
    e.target.value = "";
  });

  el("btn-enroll-path").addEventListener("click", handleFolderPathEnroll);

  el("group-file-input").addEventListener("change", (e) => {
    handleGroupUpload(e.target.files[0]);
    e.target.value = "";
  });

  el("btn-demo").addEventListener("click", runDemoMode);
  el("btn-empty-demo").addEventListener("click", runDemoMode);

  el("btn-confirm-wipe").addEventListener("click", async () => {
    const res = await fetch("/api/wipe", { method: "POST" });
    if (res.ok) {
      state.session = null;
      state.loadedImage = null;
      state.selectedFaceIndex = null;
      el("photo-title").textContent = "No group photo loaded";
      el("photo-meta").textContent = "";
      el("enroll-summary").classList.add("hidden");
      el("warning-dot").classList.add("hidden");
      ["btn-zoom-out", "btn-zoom-in", "btn-zoom-reset", "btn-export-png", "btn-export-csv"].forEach(
        (id) => { el(id).disabled = true; }
      );
      closeModal("wipe-modal");
      await refreshIdentities();
      updateSummaryAndList();
      renderExplainabilityPanel();
      drawCanvas();
      showToast("All local identities, embeddings, and session data wiped.");
    }
  });

  document.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.activeFilter = chip.dataset.filter;
      updateFilterChips();
      updateSummaryAndList();
    });
  });

  el("btn-confirm-match").addEventListener("click", () => submitCorrection({ action: "confirm" }));
  el("btn-mark-unknown").addEventListener("click", () => submitCorrection({ action: "mark_unknown" }));
  el("btn-assign-person").addEventListener("click", () => {
    const iid = el("assign-select").value;
    if (!iid) return showToast("Choose an enrolled identity first.");
    submitCorrection({ action: "assign", identity_id: iid });
  });
  el("btn-enroll-new").addEventListener("click", () => {
    const name = el("enroll-new-name").value.trim();
    if (!name) return showToast("Enter a person's name first.");
    submitCorrection({ action: "enroll_new", new_display_name: name });
    el("enroll-new-name").value = "";
  });

  el("btn-export-png").addEventListener("click", () => {
    if (state.session) window.location.href = `/api/sessions/${state.session.session_id}/export/png`;
  });
  el("btn-export-csv").addEventListener("click", () => {
    if (state.session) window.location.href = `/api/sessions/${state.session.session_id}/export/csv`;
  });

  // Zoom & Pan
  el("btn-zoom-in").addEventListener("click", () => {
    state.zoom = Math.min(4, +(state.zoom + 0.25).toFixed(2));
    el("zoom-level").textContent = `${Math.round(state.zoom * 100)}%`;
    drawCanvas();
  });
  el("btn-zoom-out").addEventListener("click", () => {
    state.zoom = Math.max(0.5, +(state.zoom - 0.25).toFixed(2));
    el("zoom-level").textContent = `${Math.round(state.zoom * 100)}%`;
    drawCanvas();
  });
  el("btn-zoom-reset").addEventListener("click", resetZoom);

  const canvas = el("photo-canvas");
  canvas.addEventListener("mousedown", (e) => {
    state.isPanning = true;
    state.didDrag = false;
    state.panStartX = e.clientX - state.panX;
    state.panStartY = e.clientY - state.panY;
  });
  window.addEventListener("mousemove", (e) => {
    if (!state.isPanning) return;
    state.didDrag = true;
    state.panX = e.clientX - state.panStartX;
    state.panY = e.clientY - state.panStartY;
    drawCanvas();
  });
  window.addEventListener("mouseup", (e) => {
    if (!state.isPanning) return;
    state.isPanning = false;
    if (!state.didDrag && state.session) {
      const m = getCanvasMetrics();
      if (!m) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      for (const face of state.session.faces) {
        const [bx, by, bw, bh] = face.bbox;
        const x = m.offsetX + bx * m.scale;
        const y = m.offsetY + by * m.scale;
        if (cx >= x && cx <= x + bw * m.scale && cy >= y && cy <= y + bh * m.scale) {
          selectFace(face.face_index);
          break;
        }
      }
    }
  });

  // Drag & Drop group photo onto stage
  const stage = el("canvas-stage");
  stage.addEventListener("dragover", (e) => {
    e.preventDefault();
    stage.classList.add("drag-over");
  });
  stage.addEventListener("dragleave", () => stage.classList.remove("drag-over"));
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    stage.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleGroupUpload(file);
  });

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeModal("enroll-modal");
      closeModal("wipe-modal");
      return;
    }
    if (!state.session || !state.session.faces.length) return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;

    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      const idx = state.selectedFaceIndex === null ? 0 : (state.selectedFaceIndex + 1) % state.session.faces.length;
      selectFace(idx);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      const n = state.session.faces.length;
      const idx = state.selectedFaceIndex === null ? 0 : (state.selectedFaceIndex - 1 + n) % n;
      selectFace(idx);
    }
  });

  window.addEventListener("resize", drawCanvas);
}

bindEvents();
fetchHealth();
refreshIdentities();
