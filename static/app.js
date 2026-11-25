// RollCall Interactive Photo Workbench (Zero-CDN Modular ES6 Client)

const state = {
  session: null,
  identities: [],
  selectedFaceIndex: null,
  activeFilter: "all",
  reviewQueueActive: false,
  drawMode: false,
  isDrawing: false,
  drawStartImg: null,
  drawRectImg: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  didDrag: false,
  loadedImage: null,
  coverageMapOpen: window.innerWidth > 960,
};

function statusCssClass(status) {
  return status === "Confirmed by reviewer" ? "ConfirmedReviewer" : status;
}

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
}

function renderFlaggedPairsHtml(cal) {
  if (!cal) return "";
  const dups = cal.possible_duplicate_pairs || [];
  const looks = cal.lookalike_pairs || [];
  let html = "";
  if (dups.length > 0) {
    html += `<div class="enroll-err-block"><strong>Possible Duplicate Identity (&ge; 0.55) (${dups.length}):</strong><ul>`;
    for (const p of dups) {
      html += `<li>${escapeHtml(p.identity_a_name)} &harr; ${escapeHtml(p.identity_b_name)} (sim: ${p.similarity.toFixed(4)})</li>`;
    }
    html += `</ul></div>`;
  }
  if (looks.length > 0) {
    html += `<div class="enroll-warn-block"><strong>Lookalike Pairs (extra margin ${cal.lookalike_margin.toFixed(3)} vs base ${cal.base_margin.toFixed(3)}) (${looks.length}):</strong><ul>`;
    for (const p of looks) {
      html += `<li>${escapeHtml(p.identity_a_name)} &harr; ${escapeHtml(p.identity_b_name)} (sim: ${p.similarity.toFixed(4)})</li>`;
    }
    html += `</ul></div>`;
  }
  return html;
}

let activeHealthFilter = "all";
let latestEnrollSummary = null;
let pendingBannerDismissed = false;

function updatePendingAttentionBanner(summary) {
  const banner = el("pending-attention-banner");
  const multifaceCount = (summary?.multiface_pending || []).length;
  const extraCount = (summary?.extra_photo_pending || []).length;
  const pendingCount = summary?.counts?.needs_attention ?? (multifaceCount + extraCount);
  if (pendingCount > 0 && !pendingBannerDismissed) {
    const parts = [];
    if (multifaceCount > 0) {
      const noun = multifaceCount === 1 ? "person is" : "people are";
      parts.push(`${multifaceCount} ${noun} not enrolled until you pick their face`);
    }
    if (extraCount > 0) {
      const enoun = extraCount === 1 ? "extra photo is" : "extra photos are";
      parts.push(`${extraCount} ${enoun} held out of matching ("may be a different person") until you keep or reject`);
    }
    el("pending-attention-text").textContent = parts.length
      ? `${parts.join("; ")}.`
      : `${pendingCount} item(s) need attention.`;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

function renderHealthFileList(records) {
  const listWrap = el("health-file-list");
  if (!listWrap) return;
  const filtered = records.filter(
    (r) => activeHealthFilter === "all" || r.status === activeHealthFilter
  );
  if (!filtered.length) {
    listWrap.innerHTML = '<div class="photo-meta">No files in this status category.</div>';
    return;
  }
  const labelMap = {
    enrolled: "Enrolled",
    enrolled_with_warnings: "Warning",
    needs_attention: "Needs Attention",
    failed: "Failed",
    ignored: "Ignored",
  };
  listWrap.innerHTML = "";
  for (const rec of filtered) {
    const row = document.createElement("div");
    row.className = "health-row";
    const thumbHtml = rec.thumb_b64
      ? `<img class="health-thumb" src="${rec.thumb_b64}" alt="${escapeHtml(rec.filename)}" />`
      : `<div class="health-thumb-ph">${rec.status === "ignored" ? "SYS" : "FILE"}</div>`;
    const titleTxt = rec.display_name ? `${rec.filename} → ${rec.display_name}` : rec.filename;
    row.innerHTML = `
      ${thumbHtml}
      <div class="health-row-main">
        <span class="health-row-title">${escapeHtml(titleTxt)}</span>
        <span class="health-row-reason">${escapeHtml(rec.reason || "")}</span>
      </div>
      <span class="status-badge badge-${rec.status}">${labelMap[rec.status] || rec.status}</span>
    `;
    listWrap.appendChild(row);
  }
}

function renderEnrollmentSummary(summary, resetDismiss = true) {
  latestEnrollSummary = summary;
  if (resetDismiss) pendingBannerDismissed = false;
  updatePendingAttentionBanner(summary);
  const box = el("enroll-summary");
  box.classList.remove("hidden");
  const counts = summary.counts || {
    selected: summary.total_files,
    enrolled: summary.enrolled_photos,
    enrolled_with_warnings: 0,
    needs_attention: (summary.multiface_pending || []).length,
    failed: (summary.errors || []).length,
    ignored: 0,
    headline: `${summary.total_files} selected`,
  };
  const attentionTotal =
    counts.enrolled_with_warnings + counts.needs_attention + counts.failed;
  const flaggedCount =
    ((summary.calibration?.possible_duplicate_pairs || []).length) +
    ((summary.calibration?.lookalike_pairs || []).length);
  el("warning-dot").classList.toggle("hidden", attentionTotal === 0 && flaggedCount === 0);

  let html = `
    <div class="health-headline">${escapeHtml(counts.headline)}</div>
    <div class="photo-meta">Unique identities: <strong>${summary.unique_identities}</strong> &bull; Cache hits: ${summary.cache_hits} &bull; RAM: ${summary.memory_rss_mb} MB</div>
  `;

  html += renderFlaggedPairsHtml(summary.calibration);

  const records = summary.file_records || [];
  if (records.length > 0) {
    const tabs = [
      ["all", `All (${counts.selected})`],
      ["enrolled", `Enrolled (${counts.enrolled})`],
      ["enrolled_with_warnings", `Warnings (${counts.enrolled_with_warnings})`],
      ["needs_attention", `Needs Attention (${counts.needs_attention})`],
      ["failed", `Failed (${counts.failed})`],
      ["ignored", `Ignored (${counts.ignored})`],
    ];
    html += `<div class="health-filters" id="health-filters">`;
    for (const [key, label] of tabs) {
      const cls = activeHealthFilter === key ? "filter-chip active" : "filter-chip";
      html += `<button type="button" class="${cls}" data-health-filter="${key}">${label}</button>`;
    }
    html += `</div><div id="health-file-list" class="health-list"></div>`;
  }

  box.innerHTML = html;

  if (records.length > 0) {
    box.querySelectorAll("[data-health-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeHealthFilter = btn.dataset.healthFilter;
        box.querySelectorAll("[data-health-filter]").forEach((b) =>
          b.classList.toggle("active", b.dataset.healthFilter === activeHealthFilter)
        );
        renderHealthFileList(records);
      });
    });
    renderHealthFileList(records);
  }

  renderMultiFaceChoices(summary.multiface_pending || []);
  renderExtraPhotoChoices(summary.extra_photo_pending || []);
}

function renderExtraPhotoChoices(items) {
  const wrap = el("extra-review-section");
  if (!wrap) return;
  wrap.innerHTML = "";
  if (!items.length) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  const heading = document.createElement("h3");
  heading.className = "multiface-heading";
  heading.textContent = `Needs Attention — Extra Photos (${items.length}) (may be a different person):`;
  wrap.appendChild(heading);

  for (const item of items) {
    const card = document.createElement("div");
    card.className = "multiface-card";
    card.innerHTML = `<div><strong>${escapeHtml(item.display_name)}</strong> <span class="photo-meta">(${escapeHtml(item.filename)} — similarity ${item.similarity.toFixed(4)} &lt; T_high ${item.t_high.toFixed(4)} — may be a different person; held out of matching)</span></div>`;

    const row = document.createElement("div");
    row.className = "extra-compare-row";
    row.innerHTML = `
      <div class="extra-thumb-box">
        <img src="${item.existing_thumb_b64 || ""}" alt="Existing enrolled portrait for ${escapeHtml(item.display_name)}" />
        <span>Enrolled (${escapeHtml(item.existing_filename)})</span>
      </div>
      <div class="crop-vs" aria-hidden="true">vs</div>
      <div class="extra-thumb-box">
        <img src="${item.new_thumb_b64 || ""}" alt="Flagged extra photo ${escapeHtml(item.filename)}" />
        <span>New Extra (${escapeHtml(item.filename)})</span>
      </div>
    `;

    const actions = document.createElement("div");
    actions.className = "multiface-actions";

    const keepBtn = document.createElement("button");
    keepBtn.type = "button";
    keepBtn.className = "btn btn-sm btn-primary";
    keepBtn.textContent = "Keep Photo (Add Exemplar)";
    keepBtn.addEventListener("click", () => resolveExtraPhoto(item.token, "keep"));

    const rejectBtn = document.createElement("button");
    rejectBtn.type = "button";
    rejectBtn.className = "btn btn-sm btn-danger-outline";
    rejectBtn.textContent = "Reject Photo";
    rejectBtn.addEventListener("click", () => resolveExtraPhoto(item.token, "reject"));

    actions.appendChild(keepBtn);
    actions.appendChild(rejectBtn);
    card.appendChild(row);
    card.appendChild(actions);
    wrap.appendChild(card);
  }
}

async function resolveExtraPhoto(token, decision) {
  const res = await fetch("/api/enroll/resolve-extra", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, decision }),
  });
  if (res.ok) {
    const data = await res.json();
    if (data.summary) {
      renderEnrollmentSummary(data.summary);
    }
    await refreshIdentities();
    showToast(decision === "keep" ? "Extra photo kept and added as exemplar." : "Extra photo rejected and removed.");
  }
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
  heading.textContent = `Needs Attention — Multi-Face Photos (${items.length}) (choose target face):`;
  wrap.appendChild(heading);

  for (const item of items) {
    let selectedIdx = item.suggested_index ?? 0;
    const card = document.createElement("div");
    card.className = "multiface-card";
    card.innerHTML = `<div><strong>${escapeHtml(item.display_name)}</strong> <span class="photo-meta">(${escapeHtml(item.filename)} — ${item.face_count} faces)</span></div>`;

    const layout = document.createElement("div");
    layout.className = "multiface-layout";

    const previewImg = document.createElement("img");
    previewImg.className = "multiface-preview";
    previewImg.src = item.candidates[selectedIdx].annotated_thumb_b64;
    previewImg.alt = `Boxed preview for ${item.filename}`;
    layout.appendChild(previewImg);

    const controls = document.createElement("div");
    controls.className = "multiface-controls";

    const grid = document.createElement("div");
    grid.className = "multiface-grid";
    const candButtons = [];

    for (const cand of item.candidates) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "candidate-btn" + (cand.candidate_index === selectedIdx ? " active" : "");
      btn.innerHTML = `<img src="${cand.crop_b64}" alt="Face option ${cand.candidate_index + 1}" /><span>Face #${cand.candidate_index + 1}</span>`;
      btn.addEventListener("click", () => {
        selectedIdx = cand.candidate_index;
        previewImg.src = cand.annotated_thumb_b64;
        candButtons.forEach((b, i) => b.classList.toggle("active", i === selectedIdx));
      });
      candButtons.push(btn);
      grid.appendChild(btn);
    }

    const actions = document.createElement("div");
    actions.className = "multiface-actions";

    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "btn btn-sm btn-primary";
    confirmBtn.textContent = "Enroll Selected Face";
    confirmBtn.addEventListener("click", async () => {
      await resolveMultiFace(item.token, selectedIdx);
    });

    const skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "btn btn-sm btn-ghost";
    skipBtn.textContent = "Skip Photo";
    skipBtn.addEventListener("click", async () => {
      await resolveMultiFace(item.token, null);
    });

    actions.appendChild(confirmBtn);
    actions.appendChild(skipBtn);
    controls.appendChild(grid);
    controls.appendChild(actions);
    layout.appendChild(controls);
    card.appendChild(layout);
    wrap.appendChild(card);
  }
}

async function resolveMultiFace(token, candidateIndex) {
  const res = await fetch("/api/enroll/resolve-multiface", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, candidate_index: candidateIndex }),
  });
  if (res.ok) {
    const data = await res.json();
    if (data.summary) {
      renderEnrollmentSummary(data.summary);
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
  const validFiles = Array.from(fileList);
  if (!validFiles.length) return;

  const form = new FormData();
  for (const file of validFiles) {
    form.append("files", file, file.webkitRelativePath || file.name);
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
  el("demo-note-banner").classList.add("hidden");
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
    const demoBanner = el("demo-note-banner");
    if (data.demo_note) {
      demoBanner.textContent = data.demo_note;
      demoBanner.classList.remove("hidden");
    } else {
      demoBanner.classList.add("hidden");
    }
    await refreshIdentities();
    loadSessionIntoUI(data);
    showToast(data.demo_note || "Demo Mode finished using the live recognition pipeline.");
  } finally {
    setLoading(false);
  }
}

function loadSessionIntoUI(data) {
  state.session = data;
  state.reviewQueueActive = false;
  state.drawMode = false;
  state.drawRectImg = null;
  el("photo-canvas").classList.remove("draw-mode");
  el("btn-draw-box").classList.remove("btn-queue-active");
  state.selectedFaceIndex = data.faces.length ? data.faces[0].face_index : null;
  state.activeFilter = "all";
  updateFilterChips();

  el("photo-title").textContent = data.filename;
  const timeTxt = data.processing_time_ms ? `${data.processing_time_ms} ms` : "";
  el("photo-meta").textContent = `${data.width}×${data.height}px ${timeTxt ? "• " + timeTxt : ""} • RAM: ${data.memory_rss_mb} MB`;

  state.coverageMapOpen = window.innerWidth > 960;
  [
    "btn-second-look",
    "btn-draw-box",
    "btn-zoom-out",
    "btn-zoom-in",
    "btn-zoom-reset",
    "btn-toggle-map",
    "btn-export-png",
    "btn-export-csv",
    "btn-print-sheet",
  ].forEach((id) => {
    el(id).disabled = false;
  });

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

function getReviewQueueIndices() {
  if (!state.session) return [];
  return state.session.faces
    .filter((f) => !f.removed && (f.status === "Likely" || f.status === "unknown"))
    .map((f) => f.face_index);
}

function updateQueueBar() {
  const bar = el("queue-bar");
  if (!state.session) {
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");
  const queue = getReviewQueueIndices();
  const btn = el("btn-toggle-queue");
  btn.textContent = state.reviewQueueActive
    ? `Exit Queue (${queue.length})`
    : `Review Queue (${queue.length})`;
  btn.classList.toggle("btn-queue-active", state.reviewQueueActive);

  const prog = el("queue-progress");
  if (state.reviewQueueActive && queue.length > 0) {
    const pos = queue.indexOf(state.selectedFaceIndex);
    prog.textContent = pos >= 0 ? `${pos + 1} of ${queue.length}` : `0 of ${queue.length}`;
  } else if (state.reviewQueueActive) {
    prog.textContent = "All reviewed";
  } else {
    prog.textContent = "";
  }
  el("btn-undo").disabled = !state.session.can_undo;
}

function stepFace(delta, queueOnly = state.reviewQueueActive) {
  if (!state.session || !state.session.faces.length) return;
  const pool = queueOnly
    ? getReviewQueueIndices()
    : state.session.faces.filter((f) => !f.removed).map((f) => f.face_index);
  if (!pool.length) {
    showToast("No faces remaining in Review Queue.");
    return;
  }
  const curPos = pool.indexOf(state.selectedFaceIndex);
  const nextPos = curPos === -1 ? 0 : (curPos + delta + pool.length) % pool.length;
  selectFace(pool[nextPos]);
}

function advanceQueueAfterDecision(prevQueue) {
  const nextQueue = getReviewQueueIndices();
  if (!state.reviewQueueActive) return;
  if (!nextQueue.length) {
    state.reviewQueueActive = false;
    updateQueueBar();
    showToast("Review Queue complete.");
    return;
  }
  // Pick next unreviewed item following previous position
  const prevIdx = state.selectedFaceIndex;
  const candidate = nextQueue.find((idx) => idx > prevIdx) ?? nextQueue[0];
  selectFace(candidate);
}

function renderAssignFilterResults(query) {
  const box = el("assign-results-list");
  const q = (query || "").trim().toLowerCase();
  const matches = state.identities.filter((p) =>
    !q || p.display_name.toLowerCase().includes(q)
  );
  box.innerHTML = "";
  if (!matches.length) {
    box.innerHTML = '<div class="assign-option">No enrolled matches</div>';
    box.classList.remove("hidden");
    return;
  }
  for (const person of matches.slice(0, 25)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "assign-option";
    btn.setAttribute("role", "option");
    btn.textContent = person.display_name;
    btn.addEventListener("click", async () => {
      box.classList.add("hidden");
      el("assign-search-input").value = "";
      await submitCorrection({ action: "assign", identity_id: person.identity_id }, true);
    });
    box.appendChild(btn);
  }
  box.classList.remove("hidden");
}

function updateSummaryAndList() {
  const s = state.session?.summary || {
    total_detected: 0,
    confirmed: 0,
    confirmed_by_reviewer: 0,
    likely: 0,
    unknown: 0,
    absent_count: 0,
  };
  el("stat-detected").textContent = String(s.total_detected);
  el("stat-confirmed").textContent = String(s.confirmed);
  el("stat-reviewer").textContent = String(s.confirmed_by_reviewer || 0);
  el("stat-likely").textContent = String(s.likely);
  el("stat-unknown").textContent = String(s.unknown);
  el("stat-absent").textContent = String(s.absent_count);
  updateQueueBar();

  const list = el("results-list");
  list.innerHTML = "";
  if (!state.session) {
    list.innerHTML = '<div class="empty-list-msg">Upload a group photo to see detected faces and attendance status.</div>';
    return;
  }

  if (state.activeFilter === "absent") {
    const absent = state.session.absent_identities || [];
    const notEnrolled = state.session.not_enrolled_identities || [];
    if (!absent.length && !notEnrolled.length) {
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
    for (const person of notEnrolled) {
      const row = document.createElement("div");
      row.className = "face-card";
      row.innerHTML = `
        <div class="face-card-main">
          <span class="face-card-name">${escapeHtml(person.display_name)}</span>
          <span class="face-card-sub">${escapeHtml(person.original_filename)} &bull; Pick a face in Enrollment to include</span>
        </div>
        <span class="status-badge badge-needs_attention">Not enrolled</span>
      `;
      list.appendChild(row);
    }
    return;
  }

  const filtered = state.session.faces.filter(
    (f) => !f.removed && (state.activeFilter === "all" || f.status === state.activeFilter)
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
    const origSub = face.manually_corrected
      ? ` • Orig: ${face.original_display_name}`
      : "";
    btn.innerHTML = `
      <img class="face-thumb" src="${face.crop_b64}" alt="Face #${face.face_index + 1}" />
      <div class="face-card-main">
        <span class="face-card-name">#${face.face_index + 1} ${escapeHtml(face.display_name)}</span>
        <span class="face-card-sub">${simTxt} &bull; ${face.quality.width_px}×${face.quality.height_px}px${escapeHtml(origSub)}</span>
      </div>
      <span class="status-badge badge-${statusCssClass(face.status)}">${badgeLabel}</span>
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
  const face = state.session.faces.find((f) => f.face_index === state.selectedFaceIndex && !f.removed);
  if (!face) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  const srcSuffix =
    face.source === "second_look"
      ? " • Second Look"
      : face.source === "manual_box"
        ? " • Manual Box"
        : "";
  el("explain-face-num").textContent = `Face #${face.face_index + 1}${srcSuffix}`;
  el("explain-title").textContent = face.display_name;

  const badge = el("explain-badge");
  badge.className = `status-badge badge-${statusCssClass(face.status)}`;
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

  const origEl = el("explain-original");
  if (face.manually_corrected) {
    const origSim = face.original_similarity !== null ? ` (sim ${face.original_similarity.toFixed(4)})` : "";
    origEl.textContent = `Original automated result: ${face.original_status} — ${face.original_display_name}${origSim}`;
    origEl.classList.remove("hidden");
  } else {
    origEl.classList.add("hidden");
  }

  el("explain-runnerup").textContent = face.runner_up_name
    ? `Runner-up candidate: ${face.runner_up_name} (${face.runner_up_similarity?.toFixed(4) ?? "—"})`
    : "No runner-up candidate above floor.";
  el("explain-reason").textContent = face.decision_reason;

  el("btn-confirm-match").classList.toggle("hidden", face.status !== "Likely");
  el("btn-remove-manual-box").classList.toggle(
    "hidden",
    face.source !== "manual_box" || Boolean(face.removed)
  );

  // Render append-only decision trail for selected face (fallback to full session if none)
  const allTrail = state.session.decision_trail || [];
  const faceTrail = allTrail.filter((t) => t.face_index === face.face_index);
  el("trail-count").textContent = String(faceTrail.length);
  const trailList = el("trail-list");
  trailList.innerHTML = "";
  for (const entry of faceTrail) {
    const row = document.createElement("div");
    row.className = "trail-item";
    const noteTxt = entry.reason_note ? ` • Note: ${entry.reason_note}` : "";
    row.textContent = `[${entry.actor}] ${entry.action}: ${entry.previous_status ? entry.previous_status + " (" + entry.previous_display_name + ") → " : ""}${entry.new_status} (${entry.new_display_name})${noteTxt}`;
    trailList.appendChild(row);
  }
}

async function submitCorrection(payload, advanceQueue = false) {
  if (!state.session || state.selectedFaceIndex === null) return;
  const reasonNote = el("correction-reason").value.trim() || null;
  const res = await fetch(`/api/sessions/${state.session.session_id}/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      face_index: state.selectedFaceIndex,
      reason_note: reasonNote,
      ...payload,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    showToast(data.detail || "Could not update identification.");
    return;
  }
  el("correction-reason").value = "";
  state.session = { ...state.session, ...data };
  await refreshIdentities();
  updateSummaryAndList();
  renderExplainabilityPanel();
  drawCanvas();
  showToast("Decision recorded in trail.");
  if (advanceQueue) {
    advanceQueueAfterDecision();
  }
}

async function undoLastCorrection() {
  if (!state.session) return;
  const res = await fetch(`/api/sessions/${state.session.session_id}/undo`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    showToast(data.detail || "Nothing to undo.");
    return;
  }
  state.session = { ...state.session, ...data };
  updateSummaryAndList();
  renderExplainabilityPanel();
  drawCanvas();
  showToast("Undid last reviewer action (recorded in trail).");
}

async function runSecondLook() {
  if (!state.session) return;
  setLoading(true, "Running Second Look tiled re-scan...");
  try {
    const res = await fetch(`/api/sessions/${state.session.session_id}/second-look`, {
      method: "POST",
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Second Look failed.");
      return;
    }
    state.session = { ...state.session, ...data };
    updateSummaryAndList();
    renderExplainabilityPanel();
    drawCanvas();
    const sl = data.second_look || {};
    showToast(
      `Second Look (${sl.tile_grid}, 20% overlap): added ${sl.new_faces_found ?? 0} face(s) in ${sl.elapsed_ms ?? 0} ms.`
    );
  } finally {
    setLoading(false);
  }
}

async function submitManualBox(box) {
  if (!state.session) return;
  setLoading(true, "Checking drawn region with YuNet + SFace...");
  try {
    const res = await fetch(`/api/sessions/${state.session.session_id}/manual-box`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(box),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Could not scan region.");
      return;
    }
    state.session = { ...state.session, ...data };
    if (data.manual_box?.found && data.manual_box.added_face_index !== null) {
      state.selectedFaceIndex = data.manual_box.added_face_index;
    }
    updateSummaryAndList();
    renderExplainabilityPanel();
    drawCanvas();
    showToast(data.manual_box?.message || "Region scanned.");
  } finally {
    setLoading(false);
  }
}

async function removeManualBox() {
  if (!state.session || state.selectedFaceIndex === null) return;
  const res = await fetch(
    `/api/sessions/${state.session.session_id}/faces/${state.selectedFaceIndex}/remove`,
    { method: "POST" }
  );
  const data = await res.json();
  if (!res.ok) {
    showToast(data.detail || "Could not remove manual box.");
    return;
  }
  state.session = { ...state.session, ...data };
  const firstActive = state.session.faces.find((f) => !f.removed);
  state.selectedFaceIndex = firstActive ? firstActive.face_index : null;
  updateSummaryAndList();
  renderExplainabilityPanel();
  drawCanvas();
  showToast("Manual box marked removed (kept in audit trail).");
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
    "Confirmed by reviewer": "#0d9488",
    Likely: "#f59e0b",
    unknown: "#ef4444",
  };

  for (const face of state.session.faces) {
    if (face.removed) continue;
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

    const prefix =
      face.status === "Confirmed"
        ? "✓"
        : face.status === "Confirmed by reviewer"
          ? "✓R"
          : face.status === "Likely"
            ? "?"
            : "!";
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

  if (state.drawRectImg) {
    const [rx, ry, rw, rh] = state.drawRectImg;
    ctx.save();
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(
      m.offsetX + rx * m.scale,
      m.offsetY + ry * m.scale,
      rw * m.scale,
      rh * m.scale
    );
    ctx.restore();
  }

  drawCoverageMap(m);
}

function getCoverageMapMetrics() {
  const mapCanvas = el("coverage-map-canvas");
  const img = state.loadedImage;
  if (!mapCanvas || !img) return null;
  const mw = mapCanvas.width;
  const mh = mapCanvas.height;
  const mapScale = Math.min(mw / img.width, mh / img.height);
  const drawW = img.width * mapScale;
  const drawH = img.height * mapScale;
  const offX = (mw - drawW) / 2;
  const offY = (mh - drawH) / 2;
  return { mapCanvas, img, mapScale, drawW, drawH, offX, offY };
}

function drawCoverageMarkerShape(ctx, status, cx, cy, r) {
  ctx.beginPath();
  if (status === "Confirmed") {
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
  } else if (status === "Confirmed by reviewer") {
    ctx.moveTo(cx, cy - r * 1.2);
    ctx.lineTo(cx + r * 1.2, cy);
    ctx.lineTo(cx, cy + r * 1.2);
    ctx.lineTo(cx - r * 1.2, cy);
    ctx.closePath();
  } else if (status === "Likely") {
    ctx.moveTo(cx, cy - r * 1.2);
    ctx.lineTo(cx + r * 1.1, cy + r);
    ctx.lineTo(cx - r * 1.1, cy + r);
    ctx.closePath();
  } else {
    ctx.rect(cx - r, cy - r, r * 2, r * 2);
  }
  ctx.fill();
}

function drawCoverageMap(mainMetrics) {
  const card = el("coverage-map-card");
  const mapBtn = el("btn-toggle-map");
  if (!state.session || !state.loadedImage) {
    card.classList.add("hidden");
    mapBtn.classList.remove("btn-queue-active");
    return;
  }
  mapBtn.classList.toggle("btn-queue-active", state.coverageMapOpen);
  card.classList.toggle("hidden", !state.coverageMapOpen);
  if (!state.coverageMapOpen) return;

  const mm = getCoverageMapMetrics();
  if (!mm) return;
  const ctx = mm.mapCanvas.getContext("2d");
  ctx.clearRect(0, 0, mm.mapCanvas.width, mm.mapCanvas.height);
  ctx.drawImage(mm.img, mm.offX, mm.offY, mm.drawW, mm.drawH);

  // Viewport rectangle inside preserved-aspect photo bounds
  if (mainMetrics) {
    const vx0 = Math.max(0, (0 - mainMetrics.offsetX) / mainMetrics.scale);
    const vy0 = Math.max(0, (0 - mainMetrics.offsetY) / mainMetrics.scale);
    const vx1 = Math.min(mm.img.width, (mainMetrics.canvas.width - mainMetrics.offsetX) / mainMetrics.scale);
    const vy1 = Math.min(mm.img.height, (mainMetrics.canvas.height - mainMetrics.offsetY) / mainMetrics.scale);
    if (vx1 > vx0 && vy1 > vy0) {
      ctx.save();
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 1.5;
      ctx.fillStyle = "rgba(37, 99, 235, 0.10)";
      const rx = mm.offX + vx0 * mm.mapScale;
      const ry = mm.offY + vy0 * mm.mapScale;
      const rw = (vx1 - vx0) * mm.mapScale;
      const rh = (vy1 - vy0) * mm.mapScale;
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.restore();
    }
  }

  const palette = {
    Confirmed: "#10b981",
    "Confirmed by reviewer": "#0d9488",
    Likely: "#f59e0b",
    unknown: "#ef4444",
  };

  for (const face of state.session.faces) {
    if (face.removed) continue;
    const [bx, by, bw, bh] = face.bbox;
    const cx = mm.offX + (bx + bw / 2) * mm.mapScale;
    const cy = mm.offY + (by + bh / 2) * mm.mapScale;
    const isSelected = face.face_index === state.selectedFaceIndex;

    ctx.save();
    ctx.fillStyle = palette[face.status] || "#94a3b8";
    drawCoverageMarkerShape(ctx, face.status, cx, cy, isSelected ? 4.5 : 3.5);
    if (isSelected) {
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.font = "700 8px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 2;
    const numTxt = String(face.face_index + 1);
    ctx.strokeText(numTxt, cx + 5, cy + 3);
    ctx.fillText(numTxt, cx + 5, cy + 3);
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

async function openAboutRunDrawer() {
  openModal("about-modal");
  const body = el("about-run-body");
  let cal = state.session?.calibration || null;
  if (!cal) {
    const res = await fetch("/api/calibration");
    if (res.ok) {
      const data = await res.json();
      cal = data.calibration;
    }
  }
  if (!cal) {
    body.innerHTML = '<p class="photo-meta">No calibration data available yet. Enroll a folder first.</p>';
    return;
  }

  const rawTxt = cal.raw_value !== null && cal.raw_value !== undefined ? cal.raw_value.toFixed(4) : "N/A";
  const rawLabel = cal.fallback_used
    ? "Raw 99.5th percentile (not applied)"
    : "Raw 99.5th percentile";
  let html = `
    <div class="about-grid">
      <div class="about-card"><span>Enrolled Identities (N)</span><strong>${cal.enrolled_identities}</strong></div>
      <div class="about-card"><span>Different-Person Pairs</span><strong>${cal.pair_count}</strong></div>
      <div class="about-card"><span>Percentile</span><strong>${escapeHtml(cal.percentile_used)}</strong></div>
      <div class="about-card"><span>${rawLabel}</span><strong>${rawTxt}</strong></div>
      <div class="about-card"><span>Final T_low (Floor ${cal.floor_value.toFixed(3)}, Cap ${cal.cap_value.toFixed(2)})</span><strong>${cal.final_t_low.toFixed(4)}</strong></div>
      <div class="about-card"><span>Final T_high (Confirmed)</span><strong>${cal.final_t_high.toFixed(4)}</strong></div>
      <div class="about-card"><span>Margins (Base / Lookalike)</span><strong>${cal.base_margin.toFixed(3)} / ${cal.lookalike_margin.toFixed(3)}</strong></div>
    </div>
    <p class="explain-detail">${escapeHtml(cal.method_summary)}</p>
  `;

  if (cal.warning) {
    html += `<div class="about-warn">${escapeHtml(cal.warning)}</div>`;
  }

  html += renderFlaggedPairsHtml(cal);

  html += `<div class="about-limits"><strong>Calibration Limits (Honest Disclosure):</strong><ul>`;
  for (const lim of cal.limitations || []) {
    html += `<li>${escapeHtml(lim)}</li>`;
  }
  html += `</ul></div>`;
  body.innerHTML = html;
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
  el("btn-banner-pick-faces").addEventListener("click", () => openModal("enroll-modal"));
  el("btn-banner-continue").addEventListener("click", () => {
    pendingBannerDismissed = true;
    el("pending-attention-banner").classList.add("hidden");
  });
  el("btn-about-run").addEventListener("click", openAboutRunDrawer);
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
      el("multiface-section").classList.add("hidden");
      el("extra-review-section")?.classList.add("hidden");
      el("pending-attention-banner").classList.add("hidden");
      el("demo-note-banner").classList.add("hidden");
      el("warning-dot").classList.add("hidden");
      ["btn-second-look", "btn-draw-box", "btn-zoom-out", "btn-zoom-in", "btn-zoom-reset", "btn-toggle-map", "btn-export-png", "btn-export-csv", "btn-print-sheet"].forEach(
        (id) => { el(id).disabled = true; }
      );
      el("coverage-map-card").classList.add("hidden");
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

  el("btn-confirm-match").addEventListener("click", () => {
    const face = state.session?.faces.find((f) => f.face_index === state.selectedFaceIndex);
    if (!face || face.status !== "Likely") return;
    submitCorrection({ action: "confirm" }, state.reviewQueueActive);
  });
  el("btn-mark-unknown").addEventListener("click", () =>
    submitCorrection({ action: "mark_unknown" }, state.reviewQueueActive)
  );
  el("btn-undo").addEventListener("click", undoLastCorrection);
  el("btn-remove-manual-box").addEventListener("click", removeManualBox);
  el("btn-second-look").addEventListener("click", runSecondLook);
  el("btn-draw-box").addEventListener("click", () => {
    if (!state.session) return;
    state.drawMode = !state.drawMode;
    el("photo-canvas").classList.toggle("draw-mode", state.drawMode);
    el("btn-draw-box").classList.toggle("btn-queue-active", state.drawMode);
    if (state.drawMode) showToast("Drag a box on the photo around a missed face.");
  });

  el("btn-toggle-queue").addEventListener("click", () => {
    if (!state.session) return;
    state.reviewQueueActive = !state.reviewQueueActive;
    if (state.reviewQueueActive) {
      const q = getReviewQueueIndices();
      if (!q.length) {
        state.reviewQueueActive = false;
        showToast("No Likely or unknown faces to review.");
      } else {
        selectFace(q[0]);
      }
    } else {
      updateQueueBar();
    }
  });
  el("btn-queue-prev").addEventListener("click", () => stepFace(-1));
  el("btn-queue-next").addEventListener("click", () => stepFace(1));
  el("btn-queue-skip").addEventListener("click", () => {
    if (state.reviewQueueActive) stepFace(1, true);
    else stepFace(1, false);
  });

  const assignInput = el("assign-search-input");
  assignInput.addEventListener("focus", () => renderAssignFilterResults(assignInput.value));
  assignInput.addEventListener("input", () => renderAssignFilterResults(assignInput.value));
  assignInput.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") {
      el("assign-results-list").classList.add("hidden");
      assignInput.blur();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const firstOpt = el("assign-results-list").querySelector("button.assign-option");
      if (firstOpt) firstOpt.click();
    }
  });

  el("btn-export-png").addEventListener("click", () => {
    if (state.session) window.location.href = `/api/sessions/${state.session.session_id}/export/png`;
  });
  el("btn-export-csv").addEventListener("click", () => {
    if (state.session) window.location.href = `/api/sessions/${state.session.session_id}/export/csv`;
  });
  el("btn-print-sheet").addEventListener("click", () => {
    if (state.session) window.open(`/api/sessions/${state.session.session_id}/export/print`, "_blank", "noopener");
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
  el("btn-toggle-map").addEventListener("click", () => {
    if (!state.session) return;
    state.coverageMapOpen = !state.coverageMapOpen;
    drawCanvas();
  });

  el("coverage-map-canvas").addEventListener("click", (e) => {
    if (!state.session || !state.loadedImage) return;
    const mm = getCoverageMapMetrics();
    if (!mm) return;
    const rect = mm.mapCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // 1. Hit-test active markers (skip removed manual boxes)
    let bestFace = null;
    let bestDist = 11;
    for (const face of state.session.faces) {
      if (face.removed) continue;
      const [bx, by, bw, bh] = face.bbox;
      const fx = mm.offX + (bx + bw / 2) * mm.mapScale;
      const fy = mm.offY + (by + bh / 2) * mm.mapScale;
      const dist = Math.hypot(mx - fx, my - fy);
      if (dist <= bestDist) {
        bestDist = dist;
        bestFace = face;
      }
    }
    if (bestFace) {
      selectFace(bestFace.face_index);
      return;
    }

    // 2. Otherwise pan main canvas to clicked photo location
    const mainM = getCanvasMetrics();
    if (!mainM) return;
    const targetImgX = (mx - mm.offX) / mm.mapScale;
    const targetImgY = (my - mm.offY) / mm.mapScale;
    if (targetImgX < 0 || targetImgX > mm.img.width || targetImgY < 0 || targetImgY > mm.img.height) {
      return;
    }
    const baseDrawW = mm.img.width * mainM.scale;
    const baseDrawH = mm.img.height * mainM.scale;
    state.panX = mainM.canvas.width / 2 - ((mainM.canvas.width - baseDrawW) / 2 + targetImgX * mainM.scale);
    state.panY = mainM.canvas.height / 2 - ((mainM.canvas.height - baseDrawH) / 2 + targetImgY * mainM.scale);
    drawCanvas();
  });

  const canvas = el("photo-canvas");
  canvas.addEventListener("mousedown", (e) => {
    if (state.drawMode && state.session) {
      const m = getCanvasMetrics();
      if (!m) return;
      const rect = canvas.getBoundingClientRect();
      const ix = (e.clientX - rect.left - m.offsetX) / m.scale;
      const iy = (e.clientY - rect.top - m.offsetY) / m.scale;
      state.isDrawing = true;
      state.drawStartImg = [ix, iy];
      state.drawRectImg = [ix, iy, 0, 0];
      return;
    }
    state.isPanning = true;
    state.didDrag = false;
    state.panStartX = e.clientX - state.panX;
    state.panStartY = e.clientY - state.panY;
  });
  window.addEventListener("mousemove", (e) => {
    if (state.isDrawing && state.drawStartImg) {
      const m = getCanvasMetrics();
      if (!m) return;
      const rect = canvas.getBoundingClientRect();
      const ix = (e.clientX - rect.left - m.offsetX) / m.scale;
      const iy = (e.clientY - rect.top - m.offsetY) / m.scale;
      const [sx, sy] = state.drawStartImg;
      state.drawRectImg = [
        Math.min(sx, ix),
        Math.min(sy, iy),
        Math.abs(ix - sx),
        Math.abs(iy - sy),
      ];
      drawCanvas();
      return;
    }
    if (!state.isPanning) return;
    state.didDrag = true;
    state.panX = e.clientX - state.panStartX;
    state.panY = e.clientY - state.panStartY;
    drawCanvas();
  });
  window.addEventListener("mouseup", async (e) => {
    if (state.isDrawing) {
      state.isDrawing = false;
      const r = state.drawRectImg;
      state.drawRectImg = null;
      state.drawStartImg = null;
      state.drawMode = false;
      canvas.classList.remove("draw-mode");
      el("btn-draw-box").classList.remove("btn-queue-active");
      drawCanvas();
      if (r && r[2] >= 10 && r[3] >= 10) {
        await submitManualBox({
          x: Math.max(0, Math.round(r[0])),
          y: Math.max(0, Math.round(r[1])),
          w: Math.round(r[2]),
          h: Math.round(r[3]),
        });
      }
      return;
    }
    if (!state.isPanning) return;
    state.isPanning = false;
    if (!state.didDrag && state.session) {
      const m = getCanvasMetrics();
      if (!m) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      for (const face of state.session.faces) {
        if (face.removed) continue;
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

  // Keyboard shortcuts (ignored while typing in inputs/selects/textareas)
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeModal("enroll-modal");
      closeModal("about-modal");
      closeModal("wipe-modal");
      el("assign-results-list").classList.add("hidden");
      if (state.reviewQueueActive) {
        state.reviewQueueActive = false;
        updateQueueBar();
      }
      return;
    }
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (!state.session || !state.session.faces.length) return;

    const k = e.key.toLowerCase();
    if (k === "r") {
      e.preventDefault();
      el("btn-toggle-queue").click();
    } else if (k === "j") {
      e.preventDefault();
      stepFace(1);
    } else if (k === "k") {
      e.preventDefault();
      stepFace(-1);
    } else if (k === "s") {
      e.preventDefault();
      el("btn-queue-skip").click();
    } else if (k === "c") {
      const face = state.session.faces.find((f) => f.face_index === state.selectedFaceIndex);
      if (face && face.status === "Likely") {
        e.preventDefault();
        submitCorrection({ action: "confirm" }, state.reviewQueueActive);
      }
    } else if (k === "u") {
      e.preventDefault();
      submitCorrection({ action: "mark_unknown" }, state.reviewQueueActive);
    } else if (k === "a") {
      e.preventDefault();
      el("assign-search-input").focus();
      renderAssignFilterResults(el("assign-search-input").value);
    } else if (k === "z") {
      e.preventDefault();
      undoLastCorrection();
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth <= 960 && state.coverageMapOpen) {
      state.coverageMapOpen = false;
    }
    drawCanvas();
  });
}

bindEvents();
fetchHealth();
refreshIdentities();
