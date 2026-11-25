/* PhotoProof Offline Frontend Controller (No external CDNs, strict CSP compliant) */

(() => {
  "use strict";

  const state = {
    session: null,
    identities: [],
    enrollSummary: null,
    selectedFaceIndex: null,
    groupImageEl: null,
    showLabels: true,
    drawMode: false,
    dragStart: null,
    dragCurrent: null,
    statusFilter: "all",
    refTab: "all",
    pollTimer: null,
  };

  const el = {
    statusBadge: document.getElementById("status-badge"),
    calibrationBadge: document.getElementById("calibration-badge"),
    btnUndo: document.getElementById("btn-undo"),
    btnClearSession: document.getElementById("btn-clear-session"),
    toast: document.getElementById("toast-banner"),

    metricRefLoaded: document.getElementById("metric-ref-loaded"),
    metricRefRejected: document.getElementById("metric-ref-rejected"),
    metricFacesDetected: document.getElementById("metric-faces-detected"),
    metricHighConf: document.getElementById("metric-high-conf"),
    metricNeedsReview: document.getElementById("metric-needs-review"),
    metricUnknown: document.getElementById("metric-unknown"),
    manualSummaryNote: document.getElementById("manual-summary-note"),

    refFolderInput: document.getElementById("ref-folder-input"),
    btnTogglePathEnroll: document.getElementById("btn-toggle-path-enroll"),
    pathEnrollBox: document.getElementById("path-enroll-box"),
    refPathInput: document.getElementById("ref-path-input"),
    btnPathEnroll: document.getElementById("btn-path-enroll"),
    chkReplaceRefs: document.getElementById("chk-replace-refs"),

    groupFileInput: document.getElementById("group-file-input"),
    groupDropzone: document.getElementById("group-dropzone"),
    btnDemo: document.getElementById("btn-demo"),

    progressCard: document.getElementById("enroll-progress-card"),
    progressLabel: document.getElementById("enroll-progress-label"),
    progressCount: document.getElementById("enroll-progress-count"),
    progressFill: document.getElementById("enroll-progress-fill"),

    refSection: document.getElementById("reference-section"),
    healthHeadline: document.getElementById("health-headline"),
    countAttentionTab: document.getElementById("count-attention-tab"),
    countRejectedTab: document.getElementById("count-rejected-tab"),
    pendingArea: document.getElementById("pending-reviews-area"),
    rejectedArea: document.getElementById("rejected-refs-area"),
    contactSheetArea: document.getElementById("contact-sheet-area"),
    contactSheetGrid: document.getElementById("contact-sheet-grid"),

    workspace: document.getElementById("workspace-section"),
    groupPhotoTitle: document.getElementById("group-photo-title"),
    groupMetaLine: document.getElementById("group-meta-line"),
    btnToggleLabels: document.getElementById("btn-toggle-labels"),
    btnSecondLook: document.getElementById("btn-second-look"),
    btnManualBox: document.getElementById("btn-manual-box"),
    canvas: document.getElementById("group-canvas"),

    btnExportPng: document.getElementById("btn-export-png"),
    btnExportCsv: document.getElementById("btn-export-csv"),
    btnExportPrint: document.getElementById("btn-export-print"),

    evidenceEmpty: document.getElementById("evidence-empty"),
    evidenceBody: document.getElementById("evidence-body"),
    evFaceLabel: document.getElementById("ev-face-label"),
    evStatusBadge: document.getElementById("ev-status-badge"),
    evSourceBadge: document.getElementById("ev-source-badge"),
    evProposedName: document.getElementById("ev-proposed-name"),
    evGroupCrop: document.getElementById("ev-group-crop"),
    evRefCrop: document.getElementById("ev-ref-crop"),
    evNoRefBox: document.getElementById("ev-no-ref-box"),
    evRefNameSub: document.getElementById("ev-ref-name-sub"),
    evRunnerupName: document.getElementById("ev-runnerup-name"),
    evRunnerupScore: document.getElementById("ev-runnerup-score"),
    evSimilarity: document.getElementById("ev-similarity"),
    evMargin: document.getElementById("ev-margin"),
    evSizeBlur: document.getElementById("ev-size-blur"),
    evWarnings: document.getElementById("ev-warnings"),
    evReasonText: document.getElementById("ev-reason-text"),
    btnEvConfirm: document.getElementById("btn-ev-confirm"),
    btnEvUnknown: document.getElementById("btn-ev-unknown"),
    btnEvUseRunnerup: document.getElementById("btn-ev-use-runnerup"),
    evPersonSelect: document.getElementById("ev-person-select"),
    btnEvAssign: document.getElementById("btn-ev-assign"),
    evNoteInput: document.getElementById("ev-note-input"),
    evRemoveBox: document.getElementById("ev-remove-box"),
    btnEvRemoveManual: document.getElementById("btn-ev-remove-manual"),

    resultsList: document.getElementById("results-list"),
  };

  function showToast(message, isError = false) {
    if (!message) {
      el.toast.classList.add("hidden");
      el.toast.textContent = "";
      return;
    }
    el.toast.textContent = message;
    el.toast.classList.toggle("toast-error", isError);
    el.toast.classList.remove("hidden");
  }

  async function apiJson(url, options = {}) {
    const resp = await fetch(url, options);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `Request failed (${resp.status})`);
    }
    return data;
  }

  function statusClassName(status) {
    if (status === "High-confidence match") return "status-confirmed";
    if (status === "Needs review") return "status-review";
    return "status-unknown";
  }

  function statusStrokeColor(status) {
    if (status === "High-confidence match") return "#2f6f4e";
    if (status === "Needs review") return "#b85c09";
    return "#525866";
  }

  function issueCategoryLabel(category) {
    switch (category) {
      case "no_face":
        return "No face";
      case "multiple_faces":
        return "Multiple faces";
      case "face_too_small":
        return "Face too small";
      case "blurry_image":
        return "Blurry image";
      case "duplicate_name":
        return "Duplicate name";
      case "unclear_filename":
        return "Unclear filename";
      case "unreadable_image":
        return "Unreadable image";
      default:
        return "Quality check";
    }
  }

  function updateReadinessMetrics() {
    const enrollSummary = state.enrollSummary;
    const session = state.session;
    const summary = session ? session.summary || {} : {};
    const enrollCounts = enrollSummary ? enrollSummary.counts || {} : {};

    const loaded =
      summary.references_loaded !== undefined
        ? summary.references_loaded
        : enrollSummary
        ? enrollSummary.enrolled_photos || 0
        : state.identities.length;

    const rejected =
      summary.references_rejected !== undefined
        ? summary.references_rejected
        : enrollCounts.references_rejected || 0;

    el.metricRefLoaded.textContent = String(loaded);
    el.metricRefRejected.textContent = String(rejected);
    el.metricFacesDetected.textContent = String(summary.faces_detected || 0);
    el.metricHighConf.textContent = String(summary.high_confidence_matches || 0);
    el.metricNeedsReview.textContent = String(summary.needs_review_faces || 0);
    el.metricUnknown.textContent = String(summary.unknown_faces || 0);

    const manualCount = summary.manual_decisions || 0;
    if (session) {
      el.manualSummaryNote.textContent =
        manualCount > 0
          ? `${manualCount} manual decision${manualCount === 1 ? "" : "s"} in this session (export to save)`
          : "Click any face to inspect its Evidence Card";
    } else {
      el.manualSummaryNote.textContent = "All computation runs locally on this laptop";
    }
  }

  function renderPersonDropdown() {
    const prevVal = el.evPersonSelect.value;
    el.evPersonSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent =
      state.identities.length > 0
        ? "Select person from reference folder…"
        : "Load a reference folder first";
    el.evPersonSelect.appendChild(placeholder);

    for (const person of state.identities) {
      const opt = document.createElement("option");
      opt.value = person.identity_id;
      opt.textContent = person.display_name;
      el.evPersonSelect.appendChild(opt);
    }
    if (prevVal && state.identities.some((i) => i.identity_id === prevVal)) {
      el.evPersonSelect.value = prevVal;
    }
  }

  function renderContactSheet() {
    el.contactSheetGrid.innerHTML = "";
    if (!state.identities || state.identities.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted-text";
      empty.textContent = "No usable reference portraits loaded yet.";
      el.contactSheetGrid.appendChild(empty);
      return;
    }

    for (const person of state.identities) {
      const card = document.createElement("div");
      card.className = "contact-card";

      const img = document.createElement("img");
      img.alt = person.display_name;
      if (person.primary_photo_id) {
        img.src = `/api/reference-crop/${person.primary_photo_id}`;
      }
      card.appendChild(img);

      const nameEl = document.createElement("div");
      nameEl.className = "contact-card-name";
      nameEl.textContent = person.display_name;
      card.appendChild(nameEl);

      const metaEl = document.createElement("div");
      metaEl.className = "contact-card-meta";
      const n = Number(person.photo_count || 1);
      metaEl.textContent = n > 1 ? `${n} photos merged` : "1 reference photo";
      card.appendChild(metaEl);

      el.contactSheetGrid.appendChild(card);
    }
  }

  function renderPendingAndRejected() {
    const s = state.enrollSummary;
    el.pendingArea.innerHTML = "";
    el.rejectedArea.innerHTML = "";

    const multiface = (s && s.multiface_pending) || [];
    const extras = (s && s.extra_photo_pending) || [];
    const rejectedRecords = ((s && s.file_records) || []).filter(
      (r) => r.status === "failed" || r.status === "rejected"
    );

    el.countAttentionTab.textContent = String(multiface.length + extras.length);
    el.countRejectedTab.textContent = String(rejectedRecords.length);

    // Multi-face items (never auto-selected silently)
    for (const item of multiface) {
      const card = document.createElement("div");
      card.className = "issue-card";

      const head = document.createElement("div");
      head.className = "issue-head";
      const title = document.createElement("strong");
      title.textContent = `${item.filename} — Multiple faces detected (${item.face_count})`;
      const tag = document.createElement("span");
      tag.className = "issue-category-tag";
      tag.textContent = "Multiple faces";
      head.appendChild(title);
      head.appendChild(tag);
      card.appendChild(head);

      const desc = document.createElement("p");
      desc.className = "muted-text";
      desc.textContent = `Choose which face belongs to ${item.display_name}, or skip this photo.`;
      card.appendChild(desc);

      const choices = document.createElement("div");
      choices.className = "multiface-choices";

      for (const cand of item.candidates) {
        const box = document.createElement("div");
        box.className = "candidate-pick";
        const img = document.createElement("img");
        img.src = cand.crop_b64;
        img.alt = `Candidate face ${cand.candidate_index + 1}`;
        box.appendChild(img);

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className =
          cand.candidate_index === item.suggested_index
            ? "btn btn-primary"
            : "btn btn-secondary";
        btn.textContent = `Choose Face ${cand.candidate_index + 1}`;
        btn.addEventListener("click", () => resolveMultiFace(item.token, cand.candidate_index));
        box.appendChild(btn);
        choices.appendChild(box);
      }

      const skipBtn = document.createElement("button");
      skipBtn.type = "button";
      skipBtn.className = "btn btn-quiet";
      skipBtn.textContent = "Skip this image";
      skipBtn.addEventListener("click", () => resolveMultiFace(item.token, null));
      choices.appendChild(skipBtn);

      card.appendChild(choices);
      el.pendingArea.appendChild(card);
    }

    // Duplicate-name extra photo mismatches
    for (const item of extras) {
      const card = document.createElement("div");
      card.className = "issue-card";

      const head = document.createElement("div");
      head.className = "issue-head";
      const title = document.createElement("strong");
      title.textContent = `${item.filename} — Duplicate name "${item.display_name}" may be a different person`;
      const tag = document.createElement("span");
      tag.className = "issue-category-tag";
      tag.textContent = "Duplicate name";
      head.appendChild(title);
      head.appendChild(tag);
      card.appendChild(head);

      const reason = document.createElement("p");
      reason.className = "muted-text";
      reason.textContent = item.reason;
      card.appendChild(reason);

      const compare = document.createElement("div");
      compare.className = "extra-compare-row";

      const prevWrap = document.createElement("div");
      prevWrap.className = "extra-thumb-wrap";
      const prevImg = document.createElement("img");
      prevImg.src = item.existing_thumb_b64 || "";
      prevImg.alt = item.existing_filename;
      const prevCap = document.createElement("span");
      prevCap.textContent = `Loaded: ${item.existing_filename}`;
      prevWrap.appendChild(prevImg);
      prevWrap.appendChild(prevCap);

      const newWrap = document.createElement("div");
      newWrap.className = "extra-thumb-wrap";
      const newImg = document.createElement("img");
      newImg.src = item.new_thumb_b64 || "";
      newImg.alt = item.filename;
      const newCap = document.createElement("span");
      newCap.textContent = `New: ${item.filename}`;
      newWrap.appendChild(newImg);
      newWrap.appendChild(newCap);

      const actions = document.createElement("div");
      actions.className = "action-buttons";
      const keepBtn = document.createElement("button");
      keepBtn.type = "button";
      keepBtn.className = "btn btn-primary";
      keepBtn.textContent = "Same person — Keep";
      keepBtn.addEventListener("click", () => resolveExtraPhoto(item.token, "keep"));

      const rejectBtn = document.createElement("button");
      rejectBtn.type = "button";
      rejectBtn.className = "btn btn-outline-danger";
      rejectBtn.textContent = "Different person — Reject";
      rejectBtn.addEventListener("click", () => resolveExtraPhoto(item.token, "reject"));

      actions.appendChild(keepBtn);
      actions.appendChild(rejectBtn);

      compare.appendChild(prevWrap);
      compare.appendChild(newWrap);
      compare.appendChild(actions);
      card.appendChild(compare);
      el.pendingArea.appendChild(card);
    }

    // Rejected / unusable reference images
    for (const rec of rejectedRecords) {
      const card = document.createElement("div");
      card.className = "issue-card issue-rejected";

      const head = document.createElement("div");
      head.className = "issue-head";
      const title = document.createElement("strong");
      title.textContent = rec.filename;
      const tag = document.createElement("span");
      tag.className = "issue-category-tag";
      tag.textContent = issueCategoryLabel(rec.issue_category);
      head.appendChild(title);
      head.appendChild(tag);
      card.appendChild(head);

      const row = document.createElement("div");
      row.className = "extra-compare-row";
      if (rec.thumb_b64) {
        const img = document.createElement("img");
        img.src = rec.thumb_b64;
        img.alt = rec.filename;
        img.width = 56;
        img.height = 56;
        row.appendChild(img);
      }
      const msg = document.createElement("span");
      msg.className = "muted-text";
      msg.textContent = rec.reason;
      row.appendChild(msg);
      card.appendChild(row);
      el.rejectedArea.appendChild(card);
    }

    const hasPending = multiface.length + extras.length > 0;
    const hasRejected = rejectedRecords.length > 0;

    if (state.refTab === "all") {
      el.pendingArea.classList.toggle("hidden", !hasPending);
      el.rejectedArea.classList.toggle("hidden", !hasRejected);
      el.contactSheetArea.classList.remove("hidden");
    } else if (state.refTab === "attention") {
      el.pendingArea.classList.toggle("hidden", !hasPending);
      el.rejectedArea.classList.add("hidden");
      el.contactSheetArea.classList.add("hidden");
      if (!hasPending) {
        const note = document.createElement("p");
        note.className = "muted-text";
        note.textContent = "No pending multi-face or duplicate-name reviews.";
        el.pendingArea.appendChild(note);
        el.pendingArea.classList.remove("hidden");
      }
    } else {
      el.pendingArea.classList.add("hidden");
      el.rejectedArea.classList.toggle("hidden", !hasRejected);
      el.contactSheetArea.classList.add("hidden");
      if (!hasRejected) {
        const note = document.createElement("p");
        note.className = "muted-text";
        note.textContent = "No unusable reference images were rejected.";
        el.rejectedArea.appendChild(note);
        el.rejectedArea.classList.remove("hidden");
      }
    }
  }

  function renderReferenceSection() {
    const s = state.enrollSummary;
    const hasData = Boolean(s) || state.identities.length > 0;
    el.refSection.classList.toggle("hidden", !hasData);
    if (!hasData) return;

    if (s && s.counts && s.counts.headline) {
      const dupInfos = (s.infos || []).filter((i) =>
        (i.reason || "").toLowerCase().includes("duplicate")
      ).length;
      el.healthHeadline.textContent =
        dupInfos > 0
          ? `${s.counts.headline} • ${dupInfos} duplicate name${dupInfos === 1 ? "" : "s"} verified & merged`
          : s.counts.headline;
    } else {
      el.healthHeadline.textContent = `${state.identities.length} people loaded in visual contact sheet.`;
    }

    renderContactSheet();
    renderPendingAndRejected();
  }

  function rectsOverlap(a, b) {
    return a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;
  }

  function chooseCanvasLabelRect(bx, by, bw, bh, boxW, boxH, imgW, imgH, placed) {
    const xCands = [bx, bx + bw - boxW, Math.max(0, bx + Math.floor(bw / 2) - Math.floor(boxW / 2)), 0];
    const yCands = [by - boxH - 4, by + bh + 4, by - 2 * (boxH + 6), by + bh + boxH + 10, by + 4];
    const step = boxH + 6;
    for (let k = 2; k < 6; k++) {
      yCands.push(by - (k + 1) * step);
      yCands.push(by + bh + k * step);
    }
    let fallback = null;
    for (const cy of yCands) {
      for (const cx of xCands) {
        const rx1 = Math.max(0, Math.min(imgW - boxW, cx));
        const ry1 = Math.max(0, Math.min(imgH - boxH, cy));
        const cand = { x1: rx1, y1: ry1, x2: Math.min(imgW, rx1 + boxW), y2: Math.min(imgH, ry1 + boxH) };
        if (!fallback) fallback = cand;
        if (!placed.some((p) => rectsOverlap(cand, p))) {
          return cand;
        }
      }
    }
    return fallback || { x1: 0, y1: 0, x2: boxW, y2: boxH };
  }

  function drawCanvas() {
    const session = state.session;
    const img = state.groupImageEl;
    if (!session || !img) return;

    const canvas = el.canvas;
    const ctx = canvas.getContext("2d");
    canvas.width = session.width;
    canvas.height = session.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const activeFaces = (session.faces || []).filter((f) => !f.removed);
    const lineThick = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 380));
    const fontSize = Math.max(13, Math.min(20, Math.round(Math.min(canvas.width, canvas.height) / 55)));
    ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;

    // Draw bounding boxes first
    for (const face of activeFaces) {
      const [bx, by, bw, bh] = face.bbox;
      const isSelected = face.face_index === state.selectedFaceIndex;
      const color = statusStrokeColor(face.status);
      ctx.strokeStyle = isSelected ? "#1d4ed8" : color;
      ctx.lineWidth = isSelected ? lineThick + 2 : lineThick;
      ctx.strokeRect(bx, by, bw, bh);
    }

    // Draw non-overlapping labels cleanly
    if (state.showLabels) {
      const ordered = [...activeFaces].sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]);
      const placed = [];
      const padX = 7;
      const boxH = fontSize + 10;

      for (const face of ordered) {
        const [bx, by, bw, bh] = face.bbox;
        const idxNum = face.face_index + 1;
        const isManual = face.decision_source === "manual";
        const suffix = isManual ? " [Manual]" : "";
        let label;
        if (face.status === "High-confidence match") {
          label = `Face ${idxNum}: ${face.display_name}${suffix}`;
        } else if (face.status === "Needs review") {
          label = `Face ${idxNum}: ${face.display_name}?${suffix}`;
        } else {
          label = `Face ${idxNum}: Unknown${suffix}`;
        }

        const tw = Math.ceil(ctx.measureText(label).width);
        const boxW = Math.min(canvas.width, tw + padX * 2);
        const rect = chooseCanvasLabelRect(bx, by, bw, bh, boxW, boxH, canvas.width, canvas.height, placed);
        placed.push(rect);

        const bgColor = face.face_index === state.selectedFaceIndex ? "#1d4ed8" : statusStrokeColor(face.status);
        ctx.fillStyle = bgColor;
        ctx.fillRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
        if (isManual) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(rect.x1 + 0.5, rect.y1 + 0.5, rect.x2 - rect.x1 - 1, rect.y2 - rect.y1 - 1);
        }
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, rect.x1 + padX, rect.y2 - 6);
      }
    }

    // Active drag box for manual face drawing
    if (state.drawMode && state.dragStart && state.dragCurrent) {
      const x = Math.min(state.dragStart.x, state.dragCurrent.x);
      const y = Math.min(state.dragStart.y, state.dragCurrent.y);
      const w = Math.abs(state.dragCurrent.x - state.dragStart.x);
      const h = Math.abs(state.dragCurrent.y - state.dragStart.y);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "#1d4ed8";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }
  }

  function renderEvidenceCard() {
    const session = state.session;
    const activeFaces = session ? (session.faces || []).filter((f) => !f.removed) : [];
    const face = activeFaces.find((f) => f.face_index === state.selectedFaceIndex) || null;

    if (!face) {
      el.evidenceEmpty.classList.remove("hidden");
      el.evidenceBody.classList.add("hidden");
      return;
    }

    el.evidenceEmpty.classList.add("hidden");
    el.evidenceBody.classList.remove("hidden");

    const idx = face.face_index + 1;
    el.evFaceLabel.textContent = `Face ${idx}`;
    el.evStatusBadge.textContent = face.status;
    el.evStatusBadge.className = `status-pill ${statusClassName(face.status)}`;

    const isManual = face.decision_source === "manual";
    el.evSourceBadge.textContent = isManual ? "Manual decision" : "Model suggestion";
    el.evSourceBadge.className = isManual ? "source-pill source-manual" : "source-pill";

    el.evProposedName.textContent =
      face.status === "Unknown" ? "Unknown person" : face.display_name;
    el.evGroupCrop.src = face.crop_b64 || "";

    if (face.matched_photo_id) {
      el.evRefCrop.src = `/api/reference-crop/${face.matched_photo_id}`;
      el.evRefCrop.classList.remove("hidden");
      el.evNoRefBox.classList.add("hidden");
      el.evRefNameSub.textContent = face.display_name;
    } else {
      el.evRefCrop.classList.add("hidden");
      el.evNoRefBox.classList.remove("hidden");
      el.evRefNameSub.textContent = "";
    }

    if (face.runner_up_name) {
      el.evRunnerupName.textContent = face.runner_up_name;
      el.evRunnerupScore.textContent =
        face.runner_up_similarity !== null && face.runner_up_similarity !== undefined
          ? `sim ${Number(face.runner_up_similarity).toFixed(3)}`
          : "";
      const runnerIdentity = state.identities.find(
        (i) => i.display_name === face.runner_up_name
      );
      if (runnerIdentity) {
        el.btnEvUseRunnerup.textContent = `Use 2nd: ${runnerIdentity.display_name}`;
        el.btnEvUseRunnerup.dataset.identityId = runnerIdentity.identity_id;
        el.btnEvUseRunnerup.classList.remove("hidden");
      } else {
        el.btnEvUseRunnerup.classList.add("hidden");
      }
    } else {
      el.evRunnerupName.textContent = "None";
      el.evRunnerupScore.textContent = "";
      el.btnEvUseRunnerup.classList.add("hidden");
    }

    el.evSimilarity.textContent =
      face.similarity !== null && face.similarity !== undefined
        ? Number(face.similarity).toFixed(4)
        : "N/A";
    el.evMargin.textContent =
      face.runner_up_margin !== null && face.runner_up_margin !== undefined
        ? `+${Number(face.runner_up_margin).toFixed(4)}`
        : "N/A";

    const q = face.quality || {};
    const blurTxt = q.blur_laplacian !== undefined ? Number(q.blur_laplacian).toFixed(1) : "—";
    el.evSizeBlur.textContent = `${q.width_px || 0}×${q.height_px || 0} px • blur ${blurTxt}`;

    el.evWarnings.innerHTML = "";
    const reasons = q.reasons || [];
    if (reasons.length === 0) {
      const okChip = document.createElement("span");
      okChip.className = "warning-chip ok-chip";
      okChip.textContent = "Clear face quality";
      el.evWarnings.appendChild(okChip);
    } else {
      for (const r of reasons) {
        const chip = document.createElement("span");
        chip.className = "warning-chip";
        chip.textContent = r;
        el.evWarnings.appendChild(chip);
      }
    }

    el.evReasonText.textContent = face.decision_reason || "";
    el.btnEvConfirm.disabled = !face.identity_id;
    el.evRemoveBox.classList.toggle("hidden", face.source !== "manual_box");
    renderPersonDropdown();
  }

  function renderResultsList() {
    el.resultsList.innerHTML = "";
    const session = state.session;
    if (!session) return;

    let activeFaces = (session.faces || []).filter((f) => !f.removed);
    if (state.statusFilter !== "all") {
      activeFaces = activeFaces.filter((f) => f.status === state.statusFilter);
    }

    if (activeFaces.length === 0) {
      const empty = document.createElement("li");
      empty.className = "muted-text";
      empty.textContent = "No faces match this filter.";
      el.resultsList.appendChild(empty);
      return;
    }

    for (const face of activeFaces) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "result-item";
      btn.setAttribute("role", "option");
      const isSelected = face.face_index === state.selectedFaceIndex;
      btn.setAttribute("aria-selected", String(isSelected));
      if (isSelected) btn.classList.add("selected");

      const thumb = document.createElement("img");
      thumb.className = "result-thumb";
      thumb.src = face.crop_b64 || "";
      thumb.alt = `Face ${face.face_index + 1}`;
      btn.appendChild(thumb);

      const main = document.createElement("div");
      main.className = "result-main";

      const top = document.createElement("div");
      top.className = "result-top";

      const fNum = document.createElement("span");
      fNum.className = "result-face-num";
      fNum.textContent = `Face ${face.face_index + 1}`;
      top.appendChild(fNum);

      const fName = document.createElement("span");
      fName.className = "result-name";
      fName.textContent = face.display_name;
      top.appendChild(fName);

      const stPill = document.createElement("span");
      stPill.className = `status-pill ${statusClassName(face.status)}`;
      stPill.textContent = face.status;
      top.appendChild(stPill);

      const srcPill = document.createElement("span");
      const isManual = face.decision_source === "manual";
      srcPill.className = isManual ? "source-pill source-manual" : "source-pill";
      srcPill.textContent = isManual ? "Manual decision" : "Model suggestion";
      top.appendChild(srcPill);

      main.appendChild(top);

      const reason = document.createElement("div");
      reason.className = "result-reason";
      reason.textContent = face.decision_reason || "";
      main.appendChild(reason);

      btn.appendChild(main);
      btn.addEventListener("click", () => selectFace(face.face_index));
      li.appendChild(btn);
      el.resultsList.appendChild(li);
    }
  }

  function selectFace(faceIndex) {
    state.selectedFaceIndex = faceIndex;
    drawCanvas();
    renderEvidenceCard();
    renderResultsList();
  }

  function applySessionPayload(payload, keepSelection = true) {
    const prevImage = state.session ? state.session.image_b64 : null;
    if (!payload.image_b64 && prevImage) {
      payload.image_b64 = prevImage;
    }
    state.session = payload;
    if (Array.isArray(payload.identities)) {
      state.identities = payload.identities;
    }

    el.workspace.classList.remove("hidden");
    el.groupPhotoTitle.textContent = `Group Photo — ${payload.filename}`;
    const timeTxt =
      payload.processing_time_ms !== undefined
        ? `${payload.processing_time_ms} ms • `
        : "";
    el.groupMetaLine.textContent = `${timeTxt}${payload.width}×${payload.height} px • Click any face box or row to inspect its Evidence Card`;

    el.btnUndo.classList.toggle("hidden", !payload.can_undo);

    if (payload.calibration && payload.calibration.final_t_high) {
      el.calibrationBadge.textContent = `Thresholds: review ≥ ${Number(payload.calibration.final_t_low).toFixed(2)}, high ≥ ${Number(payload.calibration.final_t_high).toFixed(2)}`;
      el.calibrationBadge.classList.remove("hidden");
    }

    const activeFaces = (payload.faces || []).filter((f) => !f.removed);
    if (!keepSelection || !activeFaces.some((f) => f.face_index === state.selectedFaceIndex)) {
      const firstReview = activeFaces.find((f) => f.status === "Needs review");
      state.selectedFaceIndex = firstReview
        ? firstReview.face_index
        : activeFaces.length > 0
        ? activeFaces[0].face_index
        : null;
    }

    updateReadinessMetrics();
    renderContactSheet();
    renderPersonDropdown();

    const finishRender = () => {
      drawCanvas();
      renderEvidenceCard();
      renderResultsList();
    };

    if (!state.groupImageEl || (payload.image_b64 && state.groupImageEl.src !== payload.image_b64)) {
      const img = new Image();
      img.onload = () => {
        state.groupImageEl = img;
        finishRender();
      };
      img.src = payload.image_b64;
    } else {
      finishRender();
    }
  }

  async function refreshHealth() {
    try {
      const health = await apiJson("/api/health");
      if (health.model_ready) {
        el.statusBadge.textContent = "Offline local engine ready";
        el.statusBadge.className = "status-pill status-confirmed";
      } else {
        el.statusBadge.textContent = "Model missing";
        el.statusBadge.className = "status-pill status-review";
        showToast(health.model_error || "Face models missing in models/.", true);
      }
      const idResp = await apiJson("/api/identities");
      state.identities = idResp.identities || [];
      renderPersonDropdown();
      renderReferenceSection();
      updateReadinessMetrics();
    } catch (err) {
      el.statusBadge.textContent = "Server error";
      el.statusBadge.className = "status-pill status-unknown";
      showToast(err.message, true);
    }
  }

  function pollEnrollJob(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);
    el.progressCard.classList.remove("hidden");

    const tick = async () => {
      try {
        const job = await apiJson(`/api/enroll/${jobId}`);
        const total = Math.max(1, job.total || 1);
        const pct = Math.round((job.processed / total) * 100);
        el.progressCount.textContent = `${job.processed} / ${job.total}`;
        el.progressFill.style.width = `${pct}%`;
        el.progressLabel.textContent = job.current_file
          ? `Checking ${job.current_file}…`
          : "Checking reference portraits…";

        if (job.status === "completed") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          el.progressCard.classList.add("hidden");
          state.enrollSummary = job.summary;
          if (job.summary && Array.isArray(job.summary.identities)) {
            state.identities = job.summary.identities;
          } else {
            const idResp = await apiJson("/api/identities");
            state.identities = idResp.identities || [];
          }
          renderReferenceSection();
          renderPersonDropdown();
          updateReadinessMetrics();
          showToast(`Reference folder checked: ${job.summary.references_loaded} loaded, ${job.summary.references_rejected} rejected.`);
        } else if (job.status === "failed") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          el.progressCard.classList.add("hidden");
          showToast(job.error || "Reference enrollment failed.", true);
        }
      } catch (err) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        el.progressCard.classList.add("hidden");
        showToast(err.message, true);
      }
    };
    tick();
    state.pollTimer = setInterval(tick, 250);
  }

  async function uploadReferenceFiles(fileList) {
    if (!fileList || fileList.length === 0) return;
    showToast("");
    const fd = new FormData();
    for (const f of fileList) {
      fd.append("files", f, f.webkitRelativePath || f.name);
    }
    fd.append("clear_existing", el.chkReplaceRefs.checked ? "true" : "false");
    try {
      const started = await apiJson("/api/enroll", { method: "POST", body: fd });
      pollEnrollJob(started.job_id);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function enrollFromLocalPath() {
    const pathVal = (el.refPathInput.value || "").trim();
    if (!pathVal) {
      showToast("Enter a local folder path first.", true);
      return;
    }
    showToast("");
    try {
      const started = await apiJson("/api/enroll/from-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_path: pathVal,
          clear_existing: el.chkReplaceRefs.checked,
          persist_cache: false,
        }),
      });
      pollEnrollJob(started.job_id);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function resolveMultiFace(token, candidateIndex) {
    try {
      const res = await apiJson("/api/enroll/resolve-multiface", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, candidate_index: candidateIndex }),
      });
      state.enrollSummary = res.summary || state.enrollSummary;
      state.identities = res.identities || state.identities;
      renderReferenceSection();
      renderPersonDropdown();
      updateReadinessMetrics();
      showToast(
        candidateIndex === null
          ? `Skipped multi-face photo for ${res.display_name}.`
          : `Resolved multi-face selection for ${res.display_name}.`
      );
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function resolveExtraPhoto(token, decision) {
    try {
      const res = await apiJson("/api/enroll/resolve-extra", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, decision }),
      });
      state.enrollSummary = res.summary || state.enrollSummary;
      state.identities = res.identities || state.identities;
      renderReferenceSection();
      renderPersonDropdown();
      updateReadinessMetrics();
      showToast(
        decision === "keep"
          ? `Kept extra reference photo for ${res.display_name}.`
          : `Rejected conflicting extra photo for ${res.display_name}.`
      );
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function analyseGroupFile(file) {
    if (!file) return;
    showToast("Analysing group photo locally…");
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      const payload = await apiJson("/api/identify", { method: "POST", body: fd });
      applySessionPayload(payload, false);
      showToast(`Detected ${payload.summary.faces_detected} face(s) in ${payload.filename}.`);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function runSampleDemo() {
    showToast("Loading offline sample reference folder and group photo…");
    try {
      const payload = await apiJson("/api/demo", { method: "POST" });
      if (payload.demo_enrollment) {
        state.enrollSummary = payload.demo_enrollment;
      }
      renderReferenceSection();
      applySessionPayload(payload, false);
      showToast(
        payload.demo_note
          ? `Offline sample loaded (${payload.demo_note}).`
          : "Offline sample loaded. Click any face to test Evidence Cards."
      );
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function applyFaceDecision(action, identityId = null) {
    if (!state.session || state.selectedFaceIndex === null) return;
    const note = (el.evNoteInput.value || "").trim() || null;
    try {
      const payload = await apiJson(`/api/sessions/${state.session.session_id}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          face_index: state.selectedFaceIndex,
          action,
          identity_id: identityId,
          reason_note: note,
        }),
      });
      el.evNoteInput.value = "";
      applySessionPayload(payload, true);
      showToast(`Saved manual decision for Face ${state.selectedFaceIndex + 1} (this session only).`);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function undoLastEdit() {
    if (!state.session) return;
    try {
      const payload = await apiJson(`/api/sessions/${state.session.session_id}/undo`, {
        method: "POST",
      });
      applySessionPayload(payload, true);
      showToast("Reverted last manual edit.");
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function runSecondLook() {
    if (!state.session) return;
    showToast("Scanning 2×2 tiles for smaller faces…");
    try {
      const payload = await apiJson(`/api/sessions/${state.session.session_id}/second-look`, {
        method: "POST",
      });
      applySessionPayload(payload, true);
      const count = payload.second_look ? payload.second_look.new_faces_added : 0;
      showToast(`Small-face scan complete: ${count} additional face(s) found.`);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function submitManualBox(x, y, w, h) {
    if (!state.session) return;
    try {
      const payload = await apiJson(`/api/sessions/${state.session.session_id}/manual-box`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) }),
      });
      if (payload.manual_box && payload.manual_box.added_face_index !== null) {
        state.selectedFaceIndex = payload.manual_box.added_face_index;
      }
      applySessionPayload(payload, true);
      showToast((payload.manual_box && payload.manual_box.message) || "Checked drawn region.");
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function removeManualBox() {
    if (!state.session || state.selectedFaceIndex === null) return;
    try {
      const payload = await apiJson(
        `/api/sessions/${state.session.session_id}/faces/${state.selectedFaceIndex}/remove`,
        { method: "POST" }
      );
      applySessionPayload(payload, false);
      showToast("Removed manually drawn box.");
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function clearSession() {
    try {
      await apiJson("/api/wipe", { method: "POST" });
      state.session = null;
      state.identities = [];
      state.enrollSummary = null;
      state.selectedFaceIndex = null;
      state.groupImageEl = null;
      state.drawMode = false;
      el.btnManualBox.classList.remove("is-active");
      el.calibrationBadge.classList.add("hidden");
      el.btnUndo.classList.add("hidden");
      el.refSection.classList.add("hidden");
      el.workspace.classList.add("hidden");
      el.refFolderInput.value = "";
      el.groupFileInput.value = "";
      renderPersonDropdown();
      updateReadinessMetrics();
      showToast("Session cleared. All loaded references and photo edits were removed from local memory.");
    } catch (err) {
      showToast(err.message, true);
    }
  }

  function canvasCoords(evt) {
    const rect = el.canvas.getBoundingClientRect();
    const scaleX = el.canvas.width / rect.width;
    const scaleY = el.canvas.height / rect.height;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY,
    };
  }

  // Event listeners
  el.refFolderInput.addEventListener("change", (e) => uploadReferenceFiles(e.target.files));
  el.btnTogglePathEnroll.addEventListener("click", () => {
    el.pathEnrollBox.classList.toggle("hidden");
  });
  el.btnPathEnroll.addEventListener("click", enrollFromLocalPath);

  el.groupFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      analyseGroupFile(e.target.files[0]);
    }
  });
  el.groupDropzone.addEventListener("click", () => el.groupFileInput.click());
  el.groupDropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.groupFileInput.click();
    }
  });
  el.groupDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.groupDropzone.classList.add("drag-over");
  });
  el.groupDropzone.addEventListener("dragleave", () => {
    el.groupDropzone.classList.remove("drag-over");
  });
  el.groupDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    el.groupDropzone.classList.remove("drag-over");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) analyseGroupFile(file);
  });

  el.btnDemo.addEventListener("click", runSampleDemo);
  el.btnClearSession.addEventListener("click", clearSession);
  el.btnUndo.addEventListener("click", undoLastEdit);

  el.btnToggleLabels.addEventListener("click", () => {
    state.showLabels = !state.showLabels;
    el.btnToggleLabels.textContent = state.showLabels ? "Hide labels" : "Show labels";
    drawCanvas();
  });
  el.btnSecondLook.addEventListener("click", runSecondLook);
  el.btnManualBox.addEventListener("click", () => {
    state.drawMode = !state.drawMode;
    el.btnManualBox.classList.toggle("is-active", state.drawMode);
    showToast(
      state.drawMode
        ? "Drag a rectangle over an undetected face on the photo."
        : "Exited manual box drawing."
    );
  });

  el.canvas.addEventListener("mousedown", (e) => {
    if (!state.session) return;
    const pt = canvasCoords(e);
    if (state.drawMode) {
      state.dragStart = pt;
      state.dragCurrent = pt;
      return;
    }
    const active = (state.session.faces || []).filter((f) => !f.removed);
    for (const face of active) {
      const [bx, by, bw, bh] = face.bbox;
      if (pt.x >= bx && pt.x <= bx + bw && pt.y >= by && pt.y <= by + bh) {
        selectFace(face.face_index);
        return;
      }
    }
  });
  el.canvas.addEventListener("mousemove", (e) => {
    if (!state.drawMode || !state.dragStart) return;
    state.dragCurrent = canvasCoords(e);
    drawCanvas();
  });
  el.canvas.addEventListener("mouseup", () => {
    if (!state.drawMode || !state.dragStart || !state.dragCurrent) return;
    const x = Math.min(state.dragStart.x, state.dragCurrent.x);
    const y = Math.min(state.dragStart.y, state.dragCurrent.y);
    const w = Math.abs(state.dragCurrent.x - state.dragStart.x);
    const h = Math.abs(state.dragCurrent.y - state.dragStart.y);
    state.dragStart = null;
    state.dragCurrent = null;
    state.drawMode = false;
    el.btnManualBox.classList.remove("is-active");
    drawCanvas();
    if (w >= 10 && h >= 10) {
      submitManualBox(x, y, w, h);
    }
  });

  el.btnEvConfirm.addEventListener("click", () => applyFaceDecision("confirm"));
  el.btnEvUnknown.addEventListener("click", () => applyFaceDecision("mark_unknown"));
  el.btnEvUseRunnerup.addEventListener("click", () => {
    const iid = el.btnEvUseRunnerup.dataset.identityId;
    if (iid) applyFaceDecision("assign", iid);
  });
  el.btnEvAssign.addEventListener("click", () => {
    const iid = el.evPersonSelect.value;
    if (!iid) {
      showToast("Pick a person from the dropdown list first.", true);
      return;
    }
    applyFaceDecision("assign", iid);
  });
  el.btnEvRemoveManual.addEventListener("click", removeManualBox);

  el.btnExportPng.addEventListener("click", () => {
    if (!state.session) return;
    window.location.href = `/api/sessions/${state.session.session_id}/export/png`;
  });
  el.btnExportCsv.addEventListener("click", () => {
    if (!state.session) return;
    window.location.href = `/api/sessions/${state.session.session_id}/export/csv`;
  });
  el.btnExportPrint.addEventListener("click", () => {
    if (!state.session) return;
    window.open(`/api/sessions/${state.session.session_id}/export/print?include_thumbs=true`, "_blank");
  });

  document.querySelectorAll("[data-ref-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.refTab = btn.dataset.refTab;
      document.querySelectorAll("[data-ref-tab]").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      renderPendingAndRejected();
    });
  });

  document.querySelectorAll("[data-status-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.statusFilter = btn.dataset.statusFilter;
      document.querySelectorAll("[data-status-filter]").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      renderResultsList();
    });
  });

  refreshHealth();
})();
