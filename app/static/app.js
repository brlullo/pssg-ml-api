"use strict";

const listEl = document.getElementById("model-list");
const detailEl = document.getElementById("model-detail");
const backButton = document.getElementById("back-button");
const detailTitle = document.getElementById("detail-title");
const detailDescription = document.getElementById("detail-description");
const infoBody = document.getElementById("model-info-body");
const form = document.getElementById("predict-form");
const formFields = document.getElementById("form-fields");
const formError = document.getElementById("form-error");
const submitButton = document.getElementById("submit-button");
const resultEl = document.getElementById("result");
const scenarioResultEl = document.getElementById("scenario-result");
const scenarioGrid = document.getElementById("scenario-grid");
const scenarioInterpretation = document.getElementById("scenario-interpretation");
const resultPct = document.getElementById("result-pct");
const resultBar = document.getElementById("result-bar");
const resultThresholdMark = document.getElementById("result-threshold-mark");
const resultInterpretation = document.getElementById("result-interpretation");

let currentModel = null; // detail payload for the selected model
let lastInputSummary = []; // "label: value" lines from the last submit
let lastResultSummary = ""; // one-line result from the last prediction

function prettyName(name) {
  return name
    .replace(/^(Pre-Index_|Post-Index_)/, "")
    .replace(/_/g, " ")
    .trim();
}

function fmtPct(p) {
  return (p * 100).toFixed(1) + "%";
}

async function loadModels() {
  try {
    const res = await fetch("/models");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const models = await res.json();
    if (!models.length) {
      listEl.innerHTML = "<p class='loading'>No models are currently available.</p>";
      return;
    }
    listEl.innerHTML = "";
    for (const m of models) {
      const card = document.createElement("div");
      card.className = "model-card";
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.innerHTML =
        "<h2></h2><p></p><span class='meta'></span>";
      card.querySelector("h2").textContent = m.display_name;
      card.querySelector("p").textContent = m.description;
      card.querySelector(".meta").textContent =
        m.algorithm + " · v" + m.version;
      const open = () => showDetail(m.name);
      card.addEventListener("click", open);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
      });
      listEl.appendChild(card);
    }
  } catch (err) {
    listEl.innerHTML =
      "<p class='error'>Could not load models (" + err.message + "). Please refresh.</p>";
  }
}

async function showDetail(name) {
  const res = await fetch("/models/" + encodeURIComponent(name));
  if (!res.ok) return;
  currentModel = await res.json();

  detailTitle.textContent = currentModel.display_name;
  detailDescription.textContent = currentModel.description;
  renderInfo(currentModel);
  renderForm(currentModel);

  resultEl.hidden = true;
  scenarioResultEl.hidden = true;
  formError.hidden = true;
  listEl.hidden = true;
  detailEl.hidden = false;
  window.scrollTo({ top: 0 });
}

function renderInfo(m) {
  const met = m.metrics || {};
  const cohort = m.cohort || {};
  const rows = [
    ["Algorithm", m.algorithm + " (" + m.framework + ")"],
    ["Version", m.version + (m.last_updated ? " · updated " + m.last_updated : "")],
    ["Test AUC", met.test_auc],
    ["C-statistic", met.c_statistic],
    ["Sensitivity / Specificity",
      met.sensitivity != null ? met.sensitivity + " / " + met.specificity : null],
    ["Decision threshold", met.optimal_threshold != null ? fmtPct(met.optimal_threshold) : null],
    ["Cohort", cohort.total_patients != null ? cohort.total_patients + " patients" : null],
  ].filter(([, v]) => v != null);

  const table = document.createElement("table");
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    const td2 = document.createElement("td");
    td1.textContent = k;
    td2.textContent = v;
    tr.append(td1, td2);
    table.appendChild(tr);
  }
  infoBody.innerHTML = "";
  infoBody.appendChild(table);
  if (m.limitations) {
    const p = document.createElement("p");
    p.className = "limitations";
    p.textContent = "Limitations: " + m.limitations;
    infoBody.appendChild(p);
  }
}

function renderForm(m) {
  formFields.innerHTML = "";
  if (m.ui && m.ui.groups) {
    // Curated layout: named groups of display fields, each either a number
    // input bound to one feature or a select whose options set feature values
    for (const group of m.ui.groups) {
      const heading = document.createElement("h4");
      heading.className = "field-group-title";
      heading.textContent = group.title;
      formFields.appendChild(heading);
      for (const df of group.fields) {
        const wrap = document.createElement("div");
        wrap.className = "field";
        const label = document.createElement("label");
        label.textContent = df.label;
        wrap.appendChild(label);
        let input;
        if (df.control === "select") {
          input = document.createElement("select");
          for (const opt of df.options) {
            const o = document.createElement("option");
            o.textContent = opt.label;
            o.value = JSON.stringify(opt.set);
            input.appendChild(o);
          }
          input.dataset.set = "1";
        } else {
          input = document.createElement("input");
          input.type = "number";
          input.step = "any";
          input.inputMode = "decimal";
          input.required = true;
          input.placeholder = "Enter value";
          input.dataset.feature = df.feature;
        }
        input.dataset.label = df.label;
        label.htmlFor = input.id = "f-" + (df.feature || df.label.replace(/\W+/g, "_"));
        wrap.appendChild(input);
        formFields.appendChild(wrap);
      }
    }
    return;
  }
  renderAutoForm(m.features);
}

function renderAutoForm(features) {
  for (const f of features) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const id = "f-" + f.name;

    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = prettyName(f.name);
    wrap.appendChild(label);

    if (f.description) {
      const help = document.createElement("p");
      help.className = "help";
      help.textContent = f.description;
      wrap.appendChild(help);
    }

    let input;
    if (f.type === "binary") {
      input = document.createElement("select");
      for (const [text, value] of [["No", "0"], ["Yes", "1"]]) {
        const opt = document.createElement("option");
        opt.textContent = text;
        opt.value = value;
        input.appendChild(opt);
      }
    } else if (f.type === "categorical") {
      input = document.createElement("select");
      for (const option of f.options || []) {
        const opt = document.createElement("option");
        opt.textContent = option;
        opt.value = option;
        input.appendChild(opt);
      }
      input.dataset.kind = "categorical";
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.inputMode = "decimal";
      input.required = true;
      input.placeholder = "Enter value";
    }
    input.id = id;
    input.dataset.feature = f.name;
    wrap.appendChild(input);
    formFields.appendChild(wrap);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.hidden = true;
  resultEl.hidden = true;
  scenarioResultEl.hidden = true;

  const features = {};
  lastInputSummary = [];
  for (const input of formFields.querySelectorAll("[data-feature], [data-set]")) {
    const label = input.dataset.label || prettyName(input.dataset.feature || "");
    if (input.value === "") {
      formError.textContent = "Please enter a value for “" + label + "”.";
      formError.hidden = false;
      input.focus();
      return;
    }
    if (input.dataset.set) {
      // Composite select: its options carry {feature: value} assignments
      Object.assign(features, JSON.parse(input.value));
      lastInputSummary.push(label + ": " + input.options[input.selectedIndex].text);
    } else {
      features[input.dataset.feature] =
        input.dataset.kind === "categorical" ? input.value : Number(input.value);
      const shown = input.tagName === "SELECT"
        ? input.options[input.selectedIndex].text
        : input.value;
      lastInputSummary.push(label + ": " + shown);
    }
  }

  submitButton.disabled = true;
  try {
    const res = await fetch("/predict/" + encodeURIComponent(currentModel.name), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    });
    const body = await res.json();
    if (!res.ok) {
      let detail;
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI validation errors: [{loc: [...], msg: ...}, ...]
        detail = body.detail
          .map((d) => prettyName(String((d.loc || []).slice(-1)[0] || "input")) + ": " + d.msg)
          .join("; ");
      } else {
        detail = "The server could not process this request.";
      }
      formError.textContent = detail;
      formError.hidden = false;
      return;
    }
    showResult(body);
  } catch (err) {
    formError.textContent = "Request failed (" + err.message + "). Please try again.";
    formError.hidden = false;
  } finally {
    submitButton.disabled = false;
  }
});

function showScenarioResult(r) {
  scenarioGrid.innerHTML = "";
  for (const s of r.scenarios) {
    const card = document.createElement("div");
    card.className = "scenario-card";
    card.innerHTML =
      "<p class='scenario-label'></p><p class='scenario-pct'></p>" +
      "<p class='scenario-ci'></p><div class='bar-track'><div class='bar-fill'></div></div>";
    card.querySelector(".scenario-label").textContent = s.label;
    card.querySelector(".scenario-pct").textContent = fmtPct(s.probability);
    card.querySelector(".scenario-ci").textContent =
      "95% CI " + fmtPct(s.ci_lower) + " – " + fmtPct(s.ci_upper);
    card.querySelector(".bar-fill").style.width =
      Math.min(100, s.probability * 100) + "%";
    scenarioGrid.appendChild(card);
  }

  const without = r.scenarios.find((s) => s.label.startsWith("Without"));
  const withS = r.scenarios.find((s) => !s.label.startsWith("Without"));
  if (without && withS) {
    const diff = (without.probability - withS.probability) * 100;
    scenarioInterpretation.textContent =
      diff > 0
        ? "The model predicts an absolute risk reduction of " + diff.toFixed(1) +
          " percentage points with preoperative halo-gravity traction for this patient. " +
          "(Model: " + r.model_name + " v" + r.model_version + ")"
        : "The model does not predict a risk reduction with preoperative halo-gravity " +
          "traction for this patient. (Model: " + r.model_name + " v" + r.model_version + ")";
  }
  scenarioResultEl.hidden = false;
  scenarioResultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showResult(r) {
  if (r.scenarios) return showScenarioResult(r);
  resultPct.textContent = fmtPct(r.probability);
  resultBar.style.width = Math.min(100, r.probability * 100) + "%";
  resultThresholdMark.style.left = Math.min(100, r.threshold * 100) + "%";

  const above = r.probability >= r.threshold;
  resultInterpretation.textContent =
    "This predicted probability is " + (above ? "at or above" : "below") +
    " the model's decision threshold of " + fmtPct(r.threshold) +
    ", so the model classifies this patient as " +
    (above ? "higher" : "lower") + " risk. (Model: " +
    r.model_name + " v" + r.model_version + ")";

  const note = document.getElementById("result-note");
  const noteText = currentModel && currentModel.ui && currentModel.ui.result_note;
  note.textContent = noteText || "";
  note.hidden = !noteText;

  lastResultSummary =
    "Predicted UPROR probability: " + fmtPct(r.probability) +
    " (" + (above ? "above" : "below") + " decision threshold of " +
    fmtPct(r.threshold) + ")";

  resultEl.hidden = false;
  resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("copy-summary").addEventListener("click", async (e) => {
  const text = [
    currentModel.display_name + " (" + currentModel.name + " v" + currentModel.version + ")",
    "",
    "Inputs:",
    ...lastInputSummary.map((l) => "  " + l),
    "",
    lastResultSummary,
    "",
    "Research use only — not a substitute for clinical judgment.",
  ].join("\n");
  try {
    await navigator.clipboard.writeText(text);
    e.target.textContent = "Copied!";
    setTimeout(() => { e.target.textContent = "Copy summary"; }, 1500);
  } catch {
    e.target.textContent = "Copy failed";
    setTimeout(() => { e.target.textContent = "Copy summary"; }, 1500);
  }
});

backButton.addEventListener("click", () => {
  detailEl.hidden = true;
  listEl.hidden = false;
  currentModel = null;
});

loadModels();
