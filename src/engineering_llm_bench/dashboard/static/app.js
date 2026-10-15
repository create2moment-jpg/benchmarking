// engineering-llm-bench dashboard. Vanilla JS, no dependencies.
// All values from the API are rendered with textContent (model output is untrusted).
"use strict";

const DIMENSIONS = ["functional", "regression", "quality", "robustness", "workflow", "explanation"];
const SMALL_SAMPLE = 5;
const state = { report: null, selectedGroups: new Set(), currentAttempt: null };

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
const $ = (id) => document.getElementById(id);
const unknown = (v) => (v === null || v === undefined || v === "" ? "unknown" : v);
function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return typeof v === "number" ? v.toFixed(digits) : String(v);
}
const pct = (v) => (v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`);
const groupKey = (a) => `${a.model_alias} | ${a.harness}`;
function mean(xs) { return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null; }
function sd(xs) {
  if (xs.length < 2) return null;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
}
function quantile(xs, q) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
function setTable(table, headers, rows) {
  table.replaceChildren();
  const thead = el("tr", {}, headers.map((h) => el("th", { class: h.num ? "num" : null }, h.label ?? h)));
  table.append(thead);
  if (!rows.length) {
    table.append(el("tr", {}, el("td", { colspan: headers.length, class: "muted" }, "No data")));
    return;
  }
  for (const r of rows) table.append(r);
}
const td = (v, cls) => el("td", { class: cls }, v);
const num = (v) => el("td", { class: "num" }, v);

// ------------------------------------------------------------------ aggregation (client-side, from attempts)
function summarize(attempts) {
  const executed = attempts.filter((a) => a.status !== "skipped");
  const evaluated = executed.filter((a) => a.status === "completed");
  const withPass = evaluated.filter((a) => a.passed !== null && a.passed !== undefined);
  const scores = evaluated.filter((a) => a.score !== null).map((a) => a.score);
  const walls = executed.filter((a) => a.agent_wall_s !== null).map((a) => a.agent_wall_s);
  const calls = executed.reduce((s, a) => s + (a.tool_calls || 0), 0);
  const ok = executed.reduce((s, a) => s + (a.tool_calls_ok || 0), 0);
  const guarded = evaluated.filter((a) => a.regression_failed !== null && a.regression_failed !== undefined);
  const costRows = executed.filter((a) => a.cost_usd !== null && a.cost_usd !== undefined);
  const dims = {};
  for (const d of DIMENSIONS) {
    const v = evaluated.map((a) => a.dimensions?.[d]).filter((x) => x !== null && x !== undefined);
    dims[d] = { mean: mean(v), n: v.length };
  }
  return {
    n: attempts.length, executed: executed.length, evaluated: evaluated.length,
    skipped: attempts.length - executed.length,
    passRate: withPass.length ? withPass.filter((a) => a.passed).length / withPass.length : null,
    passed: withPass.filter((a) => a.passed).length, passN: withPass.length,
    scoreMean: mean(scores), scoreSd: sd(scores), scoreMin: scores.length ? Math.min(...scores) : null,
    scoreMax: scores.length ? Math.max(...scores) : null, scoreN: scores.length,
    wallP50: quantile(walls, 0.5), wallP90: quantile(walls, 0.9),
    timeoutRate: executed.length ? executed.filter((a) => a.failure_class === "timeout").length / executed.length : null,
    failures: executed.filter((a) => a.failure_class && a.failure_class !== "none").length,
    toolOk: calls ? ok / calls : null,
    regressionRate: guarded.length ? guarded.filter((a) => a.regression_failed).length / guarded.length : null,
    cost: costRows.length ? costRows.reduce((s, a) => s + a.cost_usd, 0) : null, costN: costRows.length,
    mock: attempts.some((a) => a.provider === "fake"),
    dims,
  };
}

// ------------------------------------------------------------------ tabs
function showTab(name) {
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("active", b.dataset.tab === name);
  for (const s of document.querySelectorAll("section.tab")) s.hidden = s.id !== name;
}
for (const b of document.querySelectorAll("#tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));

// ------------------------------------------------------------------ overview
function renderOverview(r) {
  const s = summarize(r.attempts);
  const interrupted = r.attempts.filter((a) => a.status === "interrupted").length;
  const tiles = [
    ["Runs", r.runs.length], ["Attempts", s.n], ["Evaluated", s.evaluated],
    ["Pass rate", pct(s.passRate)], ["Failures", s.failures], ["Skipped", s.skipped],
    ["Interrupted", interrupted], ["Data freshness", r.freshness ? r.freshness.replace("T", " ").slice(0, 16) : "-"],
  ];
  $("tiles").replaceChildren(...tiles.map(([k, v]) => el("div", { class: "tile" }, el("div", { class: "v" }, v), el("div", { class: "k" }, k))));
  const warnings = [...r.comparability.map((c) => `Non-comparable: ${c}`)];
  if (s.mock) warnings.push("Some runs use the MOCK provider: their scores validate the pipeline, not a model.");
  $("warnings").replaceChildren(...warnings.map((w) => el("div", { class: "warning" }, w)));

  const best = {};
  for (const a of r.attempts) {
    if (a.status !== "completed" || a.score === null) continue;
    const d = a.task_domain || "unknown";
    ((best[d] ??= {})[groupKey(a)] ??= []).push(a.score);
  }
  const rows = Object.entries(best).sort().map(([domain, groups]) => {
    const [key, scores] = Object.entries(groups).sort((x, y) => mean(y[1]) - mean(x[1]))[0];
    return el("tr", {}, td(domain), td(key), num(fmt(mean(scores))), num(scores.length),
      td(scores.length < SMALL_SAMPLE ? el("span", { class: "chip warn" }, "small sample") : ""));
  });
  setTable($("best-domain"), ["Domain", "Best model | harness", { label: "Mean", num: true }, { label: "n", num: true }, ""], rows);

  const byRun = {};
  for (const a of r.attempts) (byRun[a.run_id] ??= []).push(a);
  const runRows = r.runs.slice(0, 15).map((run) => {
    const as = byRun[run.id] || [];
    const done = as.filter((a) => ["completed", "skipped"].includes(a.status)).length;
    const statusCls = run.status === "completed" ? "pass" : run.status === "running" ? "" : "fail";
    return el("tr", {}, td(run.id), td((run.created_at || "").replace("T", " ")), td(el("span", { class: statusCls }, run.status)),
      td(run.source), td(run.model_alias), td(`${run.harness} ${run.harness_version || ""}`), td(run.profile || "-"),
      num(`${done}/${as.length}`), td(run.provider === "fake" ? el("span", { class: "chip warn" }, "mock") : ""));
  });
  setTable($("latest-runs"), ["Run", "Created", "Status", "Source", "Model", "Harness", "Profile", { label: "Done", num: true }, ""], runRows);
}

// ------------------------------------------------------------------ compare
function renderGroupPicker(r) {
  const keys = [...new Set(r.attempts.map(groupKey))].sort();
  if (!state.selectedGroups.size) keys.slice(0, 6).forEach((k) => state.selectedGroups.add(k));
  $("group-picker").replaceChildren(...keys.map((k) => {
    const box = el("input", { type: "checkbox", checked: state.selectedGroups.has(k) });
    box.addEventListener("change", () => { box.checked ? state.selectedGroups.add(k) : state.selectedGroups.delete(k); renderCompare(); });
    return el("label", {}, box, " ", k);
  }));
  const domains = [...new Set(r.attempts.map((a) => a.task_domain).filter(Boolean))].sort();
  const sel = $("compare-domain");
  sel.replaceChildren(el("option", { value: "" }, "all"), ...domains.map((d) => el("option", { value: d }, d)));
  sel.onchange = renderCompare;
}

function chart(groups) {
  const rowH = 34, left = 250, width = 460, height = groups.length * rowH + 28;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", left + width + 150);
  svg.setAttribute("height", height);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Mean aggregate score per model and harness with min-max range");
  const add = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    svg.append(n);
  };
  for (const t of [0, 25, 50, 75, 100]) {
    const x = left + (t * width) / 100;
    add("line", { class: "axis", x1: x, x2: x, y1: 0, y2: groups.length * rowH });
    add("text", { x, y: groups.length * rowH + 16, "text-anchor": "middle" }, String(t));
  }
  groups.forEach(([key, s], i) => {
    const y = i * rowH;
    add("text", { x: left - 8, y: y + 21, "text-anchor": "end" }, key.length > 36 ? key.slice(0, 35) + "..." : key);
    if (s.scoreMean === null) { add("text", { x: left + 4, y: y + 21 }, "no scored attempts"); return; }
    add("rect", { class: "bar", x: left, y: y + 8, width: (s.scoreMean * width) / 100, height: 18, rx: 2 });
    if (s.scoreN > 1) add("line", { class: "whisker", x1: left + (s.scoreMin * width) / 100, x2: left + (s.scoreMax * width) / 100, y1: y + 17, y2: y + 17 });
    add("text", { x: left + width + 8, y: y + 21 }, `${fmt(s.scoreMean)} (n=${s.scoreN}${s.scoreSd !== null ? `, sd ${fmt(s.scoreSd)}` : ""})`);
  });
  return svg;
}

function renderCompare() {
  const r = state.report;
  const domain = $("compare-domain").value;
  const attempts = r.attempts.filter((a) => state.selectedGroups.has(groupKey(a)) && (!domain || a.task_domain === domain));
  const byGroup = {};
  for (const a of attempts) (byGroup[groupKey(a)] ??= []).push(a);
  const groups = Object.entries(byGroup).sort().map(([k, as]) => [k, summarize(as)]);

  const warnings = [];
  if (groups.length < 2) warnings.push("Select at least two model/harness groups to compare.");
  for (const [k, s] of groups) {
    if (s.scoreN < SMALL_SAMPLE) warnings.push(`${k}: small sample (n=${s.scoreN}); do not rank on this alone.`);
    if (s.mock) warnings.push(`${k}: MOCK provider - pipeline validation only.`);
  }
  for (const c of r.comparability) warnings.push(`Non-comparable: ${c}`);
  $("compare-warnings").replaceChildren(...warnings.map((w) => el("div", { class: "warning" }, w)));

  setTable($("compare-table"),
    ["Model | harness", { label: "Evaluated", num: true }, { label: "Skipped", num: true }, { label: "Pass rate", num: true },
      { label: "Score mean", num: true }, { label: "sd", num: true }, { label: "p50 wall s", num: true }, { label: "p90 wall s", num: true },
      { label: "Timeouts", num: true }, { label: "Failures", num: true }, { label: "Tool ok", num: true }, { label: "Regressions", num: true }, { label: "Cost", num: true }],
    groups.map(([k, s]) => el("tr", {}, td(k), num(s.evaluated), num(s.skipped), num(`${pct(s.passRate)} (${s.passed}/${s.passN})`),
      num(fmt(s.scoreMean)), num(fmt(s.scoreSd)), num(fmt(s.wallP50)), num(fmt(s.wallP90)), num(pct(s.timeoutRate)),
      num(s.failures), num(pct(s.toolOk)), num(pct(s.regressionRate)),
      num(s.cost === null ? "unknown" : `$${s.cost.toFixed(4)} (${s.costN}/${s.executed})`))));

  $("compare-chart").replaceChildren(groups.length ? chart(groups) : el("div", { class: "muted" }, "No data"),
    el("div", { class: "muted" }, "Bar = mean 0-100 aggregate score; line = min-max across attempts; n = scored attempts."));

  setTable($("compare-dims"), ["Model | harness", ...DIMENSIONS.map((d) => ({ label: d, num: true }))],
    groups.map(([k, s]) => el("tr", {}, td(k), ...DIMENSIONS.map((d) => num(s.dims[d].n ? `${fmt(s.dims[d].mean, 2)} (n=${s.dims[d].n})` : "unassessed")))));

  renderPerf(groups.map(([k]) => k));
  const tasks = [...new Set(attempts.map((a) => a.task_id))].sort();
  setTable($("compare-tasks"), ["Task", ...groups.map(([k]) => ({ label: k, num: true }))],
    tasks.map((t) => el("tr", {}, td(t), ...groups.map(([k]) => {
      const cell = byGroup[k].filter((a) => a.task_id === t);
      if (!cell.length) return num("-");
      const s = summarize(cell);
      if (s.scoreN === 0) return num(el("span", { class: "muted" }, [...new Set(cell.map((a) => a.status))].join(",")));
      return num(el("span", {}, `${fmt(s.scoreMean, 0)} `, el("span", { class: s.passed === s.passN ? "pass" : "fail" }, `${s.passed}/${s.passN}`)));
    }))));
}

// ------------------------------------------------------------------ performance
function sessionTokens(delta) {
  if (!delta) return "unknown";
  const pick = (re) => Object.entries(delta).filter(([k]) => re.test(k)).reduce((s, [, v]) => s + v, 0);
  const p = pick(/prompt_tokens/), g = pick(/generation_tokens|tokens_predicted_total/);
  return `${Math.round(p)} in / ${Math.round(g)} out`;
}
function renderPerf(keys) {
  const groups = state.report.groups || {};
  const u = (v, d = 1) => (v === null || v === undefined ? "unknown" : typeof v === "number" ? fmt(v, d) : String(v));
  setTable($("compare-perf"),
    ["Model | harness", "Server", "Quant / params", { label: "Context", num: true }, { label: "GPU share", num: true },
      { label: "Gen tok/s", num: true }, { label: "Prompt tok/s", num: true }, { label: "TTFT s", num: true }, { label: "Cold load s", num: true },
      "Session tokens", { label: "Agent out tok/s", num: true }, { label: "Model s p50", num: true }, { label: "Tools s p50", num: true },
      { label: "Eval s p50", num: true }, { label: "Eval CPU s p50", num: true }, { label: "Eval peak MB", num: true }],
    keys.filter((k) => groups[k]).map((k) => {
      const g = groups[k], p = g.server_perf || {}, t = g.time_profile || {}, c = g.compute || {};
      return el("tr", {}, td(k), td(p.server ? `${p.server} ${p.server_version || ""}` : "unknown"),
        td(p.quantization || p.parameter_size ? `${p.quantization || "?"} / ${p.parameter_size || "?"}` : "unknown"),
        num(u(p.context_length, 0)), num(p.gpu_fraction === null || p.gpu_fraction === undefined ? "unknown" : pct(p.gpu_fraction)),
        num(u(p.gen_tps)), num(u(p.prompt_tps, 0)), num(u(p.ttft_s, 2)), num(u(p.cold_load_s, 2)),
        td(sessionTokens(p.session_tokens)), num(u(t.e2e_output_tps_median)), num(u(t.model_s_median)),
        num(u(t.tools_and_overhead_s_median)), num(u(t.eval_s_median)), num(u(c.eval_cpu_s_median, 2)), num(u(c.eval_max_rss_mb_peak, 0)));
    }));
}

// ------------------------------------------------------------------ history
function fillSelect(id, values) {
  const sel = $(id);
  sel.replaceChildren(el("option", { value: "" }, "all"), ...values.map((v) => el("option", { value: v }, v)));
  sel.onchange = renderHistory;
}
function initHistory(r) {
  const uniq = (f) => [...new Set(r.attempts.map(f).filter((x) => x !== null && x !== undefined))].sort();
  fillSelect("f-model", uniq((a) => a.model_alias));
  fillSelect("f-harness", uniq((a) => a.harness));
  fillSelect("f-task", uniq((a) => a.task_id));
  fillSelect("f-profile", uniq((a) => a.profile));
  fillSelect("f-provider", uniq((a) => a.provider));
  fillSelect("f-config", uniq((a) => a.config_fingerprint));
  for (const id of ["f-from", "f-to", "f-outcome"]) $(id).onchange = renderHistory;
}
function majority(values) {
  const counts = {};
  for (const v of values) counts[v] = (counts[v] || 0) + 1;
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];
}
function renderHistory() {
  const r = state.report;
  const f = (id) => $(id).value;
  const outcome = f("f-outcome");
  const rows = r.attempts.filter((a) => {
    const day = (a.run_created_at || "").slice(0, 10);
    if (f("f-from") && day < f("f-from")) return false;
    if (f("f-to") && day > f("f-to")) return false;
    if (f("f-model") && a.model_alias !== f("f-model")) return false;
    if (f("f-harness") && a.harness !== f("f-harness")) return false;
    if (f("f-task") && a.task_id !== f("f-task")) return false;
    if (f("f-profile") && a.profile !== f("f-profile")) return false;
    if (f("f-provider") && a.provider !== f("f-provider")) return false;
    if (f("f-config") && a.config_fingerprint !== f("f-config")) return false;
    if (outcome === "pass" && a.passed !== 1) return false;
    if (outcome === "fail" && !(a.status === "completed" && a.passed === 0)) return false;
    if (outcome === "skipped" && a.status !== "skipped") return false;
    if (outcome === "interrupted" && a.status !== "interrupted") return false;
    if (outcome === "error" && a.status !== "failed") return false;
    return true;
  });
  const hashByTask = {};
  for (const a of r.attempts) if (a.task_hash) (hashByTask[a.task_id] ??= []).push(a.task_hash);
  const majorityHash = Object.fromEntries(Object.entries(hashByTask).map(([t, hs]) => [t, majority(hs)]));
  const majorityEnv = majority(r.attempts.map((a) => a.env_fingerprint).filter(Boolean));
  $("history-count").textContent = `${rows.length} attempt(s)`;
  setTable($("history-table"),
    ["Run date", "Model", "Harness", "Profile", "Task", "Status", "Failure", { label: "Score", num: true }, "Passed", { label: "Wall s", num: true }, "Comparability"],
    rows.map((a) => {
      const flags = [];
      if (a.task_hash && majorityHash[a.task_id] && a.task_hash !== majorityHash[a.task_id]) flags.push(el("span", { class: "chip warn" }, "task changed"));
      if (!a.task_hash && a.status !== "skipped") flags.push(el("span", { class: "chip warn" }, "task version unknown"));
      if (a.env_fingerprint && majorityEnv && a.env_fingerprint !== majorityEnv) flags.push(el("span", { class: "chip warn" }, "env differs"));
      if (a.provider === "fake") flags.push(el("span", { class: "chip" }, "mock"));
      const passed = a.passed === null || a.passed === undefined ? "-" : a.passed ? el("span", { class: "pass" }, "yes") : el("span", { class: "fail" }, "no");
      return el("tr", { class: "click", onclick: () => openAttempt(a.id) },
        td((a.run_created_at || "").replace("T", " ").slice(0, 16)), td(a.model_alias), td(a.harness), td(a.profile || "-"),
        td(`${a.task_id} v${a.task_version || "?"}`), td(a.status), td(a.failure_class || "-"), num(fmt(a.score)), td(passed),
        num(fmt(a.agent_wall_s)), td(flags));
    }));
}

// ------------------------------------------------------------------ detail
async function loadArtifact(attemptId, name, target) {
  target.textContent = "loading...";
  const res = await fetch(`/api/artifact?attempt=${encodeURIComponent(attemptId)}&name=${encodeURIComponent(name)}`);
  target.textContent = res.ok ? (await res.text()) || "(empty)" : "artifact not found";
}
async function openAttempt(id) {
  const tab = $("detail-tab");
  tab.disabled = false;
  showTab("detail");
  const sec = $("detail");
  sec.replaceChildren(el("div", { class: "muted" }, "loading..."));
  const res = await fetch(`/api/attempt/${encodeURIComponent(id)}`);
  if (!res.ok) { sec.replaceChildren(el("div", { class: "card error" }, "Attempt not found")); return; }
  const d = await res.json();
  const a = d.attempt, run = d.run;
  const kv = (pairs) => el("div", { class: "kv" }, pairs.flatMap(([k, v]) => [el("div", {}, k), el("div", {}, v)]));
  const viewer = el("pre", {}, "Select an artifact or check evidence to view it.");
  const tokens = a.input_tokens === null && a.output_tokens === null ? "unknown (not reported)" : `${unknown(a.input_tokens)} in / ${unknown(a.output_tokens)} out`;
  sec.replaceChildren(
    el("h2", {}, `${a.task_id} v${unknown(a.task_version)} - ${run.model_alias} [${run.harness}]`),
    a.error ? el("div", { class: "warning" }, `Error: ${a.error}`) : "",
    a.skip_reason ? el("div", { class: "warning" }, `Skipped: ${a.skip_reason}`) : "",
    run.provider === "fake" ? el("div", { class: "warning" }, "MOCK provider: this result validates the pipeline, not a model.") : "",
    el("div", { class: "grid2" },
      el("div", { class: "card" }, el("h3", {}, "Outcome"), kv([
        ["Status", a.status], ["Failure class", unknown(a.failure_class)], ["Score (0-100)", fmt(a.score)],
        ["Coverage", pct(a.coverage)], ["Passed", a.passed === null ? "-" : a.passed ? "yes" : "no"],
        ["Regression", a.regression_failed === null ? "-" : a.regression_failed ? "failed" : "ok"],
        ["Agent wall", `${fmt(a.agent_wall_s)} s`], ["Eval wall", `${fmt(a.eval_wall_s)} s`],
        ["Turns / tools ok", `${unknown(a.turns)} / ${unknown(a.tool_calls_ok)} of ${unknown(a.tool_calls)}`],
        ["Malformed outputs", unknown(a.malformed_outputs)], ["Tokens", tokens],
        ["Model time", a.model_time_s === null ? "unknown" : `${fmt(a.model_time_s)} s`],
        ["Agent out tok/s", a.model_time_s && a.output_tokens ? fmt(a.output_tokens / a.model_time_s) : "unknown"],
        ["Eval CPU / peak RSS", a.eval_cpu_s === null ? "unknown" : `${fmt(a.eval_cpu_s, 2)} s / ${unknown(a.eval_max_rss_mb)} MB`],
        ["Est. active time", (() => { const t = (d.performance?.timing?.tasks || []).find((x) => x.task_id === a.task_id); return t && t.estimated_s !== null ? `${fmt(t.estimated_s)} s (file timestamps)` : "unknown"; })()],
        ["Cost", a.cost_usd === null ? "unknown" : `$${a.cost_usd.toFixed(5)}`],
      ])),
      el("div", { class: "card" }, el("h3", {}, "Configuration"), kv([
        ["Run", run.id], ["Source", run.source], ["Model alias", run.model_alias], ["Provider", unknown(run.provider)],
        ["Provider model id", unknown(run.provider_model_id)], ["Endpoint", unknown(run.endpoint_type)],
        ["Remote", run.remote === null ? "unknown" : run.remote ? "yes" : "no"],
        ["Harness", `${run.harness} ${unknown(run.harness_version)}`], ["Mode", run.mode], ["Profile", unknown(run.profile)],
        ["Task hash", unknown(a.task_hash && a.task_hash.slice(0, 16))], ["Config fp", unknown(run.config_fingerprint)],
        ["Env fp", unknown(run.env_fingerprint)], ["Baseline / final", `${unknown(a.baseline_hash && a.baseline_hash.slice(0, 10))} / ${unknown(a.final_hash && a.final_hash.slice(0, 10))}`],
      ]))),
    el("h3", {}, "Score breakdown"),
    el("div", { class: "table-wrap" }, (() => {
      const t = el("table");
      setTable(t, ["Dimension", { label: "Score", num: true }, { label: "Weight", num: true }, "Source", "Details"],
        d.dimensions.map((x) => el("tr", {}, td(x.dimension), num(x.score === null ? "unassessed" : fmt(x.score, 2)), num(fmt(x.weight, 0)), td(x.source), td(x.details || ""))));
      return t;
    })()),
    el("h3", {}, "Evaluator evidence"),
    el("div", { class: "table-wrap" }, (() => {
      const t = el("table");
      setTable(t, ["Check", "Kind", "Dimension", "Status", { label: "Score", num: true }, "Message", ""],
        d.checks.map((c) => {
          const name = c.evidence_path ? `checks/${c.check_id}.log` : null;
          const cls = c.status === "pass" ? "pass" : c.status === "error" || c.status === "skipped" ? "warn" : "fail";
          return el("tr", {}, td(c.check_id), td(c.kind), td(c.dimension), td(el("span", { class: cls }, c.status)),
            num(fmt(c.score, 2)), td(c.message || ""), td(name ? el("a", { onclick: () => loadArtifact(a.id, name, viewer) }, "evidence") : ""));
        }));
      return t;
    })()),
    el("h3", {}, "Artifacts"),
    el("div", {}, d.artifacts.filter((x) => !x.name.startsWith("checks/")).map((x) =>
      el("a", { class: "chip", onclick: () => loadArtifact(a.id, x.name, viewer) }, `${x.name} (${x.bytes} B)`))),
    viewer,
    d.reviews.length ? el("h3", {}, "Reviews (human / judge, kept separate from deterministic scores)") : "",
    d.reviews.length ? el("div", { class: "table-wrap" }, (() => {
      const t = el("table");
      setTable(t, ["Dimension", "Reviewer", "Source", { label: "Score", num: true }, "Rationale"],
        d.reviews.map((rv) => el("tr", {}, td(rv.dimension), td(rv.reviewer), td(rv.source), num(fmt(rv.score, 2)), td(rv.rationale || ""))));
      return t;
    })()) : "",
    d.notes.length ? el("div", {}, el("h3", {}, "Notes"), el("ul", {}, d.notes.map((n) => el("li", {}, n)))) : "",
    el("h3", {}, "Model server performance (run level)"),
    el("pre", {}, d.performance ? JSON.stringify(d.performance_headline || d.performance, null, 2) : "not recorded for this run"),
    el("h3", {}, "Environment"), el("pre", {}, d.environment ? JSON.stringify(d.environment, null, 2) : "unknown"),
    el("h3", {}, "Model / harness / run config"),
    el("pre", {}, JSON.stringify({ model: d.model_config, harness: d.harness_config, run: d.run_config }, null, 2)),
  );
}

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    const res = await fetch("/api/report");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const r = await res.json();
    state.report = r;
    $("loading").hidden = true;
    $("suite").textContent = r.runs[0] ? `suite ${r.runs[0].suite_version || "?"}` : "";
    if (!r.runs.length) { $("empty").hidden = false; $("tabs").hidden = true; return; }
    renderOverview(r);
    renderGroupPicker(r);
    renderCompare();
    initHistory(r);
    renderHistory();
    showTab("overview");
  } catch (err) {
    $("loading").hidden = true;
    const box = $("error");
    box.hidden = false;
    box.textContent = `Could not load results: ${err.message}`;
  }
}
boot();
