"""Self-contained, offline-friendly HTML report for an EvalReport.

No CDN / external assets — the file should open and work from a plain
file:// URL or inside an air-gapped CI artifact viewer. Designed to be
readable by a non-technical audience (light by default — dark mode is an
explicit toggle, never an automatic OS-preference flip) as well as useful for
engineers: KPI-style stat tiles up top, then per-metric cards with
mean/median/min/max plus an ECDF plot of the score distribution, all
re-computed live as the target/tag/status filters change.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genevals.core.types import EvalReport

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --plane:        #f9f9f7;
  --surface:      #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:   #898781;
  --grid:         #e1e0d9;
  --baseline:     #c3c2b7;
  --border:       rgba(11, 11, 11, 0.10);
  --good:         #0ca30c;
  --warning:      #b8780f;
  --critical:     #d03b3b;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane:        #0d0d0d;
  --surface:      #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:   #898781;
  --grid:         #2c2c2a;
  --baseline:     #383835;
  --border:       rgba(255, 255, 255, 0.10);
  --good:         #0ca30c;
  --warning:      #d99a1c;
  --critical:     #e66767;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--plane); color: var(--text-primary); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 28px 28px 4px; }
h1 { font-size: 21px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 15px; margin: 0 0 12px; font-weight: 600; color: var(--text-primary); }
h3 { font-size: 14px; margin: 0; font-weight: 600; }
.meta { color: var(--text-secondary); font-size: 13px; }
.theme-toggle { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; width: 36px; height: 36px; color: var(--text-secondary); cursor: pointer; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
.theme-toggle:hover { border-color: var(--baseline); }

.kpi-row { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px 28px 4px; }
.stat-tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; flex: 1 1 170px; min-width: 170px; }
.stat-tile .value { font-size: 26px; font-weight: 600; line-height: 1.2; }
.stat-tile .value.good { color: var(--good); }
.stat-tile .value.warning { color: var(--warning); }
.stat-tile .value.critical { color: var(--critical); }
.stat-tile .label { font-size: 12px; color: var(--text-secondary); margin-top: 3px; }

section { padding: 20px 28px 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; padding: 12px 28px 4px; align-items: center; }
select, input[type=text] { background: var(--surface); color: var(--text-primary); border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px; font-size: 13px; }
input[type=text] { flex: 1; min-width: 220px; }

.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px 18px; }
.metric-card-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; }
.badge { font-size: 11px; color: var(--text-muted); background: var(--plane); border: 1px solid var(--border); border-radius: 999px; padding: 2px 8px; }

.stats-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 10px; }
.stats-table th, .stats-table td { padding: 4px 6px; text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.stats-table th:first-child, .stats-table td:first-child { text-align: left; color: var(--text-primary); }
.stats-table th { font-weight: 500; color: var(--text-muted); border-bottom: 1px solid var(--grid); }

.legend-key { display: inline-block; width: 14px; height: 2px; border-radius: 1px; margin-right: 6px; vertical-align: middle; }
.chart-wrap { position: relative; }
.ecdf-svg { width: 100%; height: auto; display: block; touch-action: none; }
.ecdf-svg .grid { stroke: var(--grid); stroke-width: 1; }
.ecdf-svg .baseline { stroke: var(--baseline); stroke-width: 1; }
.ecdf-svg .axis-label { fill: var(--text-muted); font-size: 9.5px; }
.ecdf-svg .crosshair { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2 2; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; font-size: 11.5px; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; }
.empty { color: var(--text-muted); font-size: 12px; font-style: italic; }

.meter-row { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 6px; }
.meter-row .name { flex: 0 0 140px; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.meter-track { flex: 1; height: 8px; border-radius: 4px; background: var(--plane); border: 1px solid var(--border); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 4px 0 0 4px; }
.meter-row .pct { flex: 0 0 40px; text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; }

.tablewrap { overflow-x: auto; padding: 8px 28px 4px; }
table.results { border-collapse: collapse; width: 100%; font-size: 13px; }
table.results th, table.results td { border-bottom: 1px solid var(--grid); padding: 8px 10px; text-align: left; vertical-align: top; }
table.results th { position: sticky; top: 0; background: var(--plane); white-space: nowrap; font-weight: 600; color: var(--text-secondary); }
td.truncate { max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
td.truncate:hover { white-space: normal; overflow: visible; }
.tag { display: inline-block; background: var(--plane); border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px; margin: 1px; font-size: 11px; color: var(--text-secondary); }
.pass { color: var(--good); font-weight: 600; }
.fail { color: var(--critical); font-weight: 600; }
.na { color: var(--text-muted); }
tr.result-row { cursor: pointer; }
tr.result-row:hover td { background: var(--plane); }
.expand-cell { width: 18px; text-align: center; color: var(--text-muted); user-select: none; }
tr.detail-row td { background: var(--plane); padding: 14px 18px; border-bottom: 1px solid var(--grid); }
.detail { display: flex; flex-direction: column; gap: 10px; }
.detail-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); margin-bottom: 2px; }
.detail-text { white-space: pre-wrap; word-break: break-word; font-size: 12.5px; color: var(--text-primary); }
.detail-text.err { color: var(--critical); }
.detail-metric { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 8px; padding: 4px 0; border-bottom: 1px dashed var(--grid); font-size: 12.5px; }
.detail-metric-name { flex: 0 0 150px; color: var(--text-secondary); }
.detail-rationale { flex-basis: 100%; font-size: 11.5px; color: var(--text-muted); }
.detail-compare-row { border: 1px solid var(--border); border-left: 3px solid var(--baseline); border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; background: var(--surface); }
.detail-compare-row:last-child { margin-bottom: 0; }
.detail-compare-target { font-size: 11.5px; font-weight: 600; color: var(--text-primary); margin-bottom: 3px; }
.detail-compare-scores { font-size: 11px; color: var(--text-muted); margin-top: 5px; }
footer { padding: 8px 28px 28px; color: var(--text-muted); font-size: 12px; }

#chart-tooltip { position: fixed; z-index: 10; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 11.5px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); pointer-events: none; }
#chart-tooltip .tt-x { color: var(--text-muted); font-size: 10.5px; margin-bottom: 4px; }
#chart-tooltip .tt-row { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
#chart-tooltip .tt-val { font-weight: 600; font-variant-numeric: tabular-nums; color: var(--text-primary); }
#chart-tooltip .tt-name { color: var(--text-secondary); }
</style>
</head>
<body>
<header>
  <div>
    <h1>__TITLE__</h1>
    <div class="meta">generated __CREATED_AT__</div>
  </div>
  <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">
    <svg class="icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"/></svg>
    <svg class="icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" hidden="hidden"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
  </button>
</header>

<div class="kpi-row" id="kpis"></div>

<div class="controls">
  <select id="f-target"><option value="">All targets</option></select>
  <select id="f-tagkey"><option value="">All tags</option></select>
  <select id="f-tagvalue" disabled><option value="">any value</option></select>
  <select id="f-status"><option value="">Pass + Fail</option><option value="pass">Pass only</option><option value="fail">Fail only</option></select>
  <input type="text" id="f-search" placeholder="search input / output..." />
</div>

<section>
  <h2>Metrics</h2>
  <div class="metric-grid" id="metric-cards"></div>
</section>

<section>
  <h2>Samples</h2>
</section>
<div class="tablewrap">
<table class="results" id="results"><thead></thead><tbody></tbody></table>
</div>
<footer id="count"></footer>

<div id="chart-tooltip" hidden></div>
<script type="application/json" id="report-data">__DATA_JSON__</script>
<script>
const report = JSON.parse(document.getElementById("report-data").textContent);
const allResults = report.results;
const allTargets = report.targets;
const metricNames = report.metrics;

const targetColor = {};
allTargets.forEach((t, i) => { targetColor[t] = "var(--series-" + ((i % 8) + 1) + ")"; });

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s === null || s === undefined ? "" : String(s);
  return d.innerHTML;
}

function formatNum(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  const abs = Math.abs(x);
  if (abs >= 1e6) return (x / 1e6).toFixed(1).replace(/\\.0$/, "") + "M";
  if (abs >= 1e3) return (x / 1e3).toFixed(1).replace(/\\.0$/, "") + "K";
  if (Number.isInteger(x)) return String(x);
  return x.toFixed(3).replace(/0+$/, "").replace(/\\.$/, "");
}
function formatPct(x) { return x === null || x === undefined ? "-" : (x * 100).toFixed(0) + "%"; }

function statsOf(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  return { n, mean, median, min: sorted[0], max: sorted[n - 1] };
}

// ---- filtering ----

function uniqueTargets() { return [...new Set(allResults.map(r => r.target))].sort(); }
function allTagKeys() { return [...new Set(allResults.flatMap(r => Object.keys(r.tags || {})))].sort(); }
function tagValuesFor(key) { return [...new Set(allResults.map(r => (r.tags || {})[key]).filter(v => v !== undefined))].sort(); }

function populateSelect(sel, values, placeholder) {
  sel.innerHTML = '<option value="">' + placeholder + "</option>" + values.map(v => '<option value="' + esc(v) + '">' + esc(v) + "</option>").join("");
}

function currentFilters() {
  return {
    target: document.getElementById("f-target").value,
    tagKey: document.getElementById("f-tagkey").value,
    tagValue: document.getElementById("f-tagvalue").value,
    status: document.getElementById("f-status").value,
    search: document.getElementById("f-search").value.trim().toLowerCase(),
  };
}

function passed(result) {
  const flags = result.scores.map(s => s.passed).filter(p => p !== null && p !== undefined);
  return flags.length > 0 && flags.every(p => p === true);
}

function matches(result, f) {
  if (f.target && result.target !== f.target) return false;
  if (f.tagKey && f.tagValue && (result.tags || {})[f.tagKey] !== f.tagValue) return false;
  if (f.status === "pass" && !passed(result)) return false;
  if (f.status === "fail" && passed(result)) return false;
  if (f.search) {
    const hay = (result.input + " " + (result.output.text || "")).toLowerCase();
    if (!hay.includes(f.search)) return false;
  }
  return true;
}

// ---- KPI row ----

function renderKPIs(results) {
  const nResults = results.length;
  const nTargets = new Set(results.map(r => r.target)).size;
  const nErrors = results.filter(r => r.output.error).length;
  const errRate = nResults ? nErrors / nResults : 0;
  const passFlags = results.flatMap(r => r.scores.map(s => s.passed)).filter(p => p !== null && p !== undefined);
  const passRate = passFlags.length ? passFlags.filter(Boolean).length / passFlags.length : null;

  const tiles = [
    { label: "Samples evaluated", value: formatNum(nResults) },
    { label: "Targets compared", value: formatNum(nTargets) },
    { label: "Error rate", value: formatPct(errRate), status: errRate > 0 ? "critical" : "good" },
    {
      label: "Overall pass rate",
      value: passRate !== null ? formatPct(passRate) : "n/a",
      status: passRate === null ? null : passRate >= 0.8 ? "good" : passRate >= 0.5 ? "warning" : "critical",
    },
  ];
  document.getElementById("kpis").innerHTML = tiles.map(t =>
    '<div class="stat-tile"><div class="value' + (t.status ? " " + t.status : "") + '">' + esc(t.value) +
    '</div><div class="label">' + esc(t.label) + "</div></div>"
  ).join("");
}

// ---- ECDF chart ----

function ecdfSteps(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const steps = [];
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && sorted[j] === sorted[i]) j++;
    steps.push([sorted[i], j / n]);
    i = j;
  }
  return steps;
}

function ecdfValueAt(steps, x) {
  let y = 0;
  for (const [v, cy] of steps) {
    if (v <= x) y = cy; else break;
  }
  return y;
}

function pathFromSteps(steps, xmin, xmax, xScale, yScale) {
  let d = "M " + xScale(xmin) + " " + yScale(0);
  for (const [v, y] of steps) {
    d += " H " + xScale(v) + " V " + yScale(y);
  }
  d += " H " + xScale(xmax);
  return d;
}

function niceTicks(min, max, count) {
  if (min === max) return [min];
  const range = max - min;
  const rawStep = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = norm < 1.5 ? mag : norm < 3 ? 2 * mag : norm < 7 ? 5 * mag : 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) {
    ticks.push(Math.round(t / step) * step);
  }
  return ticks;
}

function renderEcdfChart(container, valuesByTarget, presentTargets) {
  const targets = presentTargets.filter(t => valuesByTarget[t] && valuesByTarget[t].length);
  if (!targets.length) { container.innerHTML = '<p class="empty">no numeric data for the current filter</p>'; return; }

  const allValues = targets.flatMap(t => valuesByTarget[t]);
  const dataMin = Math.min(...allValues);
  const dataMax = Math.max(...allValues);
  let xmin = dataMin, xmax = dataMax;
  if (xmin === xmax) { xmin -= 1; xmax += 1; } else { const pad = (xmax - xmin) * 0.04; xmin -= pad; xmax += pad; }

  const W = 460, H = 220, M = { top: 10, right: 14, bottom: 26, left: 40 };
  const plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  const xScale = x => M.left + ((x - xmin) / (xmax - xmin)) * plotW;
  const yScale = y => M.top + (1 - y) * plotH;

  const series = targets.map(t => ({ target: t, color: targetColor[t], steps: ecdfSteps(valuesByTarget[t]) }));
  const xTicks = niceTicks(dataMin, dataMax, 4);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="ecdf-svg" role="img" aria-label="ECDF chart">';
  for (const yt of yTicks) {
    const y = yScale(yt);
    svg += '<line class="grid" x1="' + M.left + '" x2="' + (W - M.right) + '" y1="' + y + '" y2="' + y + '" />';
    svg += '<text class="axis-label" x="' + (M.left - 6) + '" y="' + (y + 3) + '" text-anchor="end">' + Math.round(yt * 100) + "%</text>";
  }
  for (const xt of xTicks) {
    svg += '<text class="axis-label" x="' + xScale(xt) + '" y="' + (H - M.bottom + 15) + '" text-anchor="middle">' + esc(formatNum(xt)) + "</text>";
  }
  svg += '<line class="baseline" x1="' + M.left + '" x2="' + (W - M.right) + '" y1="' + (H - M.bottom) + '" y2="' + (H - M.bottom) + '" />';
  for (const s of series) {
    svg += '<path d="' + pathFromSteps(s.steps, xmin, xmax, xScale, yScale) + '" fill="none" stroke="' + s.color + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />';
  }
  svg += '<rect class="hover-rect" x="' + M.left + '" y="' + M.top + '" width="' + plotW + '" height="' + plotH + '" fill="transparent" />';
  svg += '<line class="crosshair" x1="0" x2="0" y1="' + M.top + '" y2="' + (H - M.bottom) + '" hidden="hidden" />';
  svg += "</svg>";
  container.innerHTML = svg;

  if (targets.length > 1) {
    const legend = document.createElement("div");
    legend.className = "legend";
    for (const s of series) {
      const item = document.createElement("span");
      item.className = "legend-item";
      const key = document.createElement("span");
      key.className = "legend-key";
      key.style.background = s.color;
      const label = document.createElement("span");
      label.textContent = s.target;
      item.appendChild(key);
      item.appendChild(label);
      legend.appendChild(item);
    }
    container.appendChild(legend);
  }

  const svgEl = container.querySelector(".ecdf-svg");
  const hoverRect = svgEl.querySelector(".hover-rect");
  const crosshair = svgEl.querySelector(".crosshair");
  const tooltip = document.getElementById("chart-tooltip");

  hoverRect.addEventListener("pointermove", (evt) => {
    const rect = svgEl.getBoundingClientRect();
    const px = ((evt.clientX - rect.left) / rect.width) * W;
    const xVal = xmin + ((px - M.left) / plotW) * (xmax - xmin);
    crosshair.setAttribute("x1", px);
    crosshair.setAttribute("x2", px);
    crosshair.hidden = false;

    tooltip.innerHTML = '<div class="tt-x">' + esc(formatNum(xVal)) + "</div>" + series.map(s => {
      const y = ecdfValueAt(s.steps, xVal);
      return '<div class="tt-row"><span class="tt-key" style="display:inline-block;width:10px;height:2px;background:' + s.color +
        ';border-radius:1px"></span><span class="tt-val">' + esc(formatPct(y)) + '</span><span class="tt-name">' + esc(s.target) + "</span></div>";
    }).join("");
    tooltip.style.left = (evt.clientX + 14) + "px";
    tooltip.style.top = (evt.clientY + 14) + "px";
    tooltip.hidden = false;
  });
  hoverRect.addEventListener("pointerleave", () => { crosshair.hidden = true; tooltip.hidden = true; });
}

function renderMeter(container, passByTarget, presentTargets) {
  const targets = presentTargets.filter(t => passByTarget[t] && passByTarget[t].length);
  if (!targets.length) { container.innerHTML = ""; return; }
  container.innerHTML = targets.map(t => {
    const flags = passByTarget[t];
    const rate = flags.filter(Boolean).length / flags.length;
    const color = rate >= 0.8 ? "var(--good)" : rate >= 0.5 ? "var(--warning)" : "var(--critical)";
    return '<div class="meter-row"><span class="name" title="' + esc(t) + '">' + esc(t) + '</span>' +
      '<span class="meter-track"><span class="meter-fill" style="width:' + (rate * 100) + "%;background:" + color + '"></span></span>' +
      '<span class="pct">' + formatPct(rate) + "</span></div>";
  }).join("");
}

// ---- metric cards ----

function renderMetricCards(results) {
  const container = document.getElementById("metric-cards");
  container.innerHTML = "";

  for (const metric of metricNames) {
    const valuesByTarget = {};
    const passByTarget = {};
    let category = "";
    for (const r of results) {
      const s = r.scores.find(sc => sc.metric === metric);
      if (!s) continue;
      if (s.value !== null && s.value !== undefined) (valuesByTarget[r.target] = valuesByTarget[r.target] || []).push(s.value);
      if (s.passed !== null && s.passed !== undefined) (passByTarget[r.target] = passByTarget[r.target] || []).push(s.passed);
    }
    const presentTargets = allTargets.filter(t => (valuesByTarget[t] && valuesByTarget[t].length) || (passByTarget[t] && passByTarget[t].length));
    if (!presentTargets.length) continue;

    const hasNumeric = presentTargets.some(t => valuesByTarget[t] && valuesByTarget[t].length);
    const hasPass = presentTargets.some(t => passByTarget[t] && passByTarget[t].length);

    const card = document.createElement("div");
    card.className = "metric-card";

    const header = document.createElement("div");
    header.className = "metric-card-header";
    const h3 = document.createElement("h3");
    h3.textContent = metric;
    header.appendChild(h3);
    card.appendChild(header);

    const table = document.createElement("table");
    table.className = "stats-table";
    let theadHtml = "<thead><tr><th>target</th><th>n</th>";
    if (hasNumeric) theadHtml += "<th>mean</th><th>median</th><th>min</th><th>max</th>";
    if (hasPass) theadHtml += "<th>pass rate</th>";
    theadHtml += "</tr></thead>";
    table.innerHTML = theadHtml;

    const tbody = document.createElement("tbody");
    for (const t of presentTargets) {
      const vals = valuesByTarget[t] || [];
      const flags = passByTarget[t] || [];
      const st = vals.length ? statsOf(vals) : null;
      const rate = flags.length ? flags.filter(Boolean).length / flags.length : null;
      const tr = document.createElement("tr");
      let rowHtml = '<td><span class="legend-key" style="background:' + targetColor[t] + '"></span>' + esc(t) + "</td><td>" + (st ? st.n : flags.length) + "</td>";
      if (hasNumeric) {
        rowHtml += st
          ? "<td>" + esc(formatNum(st.mean)) + "</td><td>" + esc(formatNum(st.median)) + "</td><td>" + esc(formatNum(st.min)) + "</td><td>" + esc(formatNum(st.max)) + "</td>"
          : "<td>-</td><td>-</td><td>-</td><td>-</td>";
      }
      if (hasPass) rowHtml += "<td>" + (rate !== null ? esc(formatPct(rate)) : "-") + "</td>";
      tr.innerHTML = rowHtml;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    card.appendChild(table);

    if (hasNumeric) {
      const chartWrap = document.createElement("div");
      chartWrap.className = "chart-wrap";
      card.appendChild(chartWrap);
      renderEcdfChart(chartWrap, valuesByTarget, presentTargets);
    } else if (hasPass) {
      const meterWrap = document.createElement("div");
      card.appendChild(meterWrap);
      renderMeter(meterWrap, passByTarget, presentTargets);
    }

    container.appendChild(card);
  }
}

// ---- sample table ----

function scoreCell(result, metric) {
  const s = (result.scores || []).find(sc => sc.metric === metric);
  if (!s) return '<span class="na">-</span>';
  const cls = s.passed === true ? "pass" : (s.passed === false ? "fail" : "");
  const title = s.rationale ? esc(s.rationale) : "";
  return '<span class="' + cls + '" title="' + title + '">' + esc(scoreValueLabel(s)) + "</span>";
}

function scoreValueLabel(s) {
  if (s.value !== null && s.value !== undefined) return formatNum(s.value);
  if (s.label) return s.label;
  if (s.passed !== null && s.passed !== undefined) return s.passed ? "pass" : "fail";
  return "-";
}

function detailPanel(r) {
  const blocks = [];
  blocks.push('<div class="detail-block"><div class="detail-label">input</div><div class="detail-text">' + esc(r.input) + "</div></div>");
  if (r.reference) blocks.push('<div class="detail-block"><div class="detail-label">expected output (reference)</div><div class="detail-text">' + esc(r.reference) + "</div></div>");
  blocks.push('<div class="detail-block"><div class="detail-label">' + esc(r.target) + ' output</div><div class="detail-text">' + esc(r.output.text) + "</div></div>");
  if (r.output.error) blocks.push('<div class="detail-block"><div class="detail-label">error</div><div class="detail-text err">' + esc(r.output.error) + "</div></div>");

  const metricRows = r.scores.map(s => {
    const cls = s.passed === true ? "pass" : (s.passed === false ? "fail" : "");
    return '<div class="detail-metric"><span class="detail-metric-name">' + esc(s.metric) + '</span><span class="' + cls + '">' + esc(scoreValueLabel(s)) + "</span>" +
      (s.rationale ? '<div class="detail-rationale">' + esc(s.rationale) + "</div>" : "") + "</div>";
  }).join("");
  if (metricRows) blocks.push('<div class="detail-block"><div class="detail-label">scores — why</div>' + metricRows + "</div>");

  const siblings = allResults.filter(x => x.sample_id === r.sample_id && x.target !== r.target);
  if (siblings.length) {
    const rows = siblings.map(s => {
      const scoreLine = s.scores.map(sc => esc(sc.metric) + "=" + esc(scoreValueLabel(sc))).join("  ·  ");
      return '<div class="detail-compare-row" style="border-left-color:' + targetColor[s.target] + '">' +
        '<div class="detail-compare-target">' + esc(s.target) + "</div>" +
        '<div class="detail-text">' + esc(s.output.text) + "</div>" +
        (scoreLine ? '<div class="detail-compare-scores">' + scoreLine + "</div>" : "") +
        "</div>";
    }).join("");
    blocks.push('<div class="detail-block"><div class="detail-label">same input, other targets</div>' + rows + "</div>");
  }

  return '<div class="detail">' + blocks.join("") + "</div>";
}

function renderTable(filtered) {
  const thead = document.querySelector("#results thead");
  const tbody = document.querySelector("#results tbody");
  const colCount = 7 + metricNames.length;
  thead.innerHTML = "<tr><th></th><th>id</th><th>target</th><th>tags</th><th>input</th><th>output</th>" +
    metricNames.map(m => "<th>" + esc(m) + "</th>").join("") + "<th>error</th></tr>";
  tbody.innerHTML = filtered.map(r => '<tr class="result-row">' +
    '<td class="expand-cell">&#9656;</td>' +
    "<td>" + esc(r.sample_id) + "</td>" +
    "<td>" + esc(r.target) + "</td>" +
    "<td>" + Object.entries(r.tags || {}).map(([k, v]) => '<span class="tag">' + esc(k) + "=" + esc(v) + "</span>").join("") + "</td>" +
    '<td class="truncate">' + esc(r.input) + "</td>" +
    '<td class="truncate">' + esc(r.output.text) + "</td>" +
    metricNames.map(m => "<td>" + scoreCell(r, m) + "</td>").join("") +
    "<td>" + esc(r.output.error || "") + "</td>" +
    "</tr>" +
    '<tr class="detail-row" hidden><td colspan="' + colCount + '">' + detailPanel(r) + "</td></tr>"
  ).join("");
  document.getElementById("count").textContent = filtered.length + " / " + allResults.length + " rows shown";
}

// ---- wiring ----

function applyFilters() {
  const f = currentFilters();
  const filtered = allResults.filter(r => matches(r, f));
  renderKPIs(filtered);
  renderMetricCards(filtered);
  renderTable(filtered);
}

function syncThemeIcon() {
  const isDark = document.documentElement.dataset.theme === "dark";
  document.querySelector(".icon-moon").hidden = isDark;
  document.querySelector(".icon-sun").hidden = !isDark;
}

function initTheme() {
  const btn = document.getElementById("theme-toggle");
  let saved = null;
  try { saved = localStorage.getItem("genevals-theme"); } catch (e) { /* private/blocked storage */ }
  if (saved === "dark") document.documentElement.dataset.theme = "dark";
  syncThemeIcon();
  btn.addEventListener("click", () => {
    const isDark = document.documentElement.dataset.theme === "dark";
    if (isDark) delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = "dark";
    syncThemeIcon();
    try { localStorage.setItem("genevals-theme", isDark ? "light" : "dark"); } catch (e) { /* private/blocked storage */ }
  });
}

function init() {
  initTheme();
  populateSelect(document.getElementById("f-target"), uniqueTargets(), "All targets");
  populateSelect(document.getElementById("f-tagkey"), allTagKeys(), "All tags");
  document.getElementById("f-tagkey").addEventListener("change", (e) => {
    const tv = document.getElementById("f-tagvalue");
    if (e.target.value) {
      populateSelect(tv, tagValuesFor(e.target.value), "any value");
      tv.disabled = false;
    } else {
      tv.innerHTML = '<option value="">any value</option>';
      tv.disabled = true;
    }
    applyFilters();
  });
  for (const id of ["f-target", "f-tagvalue", "f-status"]) {
    document.getElementById(id).addEventListener("change", applyFilters);
  }
  document.getElementById("f-search").addEventListener("input", applyFilters);
  document.querySelector("#results tbody").addEventListener("click", (e) => {
    const row = e.target.closest("tr.result-row");
    if (!row) return;
    const detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains("detail-row")) return;
    detail.hidden = !detail.hidden;
    row.querySelector(".expand-cell").innerHTML = detail.hidden ? "&#9656;" : "&#9662;";
  });
  applyFilters();
}
init();
</script>
</body>
</html>
"""


def report_to_html(report: EvalReport, path: str | Path) -> None:
    data = report.model_dump(mode="json")
    data_json = json.dumps(data).replace("</script>", "<\\/script>")
    title = html_lib.escape(report.dataset_name or "genevals report")

    rendered = (
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__CREATED_AT__", html_lib.escape(str(report.created_at)))
        .replace("__DATA_JSON__", data_json)
    )
    Path(path).write_text(rendered, encoding="utf-8")
