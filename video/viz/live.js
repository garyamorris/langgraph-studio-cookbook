// live.js: runs a REAL LangGraph example from this page (POST /threads, POST /threads/{id}/runs/stream) and animates the events.
// All structure comes from GET /assistants/{id}/graph?xray=true; all data from the stream (tasks / checkpoints / custom / messages).
// What each scene runs and says lives in scenarios.js.
import { API, FONT, MONO, jget, jpost, loadAssistants, loadSchemas, buildGraph, createView, textW } from './lg.js';
import { SCENARIOS, BY_GRAPH } from './scenarios.js';

// How long things take on screen (ms at speed=1). The server finishes in milliseconds; the page paces it so humans can follow.
const PACE = { edge: 650, startDwell: 560, endDwell: 400, chip: 80, chipLater: 34, ckpt: 70, interruptWait: 2800, resumeShow: 1700, intro: 1500, hold: 2000, minRun: 1000 };
const LOOP_FACTOR = [1, 0.55, 0.34, 0.26, 0.22, 0.2];

// ============================================================================================================
const Q = new URLSearchParams(location.search);
const SCEN_KEY = Q.get('scenario') || BY_GRAPH[Q.get('graph')] || (Q.get('graph') ? 'generic' : 'ticket');
const SC = SCENARIOS[SCEN_KEY];
const GRAPH = Q.get('graph') || SC.graph || 'ticket_router';
const URL_SPEED = Math.max(0.2, parseFloat(Q.get('speed')) || 1);
const SPEED = URL_SPEED * (SC.speed || 1);
const CAPS_ON = Q.get('captions') !== '0';
const CONCEPT = Q.get('concept') === '1';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms / SPEED));
const sleepU = (ms) => new Promise((r) => setTimeout(r, ms / URL_SPEED)); // human-reading waits: not sped up by the scenario factor
const $ = (id) => document.getElementById(id);
const NSV = 'http://www.w3.org/2000/svg';

window.__events = [];
window.__captions = [];
window.__done = false;
window.__state = {};
const T0 = performance.now();
const now = () => (performance.now() - T0) / 1000;
const mark = (name, data) => { try { if (window.rec && window.rec.mark) window.rec.mark(name, data); } catch (e) { /* not recording */ } };

let G, V, A, runToken = 0;

// ---------------------------------------------------------------------------------------------------- page chrome
function buildChrome() {
  $('gid').textContent = GRAPH;
  document.title = `live ${GRAPH}`;
  // legend
  const lg = $('legend');
  lg.innerHTML = `<svg width="340" height="30" viewBox="0 0 340 30" style="display:block"><defs><marker id="lgm" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto"><path d="M0.5,0.8 L9.5,5 L0.5,9.2 L2.8,5 z" fill="#9aa7c0"/></marker></defs>
    <path d="M2 15 L40 15" stroke="#9aa7c0" stroke-width="3" marker-end="url(#lgm)" fill="none"/><text x="50" y="16" fill="#9aa7c0" font-size="17" font-weight="600" dominant-baseline="central">edge</text>
    <path d="M112 15 L156 15" stroke="#9aa7c0" stroke-width="3" stroke-dasharray="9 7" marker-end="url(#lgm)" fill="none"/><text x="170" y="16" fill="#9aa7c0" font-size="17" font-weight="600" dominant-baseline="central">conditional edge</text></svg>`;
}
function setStatus(kind, text) {
  const s = $('status');
  s.className = `pill ${kind}`;
  s.textContent = text;
}
function buildStrip() {
  const strip = $('strip');
  strip.innerHTML = '';
  for (const c of SC.channels) {
    const d = document.createElement('div');
    d.className = 'card' + (c.tag ? ' tagged' : ''); d.dataset.k = c.k; d.style.flex = `${c.w} 1 0`;
    d.innerHTML = `<div class="ck">${c.label || c.k}</div><div class="cv dim">—</div>${c.tag ? `<div class="ct">${c.tag}</div>` : ''}`;
    strip.appendChild(d);
  }
}
const LIVE = { count: {}, total: {}, base: {}, custom: {}, pending: null };
function renderCards(values, flash = true) {
  for (const c of SC.channels) {
    const card = document.querySelector(`.card[data-k="${CSS.escape(c.k)}"]`);
    if (!card) continue;
    let t = null;
    try { t = c.fmt(values || {}, LIVE); } catch (e) { t = null; }
    const cv = card.querySelector('.cv');
    const txt = t === null || t === undefined ? '—' : String(t);
    if (cv.textContent !== txt) {
      cv.textContent = txt;
      const avail = card.clientWidth - 32;
      let fs = card.classList.contains('tagged') ? 23 : 26;
      while (fs > 17 && textW(txt, fs, 800) > avail) fs -= 1;
      cv.style.fontSize = fs + 'px';
      cv.classList.toggle('dim', t === null || t === undefined);
      if (flash && t !== null) { card.classList.remove('flash'); void card.offsetWidth; card.classList.add('flash'); }
    }
  }
}
// ---- caption strip: a caption stays at least `min` seconds; newer ones replace it as soon as that has passed; soft ones are dropped when busy
const Cap = {
  queue: [], showing: null, shownAt: 0, fired: new Set(), timer: null,
  fire(at, vars = {}) {
    if (!CAPS_ON) return;
    for (const d of SC.captions.filter((c) => c.at === at)) {
      if (d.once !== false && this.fired.has(at)) continue;
      this.fired.add(at);
      this.queue.push({ ...d, vars });
    }
    this.pump();
  },
  fmt(d) {
    const v = { n: '', ckpts: STATE.ckpts, thread: STATE.thread || '', ...d.vars };
    return d.text.replace(/\{(\w+)\}/g, (m, k) => (v[k] !== undefined ? v[k] : m));
  },
  pump() {
    if (this.timer) return;
    const tick = () => {
      this.timer = null;
      const t = performance.now();
      if (this.queue.length > 1) { const hard = this.queue.filter((c) => !c.soft); if (hard.length && hard.length < this.queue.length) this.queue = hard; }
      if (this.queue.length > 2) this.queue.splice(0, this.queue.length - 2);
      const min = this.showing ? (this.showing.soft ? 1.4 : (this.showing.min || 2.6)) * 1000 / URL_SPEED : 0;
      if (this.queue.length && (!this.showing || t - this.shownAt >= min)) this.show(this.queue.shift());
      if (this.queue.length) this.timer = setTimeout(tick, 100);
    };
    tick();
  },
  show(d) {
    const text = this.fmt(d);
    this.showing = d;
    this.shownAt = performance.now();
    const el = $('cap');
    el.classList.remove('in'); void el.offsetWidth;
    el.querySelector('.ctext').innerHTML = text.replace(/\{\{(.+?)\}\}/g, '<b>$1</b>');
    el.classList.add('in');
    const rec = { t: +now().toFixed(2), at: d.at, text: text.replace(/\{\{|\}\}/g, '') };
    window.__captions.push(rec);
    mark('caption', rec);
  },
};

// ---- event tape: the raw stream, as a front-end would receive it (used by the streaming scene instead of captions)
const Tape = {
  rows: [], tok: '', on: false,
  init() {
    this.rows = []; this.tok = ''; this.on = !!SC.tape;
    const el = $('tape');
    if (!el) return;
    el.style.display = this.on ? 'flex' : 'none';
    $('cap').style.display = this.on ? 'none' : '';
    this.render();
  },
  push(label, text) { if (!this.on) return; this.rows.push({ label, text }); if (this.rows.length > 3) this.rows.shift(); this.render(); },
  token(t) { if (!this.on || !t) return; this.tok = (this.tok + t).replace(/\s+/g, ' '); this.render(); },
  render() {
    const el = $('tape');
    if (!el || !this.on) return;
    const rows = this.rows.slice(this.tok ? -2 : -3);
    const row = (label, text, cls = '') => `<div class="tr ${cls}"><span class="tl">${label}</span><span class="tx">${String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;')}</span></div>`;
    el.innerHTML = rows.map((r) => row(r.label, r.text)).join('') + (this.tok ? row('messages', this.tok.slice(-92), 'tok') : '');
  },
};

// ---- checkpoint timeline
const TL = {
  ticks: [], W: 1232, x0: 128, x1: 850,
  init() {
    const s = $('tl');
    s.innerHTML = '';
    this.svg = s;
    const mk = (tag, a, p = s) => { const e = document.createElementNS(NSV, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); p.appendChild(e); return e; };
    this.mk = mk;
    mk('line', { x1: this.x0 - 8, x2: this.x1, y1: 22, y2: 22, stroke: '#2b364f', 'stroke-width': 2 });
    const t = mk('text', { x: 0, y: 22, fill: '#9aa7c0', 'font-size': 17, 'font-weight': 600, 'dominant-baseline': 'central' }); t.textContent = 'checkpoints';
    this.g = mk('g', {});
    this.labels = mk('g', {});
    this.ticker = [mk('text', { x: 876, y: 15, fill: '#9aa7c0', 'font-size': 17, 'dominant-baseline': 'central', style: `font-family:${MONO}` }), mk('text', { x: 876, y: 38, fill: '#e8eefc', 'font-size': 17, 'dominant-baseline': 'central', style: `font-family:${MONO}` })];
    this.ticks = []; this.lines = ['', ''];
  },
  log(text) { this.lines = [this.lines[1], text]; this.ticker[0].textContent = this.lines[0]; this.ticker[1].textContent = this.lines[1]; },
  add(kind, info = {}) {
    this.ticks.push({ kind, info });
    this.layout();
  },
  layout() {
    const n = this.ticks.length;
    const gaps = this.ticks.filter((t) => t.kind === 'pause' || t.kind === 'resume').length * 34;
    const step = Math.min(15, (this.x1 - this.x0 - gaps) / Math.max(1, n));
    this.g.innerHTML = ''; this.labels.innerHTML = '';
    let x = this.x0;
    this.ticks.forEach((t, i) => {
      if (t.kind === 'resume') x += 34;
      const last = i === n - 1;
      const col = t.kind === 'pause' ? '#ffb938' : last ? '#4aa8ff' : t.kind === 'resume' ? '#f4c95d' : '#3f78b3';
      const h = t.kind === 'pause' || t.kind === 'resume' ? 34 : 24;
      this.mk('rect', { x: x - 2, y: 22 - h / 2, width: 4, height: h, rx: 2, fill: col, opacity: last ? 1 : 0.9 }, this.g);
      if (t.kind === 'pause' || t.kind === 'resume') {
        const tx = this.mk('text', { x: x, y: 47, fill: col, 'font-size': 16, 'font-weight': 700, 'dominant-baseline': 'central', 'text-anchor': 'middle' }, this.labels);
        tx.textContent = t.kind === 'pause' ? 'interrupt' : 'resume';
      }
      x += step;
      if (t.kind === 'pause') x += 34;
    });
  },
};

// ---------------------------------------------------------------------------------------------------- run state
const STATE = { thread: null, ckpts: 0, values: {}, runs: {}, lastDone: ['__start__'], seenEdge: new Set(), interrupted: null, ended: false, runNo: 1 };
function nodeRec(id) { return (STATE.runs[id] = STATE.runs[id] || { active: 0, done: 0, total: 0, runs: 0, startedAt: 0, chipOf: new Map(), fanId: null }); }

function nodeIdFor(ns, name) {
  if (ns.length && G.groups.has(ns[0].node)) return `${ns[0].node}:${name}`;
  return name;
}
function findEdge(dstId, groupName) {
  const cand = G.edges.filter((e) => (groupName ? e.toGroup === groupName : e.dst === dstId && !e.toGroup) && e.d);
  // prefer an edge whose source finished most recently
  for (const s of STATE.lastDone) {
    const e = cand.find((x) => x.src === s);
    if (e) return e;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------------- SSE
async function* sseStream(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json', accept: 'text/event-stream' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`stream ${res.status}`);
  const rd = res.body.getReader(); const dec = new TextDecoder();
  let buf = '', ev = null, data = [];
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.search(/\r\n|\n|\r/)) >= 0) {
      const m = /^(\r\n|\n|\r)/.exec(buf.slice(i));
      const line = buf.slice(0, i); buf = buf.slice(i + m[0].length);
      if (line === '') { if (ev !== null || data.length) yield { event: ev || 'message', data: data.join('\n') }; ev = null; data = []; }
      else if (line.startsWith('event:')) ev = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
  }
}
function splitName(name) {
  const i = name.indexOf('|');
  if (i < 0) return { mode: name, ns: [] };
  return { mode: name.slice(0, i), ns: name.slice(i + 1).split('|').filter(Boolean).map((s) => { const j = s.indexOf(':'); return j < 0 ? { node: s, taskId: '' } : { node: s.slice(0, j), taskId: s.slice(j + 1) }; }) };
}

// ---------------------------------------------------------------------------------------------------- playback queue
class Queue {
  constructor() { this.items = []; this.waiter = null; this.closed = false; }
  push(it) { this.items.push(it); if (this.waiter) { const w = this.waiter; this.waiter = null; w(); } }
  close() { this.closed = true; if (this.waiter) { const w = this.waiter; this.waiter = null; w(); } }
  async next() {
    while (!this.items.length) { if (this.closed) return null; await new Promise((r) => { this.waiter = r; }); }
    return this.items.shift();
  }
}

/** Reads one HTTP stream, turns frames into queue items (coalescing a burst of identical task starts into one 'start' item). */
async function readRun(tid, body, q, token) {
  const interrupts = [];
  let pend = null;
  const flush = () => { if (pend) { q.push({ type: 'start', ...pend }); pend = null; } };
  const stream = sseStream(`${API}/threads/${tid}/runs/stream`, { assistant_id: A.id, stream_mode: ['values', 'updates', 'tasks', 'checkpoints', 'custom', 'messages-tuple'], stream_subgraphs: true, stream_resumable: true, on_disconnect: 'continue', multitask_strategy: 'enqueue', ...body });
  for await (const ev of stream) {
    if (token !== runToken) return interrupts;
    const { mode, ns } = splitName(ev.event);
    let d = null;
    try { d = ev.data ? JSON.parse(ev.data) : null; } catch (e) { d = ev.data; }
    window.__events.push({ mode, ns: ns.map((n) => n.node).join('|'), name: d && d.name, t: +now().toFixed(3) });
    if (mode === 'tasks' && d && d.name && !d.name.startsWith('__')) {
      const isEnd = 'result' in d || ('error' in d && !('input' in d));
      if (!isEnd) {
        const key = ns.map((n) => n.node).join('|') + '/' + d.name;
        if (pend && pend.key === key) { pend.n++; pend.taskIds.push(d.id); pend.inputs.push(d.input); }
        else { flush(); pend = { key, node: d.name, ns, n: 1, taskIds: [d.id], inputs: [d.input] }; }
      } else {
        flush();
        const its = Array.isArray(d.interrupts) ? d.interrupts : [];
        for (const it of its) interrupts.push(it);
        q.push({ type: 'end', node: d.name, ns, taskId: d.id, result: d.result, error: d.error, interrupts: its });
      }
      continue;
    }
    if (mode === 'messages' && Array.isArray(d)) { q.push({ type: 'token', text: (d[0] && typeof d[0].content === 'string' ? d[0].content : ''), node: d[1] && d[1].langgraph_node }); continue; }
    if (ns.length && mode !== 'custom') continue; // sub-graph checkpoints / updates / values: not drawn, must not split a burst of task starts
    if (mode === 'updates' || mode === 'values' || mode === 'metadata') continue;
    flush();
    if (mode === 'checkpoints' && ns.length === 0 && d) q.push({ type: 'ckpt', id: d.config && d.config.configurable && d.config.configurable.checkpoint_id, step: d.metadata && d.metadata.step, source: d.metadata && d.metadata.source, next: d.next || [], values: d.values || {} });
    else if (mode === 'custom' && d) q.push({ type: 'custom', kind: d.kind || d.type || d.event, agent: d.agent, data: d.data, raw: d, ns });
    else if (mode === 'error') q.push({ type: 'error', message: typeof d === 'string' ? d : (d && (d.message || d.error)) || 'error' });
  }
  flush();
  return interrupts;
}

// ---------------------------------------------------------------------------------------------------- the player (consumer)
const loopFactor = (runs) => LOOP_FACTOR[Math.min(runs, LOOP_FACTOR.length - 1)];

async function playStart(it) {
  const inner = it.ns.length > 0;
  const isGroupTask = !inner && G.groups.has(it.node);
  const gname = isGroupTask ? it.node : null;
  const nid = isGroupTask ? null : nodeIdFor(it.ns, it.node);
  const rec = nodeRec(isGroupTask ? 'g:' + gname : nid);
  const resuming = !isGroupTask && STATE.interrupted === nid;
  const lf = loopFactor(rec.runs);
  // edge into the node
  let edge = null;
  if (!resuming) {
    if (isGroupTask) edge = findEdge(null, gname);
    else if (inner) edge = findEdge(nid, null) && STATE.lastDone.includes(findEdge(nid, null).src) ? findEdge(nid, null) : null;
    else edge = findEdge(nid, null);
  }
  const first = !rec.runs;
  if (rec.runs && rec.done >= rec.total) { rec.active = 0; rec.done = 0; rec.total = 0; rec.chipOf.clear(); }
  rec.runs += resuming ? 0 : 1;
  if (resuming) { V.tag(nid, 'wait', null); V.tag(nid, 'resume', null); STATE.interrupted = null; }
  if (STATE.lastDone.length === 1 && STATE.lastDone[0] === '__start__') V.setNode('__start__', 'done');
  if (edge && !(rec.active > 0)) {
    V.setEdge(edge.id, 'active');
    if (edge.cond) Cap.fire(`cond:${edge.src}>${edge.dst}`, {});
    Cap.fire(`edge:${edge.src}>${edge.dst}`, {});
    await V.travel(edge.id, PACE.edge * lf * (inner ? 0.7 : 1));
    V.setEdge(edge.id, 'done');
    if (V.edges[edge.id]) { /* keep dashed look */ }
  }
  rec.active += it.n; rec.total += it.n; rec.startedAt = performance.now();
  const key = isGroupTask ? 'g:' + gname : nid;
  if (isGroupTask) V.setGroup(gname, 'running');
  else V.setNode(nid, 'running');
  // fan-out chips and badges
  const fanCfg = !isGroupTask && SC.fan && SC.fan[nid];
  if (isGroupTask) {
    if (SC.groupFan && SC.groupFan[gname]) { showGroupFan(gname, it.n, it.taskIds); Cap.fire(`fan:${gname}`, { n: it.n }); }
  } else if (it.n > 1) {
    V.badge(nid, 'n', `×${rec.total}`, { fill: '#f4c95d' });
    if (fanCfg) {
      const cols = Math.min(fanCfg.cols || it.n, it.n);
      const labels = fanCfg.rowKey ? Array.from({ length: Math.ceil(it.n / cols) }, (_, r) => (it.inputs[r * cols] || {})[fanCfg.rowKey] || '') : null;
      V.setFan(nid, it.n, cols, { rowLabels: labels });
      rec.fanId = nid;
      it.taskIds.forEach((id, i) => rec.chipOf.set(id, i));
      for (let i = 0; i < it.n; i++) V.chipState(nid, i, 'run');
      V.fanFoot(nid, `0 / ${it.n} tasks`);
      LIVE.total[nid] = it.n; LIVE.count[nid] = 0; LIVE.base[nid] = (STATE.values.reactions || []).length;
      renderCards(STATE.values, false);
      Cap.fire(`fan:${nid}`, { n: it.n });
    }
  }
  if (rec.runs >= 2 && !isGroupTask && !inner) V.badge(nid, 'loop', `↻ ${rec.runs}`, { slot: 'tl', fill: '#2a4a73', color: '#cfe6ff', size: 17, h: 26 });
  if (inner && it.n > 1) V.badge(nid, 'n', `×${it.n}`, { fill: '#f4c95d' });
  if (!isGroupTask) {
    Cap.fire(`node:${nid}`, {});
    if (rec.runs >= 2) Cap.fire(`loop:${nid}`, {});
  }
  STATE.seenEdge.add(key);
  await sleep(PACE.startDwell * lf);
}

function showGroupFan(gname, n, taskIds) {
  const gb = G.groups.get(gname).box;
  const id = '__grp_' + gname;
  V.clearFan(id);
  // three chips under the group's left part
  const cs = V.G.opt.chip, cg = V.G.opt.chipGap;
  const w = n * (cs + cg) - cg;
  const gEl = document.createElementNS(NSV, 'g');
  gEl.setAttribute('transform', `translate(${gb.x + 22},${gb.y + gb.h + 14})`);
  V.gOver.appendChild(gEl);
  const chips = [];
  for (let i = 0; i < n; i++) { const r = document.createElementNS(NSV, 'rect'); r.setAttribute('class', 'chip run'); r.setAttribute('x', i * (cs + cg)); r.setAttribute('y', 0); r.setAttribute('width', cs); r.setAttribute('height', cs); r.setAttribute('rx', 5); gEl.appendChild(r); chips.push(r); }
  const t = document.createElementNS(NSV, 'text');
  t.setAttribute('x', w + 12); t.setAttribute('y', cs / 2 + 1); t.setAttribute('fill', '#9aa7c0'); t.setAttribute('font-size', 18); t.setAttribute('font-weight', 600); t.setAttribute('dominant-baseline', 'central'); t.setAttribute('style', `font-family:${MONO}`);
  t.textContent = `0 / ${n} quills`;
  gEl.appendChild(t);
  V.fans[id] = { g: gEl, chips, foot: t, n, cols: n, done: 0 };
  const rec = nodeRec('g:' + gname);
  rec.fanId = id;
  taskIds.forEach((tid, i) => rec.chipOf.set(tid, i));
  rec.groupName = gname;
  LIVE.total[gname] = n; LIVE.count[gname] = 0;
  renderCards(STATE.values, false);
}

async function playEnd(it) {
  const inner = it.ns.length > 0;
  const isGroupTask = !inner && G.groups.has(it.node);
  const nid = isGroupTask ? null : nodeIdFor(it.ns, it.node);
  const key = isGroupTask ? 'g:' + it.node : nid;
  const rec = nodeRec(key);
  const lf = loopFactor(Math.max(0, rec.runs - 1));
  // honest minimum on-screen "running" time (the server may finish a node in 2 ms; humans need a second)
  const el = performance.now() - rec.startedAt;
  const needed = (PACE.minRun * lf * (rec.fanId ? 0.25 : inner ? 0.5 : 1)) / SPEED;
  if (el < needed) await new Promise((r) => setTimeout(r, needed - el));
  rec.active = Math.max(0, rec.active - 1); rec.done += 1;
  const fanId = rec.fanId;
  if (fanId && V.fans[fanId]) {
    const idx = rec.chipOf.get(it.taskId);
    if (idx !== undefined) V.chipState(fanId, idx, 'done');
    const f = V.fans[fanId]; f.done++;
    const lbl = isGroupTask ? 'quills' : 'tasks';
    V.fanFoot(fanId, `${f.done} / ${f.n} ${lbl}`);
    const cnt = isGroupTask ? it.node : nid;
    LIVE.count[cnt] = f.done;
    renderCards(STATE.values, false);
    await sleep(rec.runs <= 1 ? PACE.chip : PACE.chipLater);
  }
  if (!(it.interrupts && it.interrupts.length)) {
    const mine = isGroupTask ? G.groups.get(it.node).members : [nid];
    STATE.lastDone = [...mine, ...STATE.lastDone.filter((x) => !mine.includes(x))].slice(0, 12);
  }
  if (rec.active > 0) return;
  // the whole group of tasks of this node has finished
  const interrupted = it.interrupts && it.interrupts.length;
  if (isGroupTask) V.setGroup(it.node, 'done');
  else V.setNode(nid, interrupted ? 'interrupted' : 'done');
  if (interrupted) { STATE.interrupted = nid; return; }
  if (rec.fanId && rec.total > 1) {
    Cap.fire(`fandone:${isGroupTask ? it.node : nid}`, { n: rec.total });
    const out = G.edges.find((e) => (isGroupTask ? G.groups.get(it.node).members.includes(e.src) && !G.groups.get(it.node).members.includes(e.dst) : e.src === nid) && !e.back);
    const red = SC.channels.find((c) => c.tag && c.k === (SC.reducerOf && SC.reducerOf[isGroupTask ? it.node : nid]));
    if (out && red && V.edges[out.id]) { V.edgeLabel(out.id, 'reducer ' + red.tag.replace('reducer: ', '= ')); setTimeout(() => V.edgeLabel(out.id, null), 2600 / SPEED); }
    await sleep(PACE.endDwell * 2);
  } else await sleep(PACE.endDwell * lf);
}

function dynamicChannels(values) {
  const keys = Object.keys(values).filter((k) => values[k] !== null && values[k] !== undefined && k !== 'messages').slice(0, 5);
  if (!keys.length) return;
  SC.channels = keys.map((k) => ({ k, w: 1, fmt: (v) => { const x = v[k]; if (x === undefined) return null; if (Array.isArray(x)) return `${x.length} items`; if (x && typeof x === 'object') return `${Object.keys(x).length} keys`; return String(x).slice(0, 30); } }));
  buildStrip();
}
function ckptStep(it) {
  STATE.ckpts += 1;
  if (SC.dynamic && !SC.channels.length) dynamicChannels(it.values || {});
  STATE.values = { ...STATE.values, ...it.values };
  TL.add('ckpt');
  TL.log(`checkpoint #${STATE.ckpts} · next: ${(it.next || []).filter((x, i, a) => a.indexOf(x) === i).join(', ') || 'end'}`);
  // pending-write counters give way to the checkpointed value
  renderCards(STATE.values);
}

async function playPause(it) {
  const p = it.interrupts[0].value || {};
  const nid = STATE.interrupted;
  setStatus('amber', 'WAITING FOR A HUMAN');
  V.tag(nid, 'wait', 'waiting for a human', { icon: 'pause' });
  TL.add('pause');
  TL.log(`interrupt() · ${p.kind || 'value'}`);
  if (p.mishap) { LIVE.pending = p.mishap.title; renderCards(STATE.values); }
  Cap.fire('interrupt', {}); Cap.fire(`interrupt:${nid}`, {});
  await sleepU(PACE.interruptWait * (STATE.runNo > 1 ? 0.62 : 1));
  const val = SC.answer(p.kind, p);
  const body = { command: { resume: Object.fromEntries(it.interrupts.map((i) => [i.id, val])) } };
  let txt = JSON.stringify(val);
  if (txt.length > 52) txt = txt.slice(0, 50) + '…';
  V.tag(nid, 'wait', null);
  V.tag(nid, 'resume', `Command(resume=${txt})`, { icon: 'play', fill: '#10233d', color: '#f4c95d', stroke: '#f4c95d', mono: true, size: 18 });
  setStatus('blue', 'RESUMING');
  TL.add('resume');
  TL.log(`Command(resume=…) → same thread_id`);
  STATE.runNo += 1;
  Cap.fire('resume', {});
  await sleepU(PACE.resumeShow * (STATE.runNo > 2 ? 0.7 : 1));
  LIVE.pending = null;
  setStatus('blue', 'RUNNING');
  return body;
}

async function player(q, out) {
  for (;;) {
    const it = await q.next();
    if (!it) break;
    if (it.type === 'start') { Tape.push('tasks', `${it.ns.length ? it.ns.map((n) => n.node).join('/') + '/' : ''}${it.node}${it.n > 1 ? ' ×' + it.n : ''} started`); await playStart(it); }
    else if (it.type === 'end') await playEnd(it);
    else if (it.type === 'ckpt') { ckptStep(it); Tape.push('checkpoints', `#${STATE.ckpts} step ${it.step} · next: ${(it.next || []).join(', ') || 'end'}`); await sleep(PACE.ckpt); }
    else if (it.type === 'custom') {
      LIVE.custom[it.kind] = (LIVE.custom[it.kind] || 0) + 1;
      if (!['progress', 'note'].includes(it.kind) && !/^(audience|cast\.audition)/.test(it.kind)) TL.log(`custom › ${it.kind}`);
      if (/^memory\./.test(it.kind)) { renderCards(STATE.values, true); await sleep(80); }
      if (it.raw && it.raw.type === 'progress') Tape.push('custom', `${it.raw.stage} ${String(it.raw.pct).padStart(3)}%  ${it.raw.message}`);
      else if (it.raw && it.raw.type === 'partial') Tape.push('custom', `partial: ${it.raw.name}`);
      else if (it.raw) Tape.push('custom', JSON.stringify(it.raw).slice(0, 96));
    } else if (it.type === 'token') { if (!it.node || it.node === 'write_digest') Tape.token(it.text); }
    else if (it.type === 'pause') { const body = await playPause(it); it.deferred.resolve(body); }
    else if (it.type === 'error') { setStatus('red', 'ERROR'); TL.log('error: ' + it.message); }
    else if (it.type === 'finish') {
      // final edge to END
      const last = STATE.lastDone.find((s) => G.edges.some((e) => e.src === s && e.dst === '__end__'));
      const e = G.edges.find((x) => x.src === last && x.dst === '__end__');
      if (e) { V.setEdge(e.id, 'active'); await V.travel(e.id, PACE.edge * 0.8); V.setEdge(e.id, 'done'); }
      V.setNode('__end__', 'done');
      break;
    }
  }
}

// ---------------------------------------------------------------------------------------------------- run
/** The "Ready-made case" the scene submits: SC.case is a prefix of one label of the graph's own `scenario` dropdown enum. */
async function resolveInput() {
  if (!SC.case) return SC.input ? SC.input() : {};
  const sch = await loadSchemas(A.id);
  const enumv = ((((sch.input_schema || {}).properties || {}).scenario) || {}).enum || [];
  const label = enumv.find((l) => l.startsWith(SC.case)) || enumv.find((l) => l.includes(SC.case));
  if (!label) throw new Error(`no ready-made case starting with "${SC.case}" in ${GRAPH} (have ${enumv.length})`);
  window.__case = label;
  return { scenario: label };
}
async function runOnce() {
  const token = ++runToken;
  Object.assign(STATE, { thread: null, ckpts: 0, values: {}, runs: {}, lastDone: ['__start__'], seenEdge: new Set(), interrupted: null, ended: false, runNo: 1 });
  Object.assign(LIVE, { count: {}, total: {}, base: {}, custom: {}, pending: null });
  Cap.fired = new Set(); Cap.queue = []; Cap.showing = null;
  window.__events.length = 0; window.__captions.length = 0; window.__done = false;
  V.reset(); TL.init(); Tape.init(); buildStrip(); $('cap').querySelector('.ctext').innerHTML = ''; $('cap').classList.remove('in');
  setStatus('grey', 'STARTING');
  await sleep(PACE.intro);
  const thread = await jpost(`${API}/threads`, { metadata: { graph: GRAPH, source: 'viz' } });
  STATE.thread = thread.thread_id;
  $('tid').textContent = STATE.thread;
  setStatus('blue', 'RUNNING');
  TL.log(`POST /threads → thread created`);
  Cap.fire('start', {});
  mark('run-start', { thread: STATE.thread });
  const q = new Queue();
  const out = {};
  const consumer = player(q, out);
  let body = { input: await resolveInput() };
  try {
    for (let round = 0; round < 8; round++) {
      TL.log(round === 0 ? 'POST …/runs/stream (input)' : 'POST …/runs/stream (command: resume)');
      const interrupts = await readRun(STATE.thread, body, q, token);
      if (token !== runToken) return;
      const pending = [];
      const seen = new Set();
      for (const i of interrupts) if (!seen.has(i.id)) { seen.add(i.id); pending.push(i); }
      if (!pending.length) break;
      let resolve; const deferred = { promise: new Promise((r) => { resolve = r; }) }; deferred.resolve = resolve;
      q.push({ type: 'pause', interrupts: pending, deferred });
      body = await deferred.promise;
    }
    q.push({ type: 'finish' });
    q.close();
    await consumer;
    if (token !== runToken) return;
    setStatus('green', 'DONE');
    TL.log(`run finished · ${STATE.ckpts} checkpoints`);
    Cap.fire('end', {});
    await sleep(PACE.hold);
  } catch (e) {
    console.error(e);
    setStatus('red', 'ERROR');
    $('cap').querySelector('.ctext').textContent = String(e && e.message || e);
    $('cap').classList.add('in');
  }
  mark('run-end', { checkpoints: STATE.ckpts });
  window.__state = { thread: STATE.thread, checkpoints: STATE.ckpts, captions: window.__captions.length };
  window.__done = true;
}
window.__restart = () => { window.__done = false; return runOnce(); };

// ---------------------------------------------------------------------------------------------------- concept mode (scene H opening)
async function runConcept() {
  const C = SC.concept;
  if (!C) throw new Error(`scene "${SCEN_KEY}" has no concept callouts`);
  setStatus('grey', 'THE GRAPH');
  $('tid').textContent = 'not started';
  // real channel names + types from the API's state schema
  let props = {};
  try { const sch = await loadSchemas(A.id); props = (sch.state_schema && sch.state_schema.properties) || {}; } catch (e) { /* offline */ }
  const typeOf = (p) => !p ? '' : p.type === 'array' ? 'list' + (p.items && p.items.type === 'object' ? '[dict]' : p.items && p.items.type === 'string' ? '[str]' : '') : p.type === 'object' ? 'dict' : p.type === 'string' ? 'str' : p.type === 'integer' ? 'int' : p.type === 'number' ? 'float' : p.type || (p.anyOf ? p.anyOf.map((x) => x.type).filter((t) => t && t !== 'null').join('|') : '');
  const strip = $('strip');
  strip.innerHTML = '';
  for (const k of C.channels) {
    const d = document.createElement('div');
    d.className = 'card'; d.dataset.k = k; d.style.flex = '1 1 0';
    d.innerHTML = `<div class="ck">${k}</div><div class="cv dim">${typeOf(props[k]) || '—'}</div>`;
    strip.appendChild(d);
  }
  TL.init(); TL.log(`concept view · ${GRAPH}`);
  await sleep(700);
  const over = $('over');
  const gb = $('g').getBoundingClientRect();
  const P = (id, side = 'c') => { const n = G.nodes.get(id); return { x: gb.left + n.cx, y: gb.top + n.cy + (side === 'b' ? n.h / 2 : side === 't' ? -n.h / 2 : 0) }; };
  const E = (id) => { const e = V.edges[id]; const p = e.path.getPointAtLength(e.len * 0.5); return { x: gb.left + p.x, y: gb.top + p.y }; };
  const card = (k) => document.querySelector(`.card[data-k="${k}"]`).getBoundingClientRect();
  const sb = $('strip').getBoundingClientRect();
  const sv = document.createElementNS(NSV, 'svg');
  sv.setAttribute('width', 1280); sv.setAttribute('height', 720); sv.setAttribute('viewBox', '0 0 1280 720');
  over.appendChild(sv);
  const mkEl = (tag, attrs, parent) => { const e = document.createElementNS(NSV, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; };
  for (const c of C.callouts) {
    let target;
    if (c.node) { target = P(c.node, c.side || 'c'); V.setNode(c.node, 'running'); }
    else if (c.edge) { const m = E(c.edge); target = { x: m.x + (c.tdx || 0), y: m.y + (c.tdy || 0) }; for (const id of [c.edge, ...(c.also || [])]) if (V.edges[id]) V.setEdge(id, 'active'); }
    else if (c.card) { const r = card(c.card); target = { x: r.left + r.width / 2, y: (c.side === 'b' ? sb.bottom : sb.top) + (c.side === 'b' ? 4 : -4) }; $('strip').classList.add('hi'); }
    const g = mkEl('g', { class: 'co' }, sv);
    const w1 = textW(c.text, 32, 800), w2 = textW(c.sub, 19, 600);
    const w = Math.max(w1, w2) + 40, h = 74;
    const x = c.at.x - w / 2, y = c.at.y - h / 2;
    const sx = Math.max(x + 30, Math.min(x + w - 30, target.x)), sy = target.y > c.at.y ? y + h : y;
    mkEl('path', { d: `M${sx} ${sy} C${sx} ${(sy + target.y) / 2} ${target.x} ${(sy + target.y) / 2} ${target.x} ${target.y}`, stroke: '#f4c95d', 'stroke-width': 3.5, fill: 'none', 'stroke-linecap': 'round' }, g);
    mkEl('circle', { cx: target.x, cy: target.y, r: 7, fill: '#f4c95d' }, g);
    mkEl('rect', { x, y, width: w, height: h, rx: 14, fill: '#1b1606', stroke: '#f4c95d', 'stroke-width': 3 }, g);
    mkEl('text', { x: c.at.x, y: y + 29, 'text-anchor': 'middle', fill: '#f4c95d', 'font-size': 32, 'font-weight': 800, 'dominant-baseline': 'central' }, g).textContent = c.text;
    mkEl('text', { x: c.at.x, y: y + 57, 'text-anchor': 'middle', fill: '#e8dfc4', 'font-size': 19, 'font-weight': 600, 'dominant-baseline': 'central' }, g).textContent = c.sub;
    mark('callout', { text: c.text, sub: c.sub });
    window.__captions.push({ t: +now().toFixed(2), at: 'concept', text: `${c.text}: ${c.sub}` });
    await sleep(C.dwell || 1700);
  }
  await sleep(1300);
  window.__done = true;
}

// ---------------------------------------------------------------------------------------------------- boot
(async function boot() {
  buildChrome();
  try {
    const as = await loadAssistants();
    A = as[GRAPH];
    if (!A) throw new Error('no such graph: ' + GRAPH);
    const fan = Object.fromEntries(Object.entries(SC.fan || {}).map(([k, v]) => [k, { n: v.n || 3, cols: v.cols || 3 }]));
    // auto-fit: the first (largest) size whose layout does not overflow the 400 px stage
    const FITS = [{}, { nodeH: 48, font: 23, vgap: 24, padX: 18, minW: 80 }, { nodeH: 42, font: 21, vgap: 15, padX: 15, minW: 72 }, { nodeH: 36, font: 19, vgap: 10, padX: 12, minW: 64 }, { nodeH: 32, font: 17, vgap: 8, padX: 10, minW: 56 }];
    for (const f of FITS) {
      G = await buildGraph(GRAPH, as, { W: 1232, H: 400, fan, ...(SC.layout || {}), ...f });
      if (!G.size.overflow) break;
    }
    window.__fit = { overflow: G.size.overflow, rows: G.rowsSpec, nodes: G.nodes.size };
    V = createView($('g'), G, { groupSub: SC.groupSub });
    window.__G = G;
    buildStrip(); TL.init(); Tape.init();
    if (CONCEPT) { document.body.classList.add('concept'); await runConcept(); return; }
    await runOnce();
  } catch (e) {
    console.error(e);
    $('cap').querySelector('.ctext').textContent = 'Cannot reach the LangGraph server at ' + API + ' (' + (e && e.message) + ')';
    $('cap').classList.add('in');
    window.__error = String(e);
    window.__done = true;
  }
})();
