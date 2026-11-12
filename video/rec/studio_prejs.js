// --pre-js for the Studio clips (injected at document start, after director_lib.js).
// 1) Pane layout: react-resizable-panels persists the split in localStorage and the recorder starts with an empty store every
//    run, so seed it. Width of the graph pane (percent, max 67) comes from the URL hash, e.g. ...#panes=60 (default 66).
// 2) The offscreen recorder has no real mouse, so CSS :hover never fires. Studio reveals each checkpoint's toolbar
//    ("View state", "Re-run from here") with Tailwind `group-hover:visible`; fakeHover(el) marks that row so the SAME real buttons
//    become visible (the buttons and their click handlers are Studio's own). studioHover/studioUnhover also send the mouseout
//    that React needs to un-highlight the previous step (director_lib.hover only sends mouseover).
(function () {
  var m = /panes=(\d+)/.exec(location.hash || '');
  var left = m ? Math.max(33, Math.min(67, +m[1])) : 66;
  try { localStorage.setItem('react-resizable-panels:graph-layout', JSON.stringify({ 'left,right': { expandToSizes: {}, layout: [left, 100 - left] } })); } catch (e) {}
  var st = document.createElement('style');
  st.textContent = '[data-fh] .group-hover\\:visible{visibility:visible!important}[data-fh]{background-color:rgba(255,255,255,0.05)!important}';
  (document.head || document.documentElement).appendChild(st);
  var prev = null;
  function leave(el, to) {
    if (!el || !el.isConnected) return;
    ['pointerout', 'mouseout', 'pointerleave', 'mouseleave'].forEach(function (ty) {
      var bub = ty.indexOf('leave') < 0;
      try { el.dispatchEvent(new MouseEvent(ty, { bubbles: bub, cancelable: true, view: window, relatedTarget: to || document.body })); } catch (e) {}
    });
  }
  window.fakeHover = function (el) {
    Array.prototype.forEach.call(document.querySelectorAll('[data-fh]'), function (x) { x.removeAttribute('data-fh'); });
    var g = el && el.closest && el.closest('.group');
    if (g) g.setAttribute('data-fh', '1');
    return g;
  };
  window.studioUnhover = function () { leave(prev); prev = null; window.fakeHover(null); };
  window.studioHover = async function (target, o) {
    var el = typeof target === 'string' || target instanceof RegExp ? byText(target, o) : target;
    if (!el) { rec.log('studioHover: not found ' + String(target)); return null; }
    leave(prev, el); window.fakeHover(null);
    await hover(el, o);
    prev = el;
    window.fakeHover(el);
    return el;
  };
  // the step title <span> of a step in the thread pane (right of the graph pane), e.g. studioStep('triage')
  window.studioStep = function (name) {
    var els = byTextAll(name, { exact: true });
    var lim = innerWidth * 0.45;
    return els.filter(function (e) { return e.getBoundingClientRect().left > lim && !e.closest('.react-flow'); })[0] || null;
  };
  // the checkpoint divider rows ("5 minutes ago  [View state] [Re-run from here]"), top to bottom
  window.studioCheckpoints = function () {
    return Array.prototype.slice.call(document.querySelectorAll('.group')).filter(function (g) {
      return /ago/.test((g.textContent || '')) && g.querySelector('button') && g.getBoundingClientRect().height < 60 && !g.querySelector('.group');
    });
  };

  // ---- graph viewport control (react-flow): synthetic drag-pan and ctrl+wheel zoom, both animated with the fake cursor
  var ease3 = function (p) { return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; };
  window.studioVP = function () { var v = document.querySelector('.react-flow__viewport'); return v ? v.style.transform : null; };
  window.studioPan = async function (dx, dy, ms) {
    ms = ms || 900;
    var pane = document.querySelector('.react-flow__pane'); if (!pane) return false;
    var r = pane.getBoundingClientRect();
    var x0 = r.left + r.width * 0.5, y0 = r.top + r.height * 0.62;
    await cursor.move(x0, y0);
    var ev = function (type, x, y, extra) { return new MouseEvent(type, Object.assign({ bubbles: true, cancelable: true, composed: true, view: window, clientX: x, clientY: y, screenX: x, screenY: y, button: 0, buttons: 1 }, extra || {})); };
    pane.dispatchEvent(ev('mousedown', x0, y0));
    var t0 = performance.now(), x = x0, y = y0;
    for (;;) {
      var p = Math.min(1, (performance.now() - t0) / ms), e = ease3(p);
      x = x0 + dx * e; y = y0 + dy * e;
      window.dispatchEvent(ev('mousemove', x, y));
      await cursor.move(x, y, 1);
      if (p >= 1) break;
      await frame();
    }
    window.dispatchEvent(ev('mouseup', x, y, { buttons: 0 }));
    return true;
  };
  // zoom about (cx, cy) (client px; default pane centre) by `factor` over `ms`, as a series of ctrl+wheel ticks (react-flow / d3-zoom: k *= 2^(-deltaY*0.02) for ctrl+wheel)
  window.studioZoom = async function (factor, cx, cy, ms) {
    ms = ms || 800;
    var pane = document.querySelector('.react-flow__pane'); if (!pane) return false;
    var r = pane.getBoundingClientRect();
    cx = cx == null ? r.left + r.width / 2 : cx; cy = cy == null ? r.top + r.height / 2 : cy;
    var total = Math.log2(factor), done = 0, t0 = performance.now();
    for (;;) {
      var p = Math.min(1, (performance.now() - t0) / ms), target = total * ease3(p), step = target - done;
      if (Math.abs(step) > 1e-4) {
        pane.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, composed: true, view: window, clientX: cx, clientY: cy, deltaY: -step / 0.02, deltaMode: 0, ctrlKey: true }));
        done = target;
      }
      if (p >= 1) break;
      await frame();
    }
    return true;
  };

  // frame the named graph nodes: zoom (about a computed point, no pan needed) so their bounding box fills `fill` of the pane and sits centred.
  window.studioFocus = async function (names, o) {
    o = o || {};
    var pane = document.querySelector('.react-flow__pane'); if (!pane) return false;
    var els = Array.prototype.slice.call(document.querySelectorAll('.react-flow__node')).filter(function (n) { return names.indexOf((n.textContent || '').trim()) >= 0; });
    if (!els.length) return false;
    var L = 1e9, T = 1e9, Rr = -1e9, B = -1e9;
    els.forEach(function (n) { var r = n.getBoundingClientRect(); L = Math.min(L, r.left); T = Math.min(T, r.top); Rr = Math.max(Rr, r.right); B = Math.max(B, r.bottom); });
    var pr = pane.getBoundingClientRect(), fill = o.fill || 0.86;
    var f = Math.min(fill * pr.width / (Rr - L), fill * pr.height / (B - T), o.maxZoom || 2.2);
    var Cx = (L + Rr) / 2, Cy = (T + B) / 2, Mx = pr.left + pr.width / 2, My = pr.top + pr.height * (o.cy == null ? 0.5 : o.cy);
    if (Math.abs(f - 1) < 0.02) return true;
    var Px = (Mx - f * Cx) / (1 - f), Py = (My - f * Cy) / (1 - f);
    return window.studioZoom(f, Px, Py, o.ms || 900);
  };
})();
