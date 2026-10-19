/* Equilibrium Flow Maps: toy 2D animation, computed exactly.
   Data: 10 points on a circle. Noise x0 ~ N(0, I). Exponential clock: I_tau = e^{-tau} x0 + (1 - e^{-tau}) x1.
   The autonomous field minimizing the autonomous flow matching objective is b(y) = m(y) - y, where m(y) is the mean of the
   data points weighted by A_i(y) = integral over tau of N(y; (1 - e^{-tau}) a_i, e^{-2 tau} I) (computed on a tau grid).
   Left panel: the samples follow b with small Euler steps. Right panel: the exact flow map, K jumps with
   sigma_k = 1/(K-k+1), i.e. after k jumps a fraction k/K of the way in compactified time; each jump lands on the left path.
   API: window.EqFMAnim.mount(el, { compact: bool, captionEl: element }) */
(function () {
  "use strict";
  var COL = { accent: "#1f5f8b", data: "rgba(59,125,58,0.9)", arrow: "#8a9097", fg: "#1a1a1a", muted: "#666b70",
              panel: "#fbfcfd", trail: "rgba(31,95,139,0.22)", stage: "#f2f4f5" };
  var FONT = "'Source Sans 3','Helvetica Neue',Arial,sans-serif";
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function ease(x) { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); }
  function arrowHead(ctx, x0, y0, x1, y1, size) {
    var a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - size * Math.cos(a - 0.45), y1 - size * Math.sin(a - 0.45));
    ctx.lineTo(x1 - size * Math.cos(a + 0.45), y1 - size * Math.sin(a + 0.45));
    ctx.closePath(); ctx.fill();
  }

  // ---- exact toy field ----
  var DATA = [];
  for (var q0 = 0; q0 < 10; q0++) DATA.push([1.5 * Math.cos(2 * Math.PI * (q0 + 0.25) / 10), 1.5 * Math.sin(2 * Math.PI * (q0 + 0.25) / 10)]);
  var TG = []; for (var q = 0; q < 56; q++) TG.push(5.2 * q / 55);
  var LOGW = new Float64Array(DATA.length * TG.length);
  function field(x, y) {
    var M = -1e300, k = 0, i, j;
    for (i = 0; i < DATA.length; i++) for (j = 0; j < TG.length; j++, k++) {
      var s = Math.exp(-TG[j]), c = 1 - s, dx = x - c * DATA[i][0], dy = y - c * DATA[i][1];
      var l = -(dx * dx + dy * dy) / (2 * s * s) - 2 * Math.log(s);
      LOGW[k] = l; if (l > M) M = l;
    }
    var Z = 0, mx = 0, my = 0; k = 0;
    for (i = 0; i < DATA.length; i++) { var a = 0; for (j = 0; j < TG.length; j++, k++) a += Math.exp(LOGW[k] - M); Z += a; mx += a * DATA[i][0]; my += a * DATA[i][1]; }
    return [mx / Z - x, my / Z - y];
  }
  function rng(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  var TOY = null;
  function buildToy() {
    if (TOY) return TOY;
    var R = rng(7), NP = 64, DT = 0.04, NS = 175, paths = [], tries = 0;
    function gauss() { var u = 1 - R(), v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    while (paths.length < NP && tries < 400) {
      tries++;
      var x = gauss(), y = gauss();
      if (Math.abs(x) > 2.4 || Math.abs(y) > 2.4) continue;
      var P = new Float32Array(2 * (NS + 1)); P[0] = x; P[1] = y;
      for (var n = 1; n <= NS; n++) { var b = field(x, y); x += DT * b[0]; y += DT * b[1]; P[2 * n] = x; P[2 * n + 1] = y; }
      var best = 9; DATA.forEach(function (a) { best = Math.min(best, Math.hypot(a[0] - x, a[1] - y)); });
      if (best < 0.06) paths.push(P);   // keep samples that have settled by tau = 7 (a few start next to an unstable point)
    }
    TOY = { paths: paths, NS: NS, DT: DT };
    return TOY;
  }
  function stepAt(frac, toy) { if (frac >= 1) return toy.NS; return Math.min(toy.NS, Math.round(-Math.log(1 - frac) / toy.DT)); }

  function mount(el, opts) {
    opts = opts || {};
    var compact = !!opts.compact, capEl = opts.captionEl || null;
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var toy = buildToy();
    var canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Two panels with the same noise points and the same arrow field. Left: the points follow the arrows in many small steps and settle on ten green data points on a circle. Right: the same points reach the same data points in a few jumps.");
    canvas.style.display = "block"; canvas.style.width = "100%";
    var poster = el.querySelector("img"); if (poster) poster.style.display = "none";
    el.appendChild(canvas);
    var ctx = canvas.getContext("2d"), W, H, panels, views, quiv;
    var T = { intro: 0.8, rate: 26, hop: 0.7, hold: 1.6, fade: 0.5 };
    var tFlow = toy.NS / T.rate, cyc = T.intro + tFlow + T.hold + T.fade, Ks = [4, 1], total = cyc * Ks.length;

    function layout() {
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      W = Math.max(240, Math.round(el.clientWidth || 300));
      var side = compact || W >= 560, gap = compact ? 8 : 14, head = compact ? 18 : 24;
      var pw = side ? (W - gap) / 2 : W;
      var ph = compact ? Math.round(W * 0.75) : Math.round(pw * 0.8) + head;
      H = side ? ph : 2 * ph + 12;
      if (compact) ph = H;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      panels = side ? [{ x: 0, y: 0, w: pw, h: ph }, { x: pw + gap, y: 0, w: pw, h: ph }]
                    : [{ x: 0, y: 0, w: pw, h: ph }, { x: 0, y: ph + 12, w: pw, h: ph }];
      views = panels.map(function (p) {
        var top = p.y + head, h = p.h - head - 2, sc = Math.min(p.w / 4.9, h / 4.9), cx = p.x + p.w / 2, cy = top + h / 2;
        return { X: function (x) { return cx + x * sc; }, Y: function (y) { return cy - y * sc; }, sc: sc, top: top, h: h, cx: cx, cy: cy, p: p };
      });
      // arrow grid filling each panel (same for both panels)
      var v = views[0], cell = compact ? 22 : 30, nx = Math.max(6, Math.round(v.p.w / cell)), ny = Math.max(6, Math.round(v.h / cell));
      quiv = [];
      for (var i = 0; i < nx; i++) for (var j = 0; j < ny; j++) {
        var px = v.p.x + v.p.w * (i + 0.5) / nx, py = v.top + v.h * (j + 0.5) / ny;
        var x = (px - v.cx) / v.sc, y = (v.cy - py) / v.sc, b = field(x, y);
        quiv.push([px - v.p.x, py - v.top, b[0], b[1]]);
      }
      quiv.cell = Math.min(v.p.w / nx, v.h / ny);
    }

    function drawBase(v, alpha) {
      var p = v.p;
      ctx.globalAlpha = 1; ctx.fillStyle = COL.panel; ctx.fillRect(p.x, v.top, p.w, v.h);
      ctx.globalAlpha = 0.75; ctx.strokeStyle = COL.arrow; ctx.fillStyle = COL.arrow; ctx.lineWidth = 1;
      quiv.forEach(function (qv) {
        var m = Math.hypot(qv[2], qv[3]); if (m < 1e-6) return;
        var L = quiv.cell * 0.55 * Math.min(1, Math.sqrt(m / 1.2)); if (L < 2.5) return;
        var ux = qv[2] / m, uy = qv[3] / m, cx = p.x + qv[0], cy = v.top + qv[1];
        var x0 = cx - ux * L / 2, y0 = cy + uy * L / 2, x1 = cx + ux * L / 2, y1 = cy - uy * L / 2;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); arrowHead(ctx, x0, y0, x1, y1, compact ? 2.8 : 3.5);
      });
      ctx.globalAlpha = 1;
      DATA.forEach(function (a) { ctx.beginPath(); ctx.arc(v.X(a[0]), v.Y(a[1]), compact ? 4 : 5, 0, 6.2832); ctx.fillStyle = COL.data; ctx.fill(); });
    }

    function draw(t) {
      ctx.globalAlpha = 1; ctx.fillStyle = COL.stage; ctx.fillRect(0, 0, W, H);
      var c = Math.min(Ks.length - 1, Math.floor(t / cyc)), tc = t - c * cyc, K = Ks[c];
      var alpha = tc > cyc - T.fade ? 1 - ease((tc - (cyc - T.fade)) / T.fade) : 1;
      var vL = views[0], vR = views[1], r = compact ? 2.1 : 2.6;
      drawBase(vL); drawBase(vR);
      var appear = ease(tc / 0.5);
      var n = clamp(Math.floor((tc - T.intro) * T.rate), 0, toy.NS);
      ctx.save(); ctx.beginPath(); ctx.rect(vL.p.x, vL.top, vL.p.w, vL.h); ctx.clip();
      ctx.globalAlpha = alpha; ctx.lineWidth = 1; ctx.strokeStyle = COL.trail;
      toy.paths.forEach(function (P) {
        if (n > 0) { ctx.beginPath(); ctx.moveTo(vL.X(P[0]), vL.Y(P[1])); for (var s = 1; s <= n; s++) ctx.lineTo(vL.X(P[2 * s]), vL.Y(P[2 * s + 1])); ctx.stroke(); }
      });
      ctx.fillStyle = COL.accent; ctx.globalAlpha = alpha * appear;
      toy.paths.forEach(function (P) { ctx.beginPath(); ctx.arc(vL.X(P[2 * n]), vL.Y(P[2 * n + 1]), r, 0, 6.2832); ctx.fill(); });
      ctx.restore();
      var tj = tc - T.intro, done = clamp(Math.floor(tj / T.hop), 0, K), f = done < K ? ease((tj - done * T.hop) / T.hop) : 0;
      if (tj < 0) { done = 0; f = 0; }
      var st = []; for (var k = 0; k <= K; k++) st.push(stepAt(k / K, toy));
      ctx.save(); ctx.beginPath(); ctx.rect(vR.p.x, vR.top, vR.p.w, vR.h); ctx.clip();
      ctx.globalAlpha = alpha * 0.5; ctx.strokeStyle = COL.accent; ctx.lineWidth = 1;
      toy.paths.forEach(function (P) {
        ctx.beginPath(); ctx.moveTo(vR.X(P[0]), vR.Y(P[1]));
        for (var k2 = 1; k2 <= done; k2++) ctx.lineTo(vR.X(P[2 * st[k2]]), vR.Y(P[2 * st[k2] + 1]));
        if (done < K && f > 0) { var a0 = st[done], a1 = st[done + 1];
          ctx.lineTo(vR.X(P[2 * a0] + (P[2 * a1] - P[2 * a0]) * f), vR.Y(P[2 * a0 + 1] + (P[2 * a1 + 1] - P[2 * a0 + 1]) * f)); }
        ctx.stroke();
      });
      ctx.globalAlpha = alpha * appear; ctx.fillStyle = COL.accent;
      toy.paths.forEach(function (P) {
        var a0 = st[done], x = P[2 * a0], y = P[2 * a0 + 1];
        if (done < K && f > 0) { var a1 = st[done + 1]; x += (P[2 * a1] - x) * f; y += (P[2 * a1 + 1] - y) * f; }
        ctx.beginPath(); ctx.arc(vR.X(x), vR.Y(y), r, 0, 6.2832); ctx.fill();
      });
      ctx.restore();
      ctx.globalAlpha = 1;
      var calls = tj < 0 ? 0 : Math.min(K, done + (f > 0 ? 1 : 0));
      var fs = compact ? 11.5 : 13, yy = compact ? 12 : 15;
      ctx.textAlign = "left"; ctx.fillStyle = COL.fg;
      if (compact) {
        ctx.font = "600 " + fs + "px " + FONT;
        ctx.fillText("Equilibrium field", vL.p.x + 2, yy); ctx.fillText("Equilibrium flow map", vR.p.x + 2, yy);
        ctx.font = fs + "px " + FONT; ctx.textAlign = "right"; ctx.fillStyle = COL.accent;
        ctx.fillText(n + " calls", vL.p.x + vL.p.w - 2, yy);
        ctx.fillText(calls + (calls === 1 ? " call" : " calls"), vR.p.x + vR.p.w - 2, yy);
      } else {
        ctx.font = "600 " + fs + "px " + FONT;
        ctx.fillText("Equilibrium field", vL.p.x + 2, yy); ctx.fillText("Equilibrium flow map", vR.p.x + 2, yy);
        ctx.font = fs + "px " + FONT; ctx.textAlign = "right"; ctx.fillStyle = COL.accent;
        ctx.fillText(n + " network calls", vL.p.x + vL.p.w - 2, yy);
        ctx.fillText(calls + (calls === 1 ? " network call" : " network calls"), vR.p.x + vR.p.w - 2, yy);
      }
      ctx.textAlign = "left";
      if (capEl) {
        var cap = compact
          ? "Same noise, same field (arrows). Left: following the equilibrium field, " + toy.NS + " small steps settle on the data (green). Right: the equilibrium flow map gets there in " + K + (K === 1 ? " jump." : " jumps.")
          : "Same noise samples and the same field (gray arrows) in both panels. Samples settle on the data points (green), where the field is zero. Left: " + toy.NS +
            " small steps. Right: the equilibrium flow map reaches the same points in " + K + (K === 1 ? " jump." : " jumps, each landing on the path traced on the left.");
        if (capEl.textContent !== cap) capEl.textContent = cap;
      }
    }

    layout();
    var start = null, raf = 0, visible = true;
    function tick(now) {
      if (start === null) start = now;
      draw(((now - start) / 1000) % total);
      raf = requestAnimationFrame(tick);
    }
    function still() { draw(T.intro + tFlow + 0.2); }
    if (reduce) still();
    else {
      raf = requestAnimationFrame(tick);
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (es) {
          es.forEach(function (e) {
            if (e.isIntersecting && !visible) { visible = true; raf = requestAnimationFrame(tick); }
            else if (!e.isIntersecting && visible) { visible = false; cancelAnimationFrame(raf); }
          });
        }).observe(el);
      }
    }
    var rt = null;
    window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(function () { layout(); if (reduce) still(); }, 120); });
    el.__eqfm = { duration: total, seek: function (s) { cancelAnimationFrame(raf); visible = false; draw(Math.min(s, total - 0.01)); } };
    return el.__eqfm;
  }

  window.EqFMAnim = { mount: mount };
  document.querySelectorAll('[data-anim="eqfm"][data-automount]').forEach(function (el) {
    mount(el, { compact: el.getAttribute("data-mode") === "compact", captionEl: document.getElementById(el.getAttribute("data-caption") || "") });
  });
})();
