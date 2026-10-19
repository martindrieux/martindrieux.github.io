/* NFVM animation: a 1-D road cut into cells, a two-cell window feeding a "neural flux" box,
   flux arrows at every interface, the finite-volume update, and the density history.
   Self-contained, no libraries, no network.
   Usage: window.NFVMAnim.mount(element, {compact: true|false})

   Physics: LWR traffic model with Greenshields flow f(u) = u(1 - u), u = density / jam density,
   free-flow speed 1 cell per time unit. Cars drive left to right.
   For illustration the fluxes are computed with Godunov's classical scheme, not with the trained network.
   Cars are placed with cumulative vehicle counts, so they match the cell densities exactly. */
(function () {
  "use strict";

  var COL = {
    accent: "#1f5f8b", muted: "#666b70", tint: "#f3f4f5", fg: "#1a1a1a",
    flux: "#b45f1a", road: "#e2e5e8", rule: "#d5d9dd", car: "#2c3135", bg: "#ffffff"
  };

  var LAMBDA = 0.8;      // time step / cell width (CFL number, max wave speed is 1)
  var U_IN = 0.35;       // density of the free-flowing traffic arriving from the left
  var CAR_MASS = 0.5;    // one car = half a cell at jam density

  function flow(u) { return u * (1 - u); }
  // Godunov flux for a concave flow with its peak at u = 1/2 (demand / supply form).
  function godunov(a, b) {
    var demand = a < 0.5 ? flow(a) : 0.25;
    var supply = b > 0.5 ? flow(b) : 0.25;
    return Math.min(demand, supply);
  }

  function initial(N) {
    var u = new Array(N);
    for (var j = 0; j < N; j++) {
      var x = (j + 0.5) / N;
      u[j] = x < 0.45 ? U_IN : (x < 0.72 ? 0.95 : 0.05);  // moving traffic, a jam, a nearly empty road
    }
    return u;
  }

  function fluxes(u) {
    var N = u.length, F = new Array(N + 1);
    F[0] = godunov(U_IN, u[0]);                 // inflow of light traffic at the left edge
    for (var i = 1; i < N; i++) F[i] = godunov(u[i - 1], u[i]);
    F[N] = flow(u[N - 1]);                      // free outflow at the right edge
    return F;
  }

  function simulate(N, K) {
    var states = [initial(N)], F = [], Q = [0];
    for (var n = 0; n <= K; n++) {
      var u = states[n], Fn = fluxes(u);
      F.push(Fn);
      if (n === K) break;
      var v = new Array(N);
      for (var j = 0; j < N; j++) v[j] = u[j] + LAMBDA * (Fn[j] - Fn[j + 1]);
      states.push(v);
      Q.push(Q[n] + LAMBDA * Fn[0]);            // cumulative number of cars that entered (in mass units)
    }
    return { states: states, F: F, Q: Q };
  }

  // Interface with the largest upward jump in density (the back of the jam), or -1.
  function shockInterface(u) {
    var best = -1, jump = 0.12;
    for (var i = 1; i < u.length; i++) {
      var d = u[i] - u[i - 1];
      if (d > jump) { jump = d; best = i; }
    }
    return best;
  }

  function mix(a, b, t) {
    var pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    var r = Math.round(((pa >> 16) & 255) * (1 - t) + ((pb >> 16) & 255) * t);
    var g = Math.round(((pa >> 8) & 255) * (1 - t) + ((pb >> 8) & 255) * t);
    var bl = Math.round((pa & 255) * (1 - t) + (pb & 255) * t);
    return "rgb(" + r + "," + g + "," + bl + ")";
  }
  function heatColor(u) { return mix("#f6f7f8", COL.accent, Math.max(0, Math.min(1, u))); }
  function ease(t) { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); }

  var STATUS = {
    start: "Start: moving traffic (left) is about to reach a jam. Each bar is the average car density in one cell of road.",
    sweep1: "Step 1: the neural network reads two neighboring cells and outputs the flow of cars between them (orange arrow), interface after interface.",
    update1: "Step 2: each cell gains the cars that flow in from the left and loses those that flow out to the right. The cars move.",
    sweep2: "Same two steps again, faster: new flows at every interface, then the update.",
    update2: "Same two steps again, faster: new flows at every interface, then the update.",
    run: "The two steps now repeat on their own (same network at every interface). The back of the jam stays a sharp jump (a shock) and moves upstream, while its front spreads out as cars leave.",
    still: "Back of the jam: a sharp jump in density (a shock). Front: the jam spreads out as cars drive away. The network computes the flow at each interface from the two cells around it."
  };

  function mount(el, opts) {
    if (!el) return null;
    if (el.__nfvm) return el.__nfvm;
    opts = opts || {};
    var compact = !!opts.compact;
    var N = compact ? 20 : 24;
    var K = compact ? 30 : 34;
    var sim = simulate(N, K);

    var reduce = false;
    try { reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) {}

    el.textContent = "";
    var canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Animation of a neural finite volume solver on a road: cars, the density in each cell as bars, " +
      "a two-cell window feeding a small neural network that outputs the flow at the interface, and the density history over time.");
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.borderRadius = "6px";
    canvas.style.background = COL.tint;
    el.appendChild(canvas);
    var statusEl = opts.captionEl || null;
    if (!compact && !statusEl) {
      statusEl = document.createElement("p");
      statusEl.className = "nfvm-status";
      statusEl.style.cssText = "font-family:var(--font-ui,'Helvetica Neue',Arial,sans-serif);font-size:.9rem;line-height:1.45;color:" +
        COL.fg + ";margin:.5rem 0 0;min-height:2.9em";
      el.appendChild(statusEl);
    }
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0, L = null;

    function layout() {
      var w = Math.max(240, Math.round(el.clientWidth || 300));
      var h = compact ? Math.round(w * 0.75) : Math.round(Math.max(330, Math.min(470, w * 0.64)));
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      W = w; H = h;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var pad = compact ? 8 : 12, lab = compact ? 0 : 15, ax = compact ? 24 : 40, xlab = compact ? 13 : 16;
      var gx = pad + ax, gw = w - gx - pad, cw = gw / N;
      var y = pad;
      var o = { pad: pad, gx: gx, gw: gw, cw: cw, lab: lab, ax: ax, xlab: xlab };
      o.roadLab = y; y += lab;
      o.roadY = y; o.roadH = compact ? 16 : 22; y += o.roadH + (compact ? 6 : 8);
      o.barsLab = y; y += lab;
      var fluxH = compact ? 12 : 16, boxH = compact ? 30 : 42;
      var rest = h - y - pad - fluxH - boxH - (compact ? 6 : 8) - lab - xlab;
      o.barsY = y; o.barsH = Math.round(rest * 0.54); y += o.barsH;
      o.fluxY = y; o.fluxH = fluxH; y += fluxH;
      o.boxY = y; o.boxH = boxH; y += boxH + (compact ? 6 : 8);
      o.heatLab = y; y += lab;
      o.heatY = y; o.heatH = h - pad - xlab - y;
      o.font = compact ? 10 : 11.5;
      L = o;
    }

    // ---- timeline ----
    // start, slow pass (flows interface by interface), update, fast pass, update, then continuous evolution.
    var T_HOLD0 = 1.8, SLOT1 = compact ? 0.3 : 0.34, T_SW1 = (N - 1) * SLOT1, T_UPD1 = 1.6, T_GAP = 0.6;
    var SLOT2 = 0.06, T_SW2 = (N - 1) * SLOT2, T_UPD2 = 0.6;
    var T_STEP = 0.32, T_RUN = T_STEP * (K - 2), T_HOLD1 = 2.2, T_FADE = 0.5;
    var T_TOTAL = T_HOLD0 + T_SW1 + T_UPD1 + T_GAP + T_SW2 + T_UPD2 + T_RUN + T_HOLD1 + T_FADE;
    var boxX = 1, lastStatus = "";

    function sweep(f, t, slot, n, name) {
      var p = t / slot, k = Math.floor(p), fr = p - k;
      f.phase = name; f.n = n; f.s = 0;
      f.win = Math.min(N - 1, 1 + k + ease((fr - 0.72) / 0.28));      // window rests on an interface, then glides to the next (stops at the last one)
      f.arrows = k + 1; f.newest = k + 1; f.newestA = ease(fr / 0.35);
      boxX = f.win; return f;
    }

    function frameAt(tau, dt) {
      var f = { n: 0, s: 0, phase: "start", arrows: -1, newest: -1, newestA: 1, win: 1, alpha: 1 };
      var t = tau;
      if (t < T_HOLD0) { f.alpha = ease(t / 0.5); boxX = 1; f.win = 1; return f; }
      t -= T_HOLD0;
      if (t < T_SW1) return sweep(f, t, SLOT1, 0, "sweep1");
      t -= T_SW1;
      if (t < T_UPD1) { f.phase = "update1"; f.n = 0; f.s = ease(t / T_UPD1); f.arrows = N; f.win = boxX; return f; }
      t -= T_UPD1;
      if (t < T_GAP) { f.phase = "update1"; f.n = 1; f.s = 0; f.arrows = N; f.win = boxX; f.fluxN = 0; return f; }
      t -= T_GAP;
      if (t < T_SW2) { boxX = 1; return sweep(f, t, SLOT2, 1, "sweep2"); }
      t -= T_SW2;
      if (t < T_UPD2) { f.phase = "update2"; f.n = 1; f.s = ease(t / T_UPD2); f.arrows = N; f.win = boxX; return f; }
      t -= T_UPD2;
      if (t < T_RUN) {
        var q = t / T_STEP, m = Math.floor(q);
        f.phase = "run"; f.n = 2 + m; f.s = q - m; f.arrows = N;
        f.win = boxX; f.boxA = 1 - ease(t / 0.5); return f;   // window and network fade out, stay where the fast pass left them
      }
      t -= T_RUN;
      f.phase = "run"; f.n = K; f.s = 0; f.arrows = N;
      f.win = boxX; f.boxA = 0;
      if (t > T_HOLD1) f.alpha = 1 - ease((t - T_HOLD1) / T_FADE);
      return f;
    }

    function displayed(f) {
      var a = sim.states[f.n], b = sim.states[Math.min(K, f.n + 1)], u = new Array(N);
      for (var j = 0; j < N; j++) u[j] = a[j] + (b[j] - a[j]) * f.s;
      var Q = sim.Q[f.n] + (sim.Q[Math.min(K, f.n + 1)] - sim.Q[f.n]) * f.s;
      return { u: u, Q: Q };
    }

    function draw(f) {
      var o = L, cw = o.cw, gx = o.gx;
      ctx.globalAlpha = 1;
      ctx.fillStyle = COL.tint; ctx.fillRect(0, 0, W, H);
      ctx.globalAlpha = f.alpha;
      var d = displayed(f), u = d.u;
      ctx.font = o.font + "px 'Source Sans 3','Helvetica Neue',Arial,sans-serif";
      ctx.textBaseline = "alphabetic";

      // panel labels (full mode)
      if (!compact) {
        ctx.fillStyle = COL.muted;
        ctx.fillText("Cars on the road (driving right)", gx, o.roadLab + 11);
        ctx.fillText("Average density in each cell", gx, o.barsLab + 11);
        ctx.fillText("History of the density (time runs upward)", gx, o.heatLab + 11);
      }

      // road with cars
      ctx.fillStyle = COL.road; ctx.fillRect(gx, o.roadY, o.gw, o.roadH);
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1;
      for (var i = 1; i < N; i++) {
        var xi = Math.round(gx + i * cw) + 0.5;
        ctx.beginPath(); ctx.moveTo(xi, o.roadY); ctx.lineTo(xi, o.roadY + o.roadH); ctx.stroke();
      }
      var C = [0];
      for (var j = 0; j < N; j++) C.push(C[j] + u[j]);
      var total = C[N], carL = Math.max(3, cw * CAR_MASS * 0.78), carH = Math.max(5, o.roadH * 0.42);
      var kHi = Math.floor(d.Q / CAR_MASS - 0.5), kLo = Math.ceil((d.Q - total) / CAR_MASS - 0.5);
      ctx.save(); ctx.beginPath(); ctx.rect(gx, o.roadY, o.gw, o.roadH); ctx.clip();
      ctx.fillStyle = COL.car;
      var jj = 0;
      for (var kk = kHi; kk >= kLo; kk--) {
        var target = d.Q - (kk + 0.5) * CAR_MASS;  // mass between the road start and this car
        if (target < 0 || target > total) continue;
        while (jj < N - 1 && C[jj + 1] < target) jj++;
        var x = jj + (u[jj] > 1e-6 ? (target - C[jj]) / u[jj] : 0);
        var px = gx + x * cw;
        roundRect(px - carL / 2, o.roadY + (o.roadH - carH) / 2, carL, carH, 1.5);
        ctx.fill();
      }
      ctx.restore();

      // bars
      ctx.strokeStyle = COL.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(gx, o.barsY + 0.5); ctx.lineTo(gx + o.gw, o.barsY + 0.5); ctx.stroke();
      var winL = gx + (f.win - 1) * cw, winR = winL + 2 * cw;
      var boxA = f.boxA == null ? 1 : f.boxA;
      ctx.globalAlpha = f.alpha * boxA;
      ctx.fillStyle = "rgba(31,95,139,0.08)";
      ctx.fillRect(winL, o.barsY, 2 * cw, o.barsH + o.fluxH);
      ctx.globalAlpha = f.alpha;
      for (j = 0; j < N; j++) {
        var bh = u[j] * (o.barsH - 2);
        ctx.fillStyle = COL.accent;
        ctx.globalAlpha = f.alpha * 0.88;
        ctx.fillRect(gx + j * cw + Math.max(1, cw * 0.08), o.barsY + o.barsH - bh, cw - 2 * Math.max(1, cw * 0.08), bh);
      }
      ctx.globalAlpha = f.alpha;
      ctx.strokeStyle = COL.muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(gx, o.barsY + o.barsH + 0.5); ctx.lineTo(gx + o.gw, o.barsY + o.barsH + 0.5); ctx.stroke();

      // window outline
      ctx.globalAlpha = f.alpha * boxA;
      ctx.strokeStyle = COL.accent; ctx.lineWidth = 1.5;
      ctx.strokeRect(winL + 0.75, o.barsY + 0.75, 2 * cw - 1.5, o.barsH + o.fluxH - 1.5);

      ctx.globalAlpha = f.alpha;
      // flux arrows
      var Fn = sim.F[f.fluxN != null ? f.fluxN : f.n], ay = o.fluxY + o.fluxH / 2 + 1;
      var sweeping = f.phase === "sweep1" || f.phase === "sweep2";
      for (i = 0; i <= N; i++) {
        if (f.phase === "start") continue;
        if (sweeping && i > f.arrows && !(i === N && f.arrows >= N - 1)) continue;
        var a = (sweeping && i === f.newest) ? f.newestA : 1;
        var len = Math.max(3, (Fn[i] / 0.25) * cw * 0.82);
        var cx = gx + i * cw;
        ctx.globalAlpha = f.alpha * a;
        arrow(cx - len / 2, ay, cx + len / 2, ay, COL.flux, compact ? 1.6 : 2);
      }
      ctx.globalAlpha = f.alpha;

      // neural flux box under the window
      ctx.globalAlpha = f.alpha * boxA;
      var bxC = gx + f.win * cw, bw = compact ? 40 : 54, bhh = compact ? 20 : 26;
      var by = o.boxY + (compact ? 7 : 10);
      var bl = Math.max(gx, Math.min(gx + o.gw - bw, bxC - bw / 2));
      var nin = [[bl + bw * 0.22, by + bhh * 0.33], [bl + bw * 0.22, by + bhh * 0.67]];
      var nh = [[bl + bw * 0.5, by + bhh * 0.22], [bl + bw * 0.5, by + bhh * 0.5], [bl + bw * 0.5, by + bhh * 0.78]];
      var nout = [bl + bw * 0.78, by + bhh * 0.5];
      // inputs: from the two cell centres down into the box (drawn first, the box covers their ends)
      ctx.strokeStyle = COL.accent; ctx.lineWidth = 1.2;
      line(bxC - cw / 2, o.barsY + o.barsH, bxC - cw / 2, o.fluxY + o.fluxH);
      line(bxC + cw / 2, o.barsY + o.barsH, bxC + cw / 2, o.fluxY + o.fluxH);
      line(bxC - cw / 2, o.fluxY + o.fluxH, nin[0][0], nin[0][1]);
      line(bxC + cw / 2, o.fluxY + o.fluxH, nin[1][0], nin[1][1]);
      ctx.fillStyle = COL.bg; ctx.strokeStyle = COL.accent; ctx.lineWidth = 1.3;
      roundRect(bl, by, bw, bhh, 4); ctx.fill(); ctx.stroke();
      // tiny network inside the box: 2 inputs, 3 hidden, 1 output
      ctx.strokeStyle = "rgba(102,107,112,0.55)"; ctx.lineWidth = 0.8;
      for (var p1 = 0; p1 < 2; p1++) for (var p2 = 0; p2 < 3; p2++) line(nin[p1][0], nin[p1][1], nh[p2][0], nh[p2][1]);
      for (p2 = 0; p2 < 3; p2++) line(nh[p2][0], nh[p2][1], nout[0], nout[1]);
      ctx.fillStyle = COL.accent;
      var r = compact ? 1.7 : 2.2;
      nin.concat(nh).forEach(function (q) { ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 6.2832); ctx.fill(); });
      ctx.fillStyle = COL.flux; ctx.beginPath(); ctx.arc(nout[0], nout[1], r + 0.4, 0, 6.2832); ctx.fill();
      // output: out of the box and up to the arrow at the interface
      ctx.strokeStyle = COL.flux; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(nout[0], nout[1]); ctx.lineTo(bl + bw + 2, nout[1]);
      ctx.quadraticCurveTo(bl + bw + 9, o.fluxY + o.fluxH - 1, bxC + 1, o.fluxY + o.fluxH - 1);
      ctx.stroke();
      // label
      ctx.fillStyle = COL.fg;
      var label = compact ? "neural flux" : "neural flux, same at every interface", tw = ctx.measureText(label).width;
      var lx = bl + bw + 8, ly = by + bhh / 2 + 4;
      if (lx + tw > gx + o.gw) lx = bl - 6 - tw;
      ctx.fillText(label, lx, ly);
      ctx.globalAlpha = f.alpha;

      // history heatmap: rows 0..n, time upward
      var rows = K + 1, rh = o.heatH / rows;
      ctx.fillStyle = "#ffffff"; ctx.fillRect(gx, o.heatY, o.gw, o.heatH);
      var lastRow = f.n + (f.s > 0 ? 1 : 0);
      for (var n = 0; n <= lastRow && n <= K; n++) {
        var row = (n === f.n + 1) ? u : sim.states[n];
        var yb = o.heatY + o.heatH - (n + 1) * rh;
        for (j = 0; j < N; j++) {
          ctx.fillStyle = heatColor(row[j]);
          ctx.fillRect(gx + j * cw - 0.3, yb - 0.3, cw + 0.6, rh + 0.6);
        }
      }
      ctx.strokeStyle = COL.rule; ctx.lineWidth = 1;
      ctx.strokeRect(gx + 0.5, o.heatY + 0.5, o.gw - 1, o.heatH - 1);
      var yNow = o.heatY + o.heatH - (lastRow + 1) * rh;
      ctx.strokeStyle = COL.muted; ctx.lineWidth = 1;
      line(gx, Math.round(yNow) + 0.5, gx + o.gw, Math.round(yNow) + 0.5);
      if (!compact) {
        ctx.fillStyle = COL.muted; ctx.textAlign = "right";
        ctx.fillText("now", gx + o.gw - 4, Math.max(o.heatY + 11, yNow - 3));
        ctx.textAlign = "left";
      }
      drawAxes(o);
      ctx.globalAlpha = 1;

      if (statusEl) {
        var st = f.still ? STATUS.still : STATUS[f.phase];
        if (st !== lastStatus) { statusEl.textContent = st; lastStatus = st; }
      }
    }

    function drawAxes(o) {
      var fs = compact ? 9 : 11;
      ctx.save();
      ctx.font = fs + "px 'Source Sans 3','Helvetica Neue',Arial,sans-serif";
      ctx.strokeStyle = COL.muted; ctx.fillStyle = COL.muted; ctx.lineWidth = 1;
      var x0 = Math.round(o.gx) - 0.5;
      // density axis for the bars: 0 at the bottom, jam density (1) at the top
      var yb = o.barsY + o.barsH, yt = o.barsY + 2;
      line(x0, yt, x0, yb);
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      [[0, "0"], [0.5, "0.5"], [1, "1"]].forEach(function (tk) {
        var yy = Math.round(yb - tk[0] * (o.barsH - 2)) + 0.5;
        line(x0 - 3, yy, x0, yy);
        if (!compact || tk[0] !== 0.5) ctx.fillText(tk[1], x0 - 5, yy);
      });
      vlabel(compact ? "density" : "car density", o.pad + (compact ? 4 : 6), (yt + yb) / 2);
      // time axis for the history: time runs upward
      var hb = o.heatY + o.heatH, ht = o.heatY;
      line(x0, ht, x0, hb);
      ctx.beginPath(); ctx.moveTo(x0, ht - 1); ctx.lineTo(x0 - 3, ht + 5); ctx.lineTo(x0 + 3, ht + 5); ctx.closePath(); ctx.fill();
      vlabel("time", o.pad + (compact ? 4 : 6), (ht + hb) / 2);
      // position axis under the history
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillText("position on the road, cars drive right \u2192", o.gx, hb + (compact ? 11 : 13));
      ctx.restore();
    }
    function vlabel(text, x, y) {
      ctx.save(); ctx.translate(x, y); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, 0, 0); ctx.restore();
    }
    function line(x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
    function roundRect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
    }
    function arrow(x1, y, x2, y2, color, wdt) {
      var head = Math.min(5, Math.max(3, (x2 - x1) * 0.45));
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = wdt;
      ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2 - head * 0.8, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, y); ctx.lineTo(x2 - head, y - head * 0.7); ctx.lineTo(x2 - head, y + head * 0.7); ctx.closePath(); ctx.fill();
    }

    function staticFrame() {
      var n = Math.round(K * 0.65), tg = shockInterface(sim.states[n]);
      boxX = tg > 0 ? tg : 7;
      draw({ n: n, s: 0, phase: "run", arrows: N, newest: -1, newestA: 1, win: boxX, alpha: 1, still: true });
    }

    layout();
    var running = !reduce, start = null, prev = null, raf = 0;
    function tick(now) {
      if (start === null) { start = now; prev = now; }
      var dt = Math.min(0.1, (now - prev) / 1000); prev = now;
      var tau = ((now - start) / 1000) % T_TOTAL;
      draw(frameAt(tau, dt));
      raf = requestAnimationFrame(tick);
    }
    if (running) raf = requestAnimationFrame(tick); else staticFrame();

    var lastW = W;
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(function () {
        var w = Math.round(el.clientWidth || 0);
        if (w && Math.abs(w - lastW) > 1) { lastW = w; layout(); if (!running) staticFrame(); }
      }).observe(el);
    }
    el.__nfvm = {
      // for testing: draw the frame at a given loop time (seconds) and stop the loop
      seek: function (tau) {
        if (raf) cancelAnimationFrame(raf); raf = 0; running = false;
        boxX = 1; var t = 0, f;
        while (t < tau) { t = Math.min(tau, t + 1 / 30); f = frameAt(t, 1 / 30); }
        draw(f || frameAt(0, 0));
      },
      duration: T_TOTAL
    };
    return el.__nfvm;
  }

  window.NFVMAnim = { mount: mount };

  // Auto-mount any element marked data-anim="nfvm" with data-mode="full" or "compact".
  function auto() {
    var els = document.querySelectorAll('[data-anim="nfvm"][data-automount]');
    for (var i = 0; i < els.length; i++) {
      mount(els[i], { compact: els[i].getAttribute("data-mode") !== "full" });
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", auto); else auto();
})();
