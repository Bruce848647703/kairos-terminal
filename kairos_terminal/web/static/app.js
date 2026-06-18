/* Kairos Terminal —— 单页终端逻辑（原生 JS，无任何依赖） */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const state = {
    snap: null,
    meta: null,
    symbol: null,
    playing: false,
    timer: null,
    side: "buy",
    procKey: null,
    strategyKey: "",
    crosshair: -1,
    datasets: [],
  };

  // ------------------------------------------------------------------ 工具
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
    ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

  const signed = (v, d = 2) => (v > 0 ? "+" : "") + fmt(v, d);

  const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "muted");

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(path, body) {
    const opt = body === undefined ? {} : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
    const res = await fetch(path, opt);
    let payload;
    try {
      payload = await res.json();
    } catch (err) {
      throw new Error(`服务端返回了非 JSON 响应（HTTP ${res.status}）`);
    }
    if (!res.ok || payload.ok === false) {
      throw new Error(payload.error || `请求失败（HTTP ${res.status}）`);
    }
    return payload.data;
  }

  function flash(message) {
    const hint = $("#ticket-hint");
    if (!hint) return;
    hint.textContent = message;
    hint.style.color = "var(--gold)";
    clearTimeout(flash._t);
    flash._t = setTimeout(() => {
      hint.textContent = "市价单将在下一根 bar 开盘成交。";
      hint.style.color = "";
    }, 3500);
  }

  // ------------------------------------------------------------------ 刷新
  async function refresh() {
    const q = state.symbol ? `?symbol=${encodeURIComponent(state.symbol)}` : "";
    state.snap = await api("/api/state" + q);
    state.symbol = state.snap.active_symbol;
    render();
  }

  async function loadMeta() {
    state.meta = await api("/api/meta");
    fillProcessSelect($("#proc-select"), state.meta.processes);
    fillProcessSelect($("#add-proc"), state.meta.processes);
    const stratSel = $("#strategy");
    stratSel.innerHTML = '<option value="">（不挂载）</option>' +
      state.meta.strategies.map((s) => `<option value="${esc(s.key)}">${esc(s.name)}</option>`).join("");
    const labSel = $("#lab-symbol");
    labSel.innerHTML = state.meta.instruments
      .map((i) => `<option value="${esc(i.symbol)}">${esc(i.symbol)} ${esc(i.name)}</option>`).join("");
    bindParamForm($("#add-proc"), $("#add-params"));
    bindParamForm($("#proc-select"), $("#proc-params"));
  }

  function fillProcessSelect(select, processes) {
    select.innerHTML = processes
      .map((p) => `<option value="${esc(p.key)}">${esc(p.name)}</option>`).join("");
  }

  function processMeta(key) {
    return (state.meta?.processes || []).find((p) => p.key === key);
  }

  function bindParamForm(selectEl, container) {
    selectEl.addEventListener("change", () => {
      const meta = processMeta(selectEl.value);
      if (meta) renderParams(container, meta, meta.defaults);
    });
  }

  // ------------------------------------------------------------------ 渲染
  function render() {
    const s = state.snap;
    if (!s) return;
    const acct = s.account;
    const perf = s.performance || {};

    $("#m-equity").textContent = fmt(acct.equity, 0);
    $("#m-cash").textContent = fmt(acct.cash, 0);
    $("#m-upnl").textContent = signed(acct.unrealized_pnl, 0);
    $("#m-upnl").className = cls(acct.unrealized_pnl);
    $("#m-ret").textContent = signed(acct.return_pct, 2) + "%";
    $("#m-ret").className = cls(acct.return_pct);
    $("#m-mdd").textContent = fmt((perf.max_drawdown || 0) * 100, 2) + "%";
    $("#m-sharpe").textContent = fmt(perf.sharpe || 0, 2);

    const pctDone = (s.index / s.n_steps) * 100;
    $("#progress-fill").style.width = pctDone + "%";
    $("#progress-text").textContent = `bar ${s.index} / ${s.n_steps} · seed ${s.seed}`;

    renderWatchlist(s);
    renderHeader(s);
    renderProcessCard(s);
    renderTables(s);
    renderLogs(s);
    $("#strategy").value = s.strategy?.key || "";

    drawChart($("#chart"), s, state.crosshair);
    drawEquity($("#equity"), acct.equity_curve);
  }

  function renderWatchlist(s) {
    const ul = $("#watchlist");
    ul.innerHTML = s.instruments.map((i) => {
      const d = i.precision ?? 2;
      const pos = i.position_qty ? `<span class="badge soft">持仓 ${fmt(i.position_qty, 0)}</span>` : "";
      const badge = i.option
        ? `${i.option.underlying} ${i.option.kind_label}${fmt(i.option.strike, 2)} · 剩 ${Math.round((i.option.tau || 0) * 252)}`
        : i.process_name;
      return `<li data-sym="${esc(i.symbol)}" class="${i.symbol === s.active_symbol ? "active" : ""}">
        <div class="wl-sym">${esc(i.symbol)}</div>
        <div class="wl-px ${cls(i.change_pct)}">${fmt(i.price, d)}</div>
        <div class="wl-name">${esc(i.name)}</div>
        <div class="wl-chg ${cls(i.change_pct)}">${signed(i.change_pct, 2)}%</div>
        <div class="wl-badge">${esc(badge)} ${pos}</div>
      </li>`;
    }).join("");
    ul.querySelectorAll("li").forEach((li) => {
      li.addEventListener("click", () => {
        state.symbol = li.dataset.sym;
        state.crosshair = -1;
        refresh().catch((e) => flash(e.message));
      });
    });
  }

  function renderHeader(s) {
    const inst = s.instruments.find((i) => i.symbol === s.active_symbol) || s.instruments[0];
    const d = inst.precision ?? 2;
    $("#sym").textContent = inst.symbol;
    $("#sym-name").textContent = inst.name;
    $("#sym-class").textContent = inst.asset_class_label;
    $("#last").textContent = fmt(inst.price, d);
    $("#last").className = cls(inst.change_pct);
    $("#chg").textContent = `${signed(inst.change_pct, 2)}%`;
    $("#chg").className = cls(inst.change_pct);
    $("#tags").innerHTML = `<span class="badge">${esc(inst.option ? "期权" : inst.process_name)}</span>` +
      (inst.allow_short ? '<span class="badge soft">可做空</span>' : "");

    if (inst.option) {
      const o = inst.option;
      const g = o.greeks || {};
      $("#greeks").innerHTML =
        `<span>${esc(o.underlying)} ${esc(o.kind_label)} K=<b>${fmt(o.strike, 2)}</b></span>` +
        `<span>IV <b>${fmt(o.iv * 100, 1)}%</b></span>` +
        (g.delta !== undefined
          ? `<span>Δ <b>${fmt(g.delta, 3)}</b></span><span>Γ <b>${fmt(g.gamma, 4)}</b></span>` +
            `<span>ν <b>${fmt(g.vega, 3)}</b></span><span>Θ <b>${fmt(g.theta, 3)}</b></span>`
          : "") +
        `<span>剩余 <b>${Math.round((o.tau || 0) * 252)}</b> bar</span>`;
    } else {
      $("#greeks").innerHTML = "";
    }

    $("#ticket-sym").textContent = `${inst.symbol} · ${fmt(inst.price, d)}`;
    $("#hover").textContent = s.finished ? "已到末尾（可重置）" : "";

    if (document.activeElement !== $("#lab-spot")) {
      $("#lab-spot").value = Number(inst.price.toFixed(Math.max(d, 2)));
    }
    if (document.activeElement !== $("#lab-symbol")) {
      $("#lab-symbol").value = inst.symbol;
    }
  }

  function renderProcessCard(s) {
    const inst = s.instruments.find((i) => i.symbol === s.active_symbol) || s.instruments[0];
    const meta = processMeta(inst.process);
    const select = $("#proc-select");
    const apply = $("#btn-apply-proc");

    if (inst.option) {
      const o = inst.option;
      $("#proc-badge").textContent = "期权定价";
      $("#proc-desc").textContent =
        `价格 = 标的 ${o.underlying} 的行情 + Black-Scholes 逐 bar 重估；到期按内在价值现金交割。`;
      select.disabled = true;
      apply.disabled = true;
      select.closest(".field").hidden = true;
      apply.hidden = true;
      $("#proc-params").innerHTML =
        `<p class="hint">K=${fmt(o.strike, 2)} · ${esc(o.kind_label)} · IV=${fmt(o.iv * 100, 1)}% · ` +
        `到期 bar ${o.expiry_index}。想改变期权路径，请切换标的 ${o.underlying} 的价格过程。</p>`;
      state.procKey = null;
      return;
    }

    select.disabled = false;
    apply.disabled = false;
    select.closest(".field").hidden = false;
    apply.hidden = false;
    $("#proc-badge").textContent = inst.process_name;
    $("#proc-desc").textContent = meta ? meta.description : "";
    if (state.procKey !== inst.symbol + ":" + inst.process) {
      state.procKey = inst.symbol + ":" + inst.process;
      select.value = inst.process;
      if (meta) renderParams($("#proc-params"), meta, inst.process_params);
    }
  }

  function renderParams(container, meta, values) {
    container.innerHTML = meta.params.map((p) => {
      const id = `p-${p.name}`;
      let input;
      if (p.kind === "choice") {
        input = `<select data-name="${esc(p.name)}">${p.choices.map((c) =>
          `<option value="${esc(c)}">${esc(c)}</option>`).join("")}</select>`;
      } else if (p.kind === "bool") {
        input = `<select data-name="${esc(p.name)}">
          <option value="true">是</option><option value="false">否</option></select>`;
      } else {
        const step = p.step ?? (p.kind === "int" ? 1 : 0.01);
        input = `<input data-name="${esc(p.name)}" type="number" step="${step}" value="${esc(values[p.name])}">`;
      }
      const unit = p.unit ? ` <span class="muted">${esc(p.unit)}</span>` : "";
      const wide = p.kind === "str" ? " wide" : "";
      return `<label class="field${wide}"><span>${esc(p.label)}${unit}</span>${input}</label>`;
    }).join("");

    container.querySelectorAll("[data-name]").forEach((el) => {
      const name = el.dataset.name;
      if (el.tagName === "SELECT" && el.querySelector('option[value="true"]')) {
        el.value = String(values[name]) === "true" ? "true" : "false";
      } else if (values[name] !== undefined) {
        el.value = values[name];
      }
    });
  }

  function readParams(container) {
    const out = {};
    container.querySelectorAll("[data-name]").forEach((el) => {
      const name = el.dataset.name;
      if (el.tagName === "SELECT") {
        out[name] = el.value === "true" ? true : el.value === "false" ? false : el.value;
      } else if (el.value !== "") {
        out[name] = Number(el.value);
      }
    });
    return out;
  }

  function renderTables(s) {
    const d = (sym) => {
      const i = s.instruments.find((x) => x.symbol === sym);
      return i ? (i.precision ?? 2) : 2;
    };

    const pos = s.account.positions;
    $("#positions").querySelector("tbody").innerHTML = pos.length
      ? `<tr><th>品种</th><th>数量</th><th>成本</th><th>最新</th><th>盈亏</th></tr>` +
        pos.map((p) => `<tr>
          <td>${esc(p.symbol)}</td>
          <td>${fmt(p.qty, 0)}</td>
          <td>${fmt(p.entry_price, d(p.symbol))}</td>
          <td>${fmt(p.last_price, d(p.symbol))}</td>
          <td class="${cls(p.unrealized_pnl)}">${signed(p.unrealized_pnl, 2)}</td>
        </tr>`).join("")
      : `<tr><td class="empty">暂无持仓</td></tr>`;

    const orders = s.orders || [];
    $("#orders").querySelector("tbody").innerHTML = orders.length
      ? `<tr><th>#</th><th>品种</th><th>方向</th><th>数量</th><th>状态</th><th></th></tr>` +
        orders.slice(0, 12).map((o) => `<tr>
          <td>${o.id}</td>
          <td>${esc(o.symbol)}</td>
          <td class="${o.side === 'buy' ? 'up' : 'down'}">${o.side_label}</td>
          <td>${fmt(o.qty, 0)}</td>
          <td class="muted">${o.status_label}</td>
          <td>${o.status === "pending" ? `<button data-cancel="${o.id}">撤</button>` : ""}</td>
        </tr>`).join("")
      : `<tr><td class="empty">暂无委托</td></tr>`;

    $("#orders").querySelectorAll("[data-cancel]").forEach((btn) => {
      btn.addEventListener("click", () => {
        api("/api/cancel", { id: Number(btn.dataset.cancel) })
          .then(refresh).catch((e) => flash(e.message));
      });
    });
  }

  function renderLogs(s) {
    const box = $("#logs");
    box.innerHTML = (s.logs || []).map((l) => (
      `<div class="k-${esc(l.kind)}"><span class="t">[${l.index}]</span>${esc(l.message)}</div>`
    )).join("");
  }

  // ------------------------------------------------------------------ 绘图
  const COLORS = {
    up: "#f6465d", down: "#2ebd85", grid: "#1b222c",
    axis: "#5d6675", ma5: "#f0b90b", ma20: "#c792ea",
    equity: "#4c8dff", fillUp: "rgba(246,70,93,.16)",
  };

  let lastChart = null;

  function drawChart(canvas, snap, crosshair) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w < 20 || h < 20) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const bars = snap.bars || [];
    lastChart = { snap, w, h };
    if (!bars.length) {
      ctx.fillStyle = COLORS.axis;
      ctx.font = "12px sans-serif";
      ctx.fillText("等待行情…（点击 ▶ 播放 或 ⏭ 单步）", 12, 24);
      return;
    }

    const padL = 8, padR = 62, padT = 8, padB = 18;
    const volH = Math.max(28, h * 0.16);
    const priceH = h - padT - padB - volH - 6;
    const innerW = w - padL - padR;
    const n = bars.length;
    const step = innerW / n;
    const bw = Math.max(1, Math.min(step * 0.62, 14));

    let lo = Infinity, hi = -Infinity;
    for (const b of bars) { lo = Math.min(lo, b.l); hi = Math.max(hi, b.h); }
    const ma5 = sma(bars.map((b) => b.c), 5);
    const ma20 = sma(bars.map((b) => b.c), 20);
    for (const v of [...ma5, ...ma20]) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.06 || 0.5;
    lo -= pad; hi += pad;
    const y = (p) => padT + (hi - p) / (hi - lo) * priceH;
    const x = (i) => padL + step * (i + 0.5);
    const maxVol = Math.max(...bars.map((b) => b.v), 1);
    const volY = (v) => h - padB - (v / maxVol) * volH;

    // 网格 + 右侧价格轴
    ctx.font = "10px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const p = hi - (hi - lo) * (i / 4);
      const yy = y(p);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(padL + innerW, yy); ctx.stroke();
      ctx.fillStyle = COLORS.axis;
      ctx.fillText(fmt(p, 2), padL + innerW + 6, yy);
    }

    // 时间轴刻度
    const tickEvery = Math.max(1, Math.floor(n / 6));
    ctx.fillStyle = COLORS.axis;
    ctx.textAlign = "center";
    for (let i = 0; i < n; i += tickEvery) {
      ctx.fillText(String(bars[i].t), x(i), h - 6);
    }
    ctx.textAlign = "left";

    // 成交量
    for (let i = 0; i < n; i++) {
      const b = bars[i];
      ctx.fillStyle = b.c >= b.o ? "rgba(246,70,93,.55)" : "rgba(46,189,133,.55)";
      const top = volY(b.v);
      ctx.fillRect(x(i) - bw / 2, top, bw, h - padB - top);
    }

    // K 线
    for (let i = 0; i < n; i++) {
      const b = bars[i];
      const up = b.c >= b.o;
      ctx.strokeStyle = up ? COLORS.up : COLORS.down;
      ctx.fillStyle = up ? COLORS.up : COLORS.down;
      const xi = x(i);
      ctx.beginPath(); ctx.moveTo(xi, y(b.h)); ctx.lineTo(xi, y(b.l)); ctx.stroke();
      const top = y(Math.max(b.o, b.c));
      const height = Math.max(1, Math.abs(y(b.o) - y(b.c)));
      ctx.fillRect(xi - bw / 2, top, bw, height);
    }

    // 均线
    drawLine(ctx, ma5, x, y, COLORS.ma5);
    drawLine(ctx, ma20, x, y, COLORS.ma20);

    // 成交标记
    const start = snap.bar_window_start || 0;
    for (const f of snap.fills || []) {
      if (f.symbol !== snap.active_symbol) continue;
      const bi = f.index - 1 - start;
      if (bi < 0 || bi >= n) continue;
      const buy = f.side === "buy";
      const yy = y(f.price);
      ctx.fillStyle = buy ? COLORS.up : COLORS.down;
      ctx.beginPath();
      if (buy) { ctx.moveTo(x(bi), yy + 8); ctx.lineTo(x(bi) - 4, yy + 14); ctx.lineTo(x(bi) + 4, yy + 14); }
      else { ctx.moveTo(x(bi), yy - 8); ctx.lineTo(x(bi) - 4, yy - 14); ctx.lineTo(x(bi) + 4, yy - 14); }
      ctx.closePath(); ctx.fill();
    }

    // 十字光标
    if (crosshair >= 0 && crosshair < n) {
      const b = bars[crosshair];
      const xi = x(crosshair);
      ctx.strokeStyle = "rgba(255,255,255,.28)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xi, padT); ctx.lineTo(xi, h - padB); ctx.stroke();
      ctx.setLineDash([]);
      const d = instPrecision(snap);
      tooltip(snap, b, x(crosshair), y(b.c), w, h, d);
    } else {
      $("#chart-tip").hidden = true;
    }
  }

  function instPrecision(snap) {
    const inst = (snap.instruments || []).find((i) => i.symbol === snap.active_symbol);
    return inst ? (inst.precision ?? 2) : 2;
  }

  function tooltip(snap, b, px, py, w, h, d) {
    const tip = $("#chart-tip");
    tip.hidden = false;
    tip.innerHTML =
      `<div>bar ${b.t}</div>` +
      `<div>开 ${fmt(b.o, d)}　高 ${fmt(b.h, d)}</div>` +
      `<div>低 ${fmt(b.l, d)}　收 ${fmt(b.c, d)}</div>` +
      `<div>量 ${fmt(b.v, 0)}</div>`;
    const left = px > w / 2 ? px - tip.offsetWidth - 12 : px + 12;
    const top = Math.min(Math.max(py - 40, 4), h - tip.offsetHeight - 4);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function sma(arr, n) {
    const out = new Array(arr.length).fill(NaN);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
      if (i >= n) sum -= arr[i - n];
      if (i >= n - 1) out[i] = sum / n;
    }
    return out;
  }

  function drawLine(ctx, values, x, y, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let started = false;
    values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      if (!started) { ctx.moveTo(x(i), y(v)); started = true; }
      else ctx.lineTo(x(i), y(v));
    });
    if (started) ctx.stroke();
  }

  function drawEquity(canvas, curve) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w < 20 || h < 10) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const data = curve && curve.length ? curve : [0];
    if (data.length < 2) {
      ctx.fillStyle = COLORS.axis;
      ctx.font = "11px sans-serif";
      ctx.fillText("尚无净值数据", 10, 20);
      return;
    }
    const lo = Math.min(...data), hi = Math.max(...data);
    const pad = (hi - lo) * 0.12 || 1;
    const x = (i) => 4 + (i / (data.length - 1)) * (w - 68);
    const y = (v) => 6 + (hi + pad - v) / (hi - lo + 2 * pad) * (h - 12);

    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(76,141,255,.30)");
    grad.addColorStop(1, "rgba(76,141,255,0)");
    ctx.beginPath();
    ctx.moveTo(x(0), y(data[0]));
    data.forEach((v, i) => ctx.lineTo(x(i), y(v)));
    ctx.lineTo(x(data.length - 1), h);
    ctx.lineTo(x(0), h);
    ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    data.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.strokeStyle = COLORS.equity;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const last = data[data.length - 1];
    ctx.fillStyle = last >= data[0] ? "#f6465d" : "#2ebd85";
    ctx.font = "10px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "middle";
    ctx.fillText(fmt(last, 0), w - 62, y(last));
    ctx.fillText(fmt(hi, 0), w - 62, y(hi));
    ctx.fillText(fmt(lo, 0), w - 62, y(lo));
    $("#equity-note").textContent =
      `${data.length - 1} bars · ${signed((last / data[0] - 1) * 100, 2)}%`;
  }

  // ------------------------------------------------------------------ 交互
  function bindTicket() {
    document.querySelectorAll("#side button").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.side = btn.dataset.v;
        document.querySelectorAll("#side button").forEach((b) => b.classList.toggle("on", b === btn));
        const submit = $("#btn-submit");
        submit.textContent = state.side === "buy" ? "买入" : "卖出";
        submit.className = "primary " + state.side;
      });
    });

    $("#otype").addEventListener("change", () => {
      const t = $("#otype").value;
      $("#row-price").hidden = t === "market";
      $("#price-label").textContent = t === "limit" ? "限价" : "触发价";
    });

    $("#btn-submit").addEventListener("click", async () => {
      const type = $("#otype").value;
      const payload = {
        symbol: state.symbol,
        side: state.side,
        qty: Number($("#qty").value),
        type,
      };
      if (type === "limit") payload.limit_price = Number($("#price").value);
      if (type === "stop") payload.stop_price = Number($("#price").value);
      try {
        const order = await api("/api/order", payload);
        flash(`已提交 #${order.id} ${order.symbol} ${order.side_label} ${order.qty}（${order.type_label}）`);
        await refresh();
      } catch (e) { flash("下单失败：" + e.message); }
    });
  }

  function bindControls() {
    $("#btn-step").addEventListener("click", () => stepOnce(1));
    $("#btn-play").addEventListener("click", () => togglePlay());
    $("#btn-reset").addEventListener("click", async () => {
      try {
        stopPlay();
        await api("/api/reset", { seed: Number($("#seed").value), n_steps: state.snap.n_steps });
        state.crosshair = -1;
        await refresh();
      } catch (e) { flash("重置失败：" + e.message); }
    });
    $("#speed").addEventListener("change", () => { if (state.playing) { stopPlay(); togglePlay(); } });

    $("#btn-apply-proc").addEventListener("click", async () => {
      try {
        const proc = $("#proc-select").value;
        await api("/api/process", {
          symbol: state.symbol, process: proc, params: readParams($("#proc-params")),
        });
        state.procKey = null;
        await refresh();
        flash(`已切换 ${state.symbol} 的价格过程为 ${processMeta(proc)?.name || proc}`);
      } catch (e) { flash("切换失败：" + e.message); }
    });

    $("#btn-strategy").addEventListener("click", async () => {
      try {
        const key = $("#strategy").value;
        await api("/api/strategy", { key: key || null, params: {} });
        await refresh();
      } catch (e) { flash("策略设置失败：" + e.message); }
    });

    $("#btn-add").addEventListener("click", () => openModal());
    $("#btn-chain").addEventListener("click", async () => {
      const inst = (state.snap?.instruments || []).find((i) => i.symbol === state.symbol);
      if (!inst) return;
      const underlying = inst.option ? inst.option.underlying : inst.symbol;
      try {
        const res = await api("/api/chain", {
          underlying, expiries: [60, 120], n_strikes: 3, width: 0.03,
        });
        await refresh();
        flash(`已生成 ${res.count} 张期权（标的 ${underlying}，ATM IV ${(res.anchor_iv * 100).toFixed(1)}%）`);
      } catch (e) { flash("生成期权链失败：" + e.message); }
    });
    $("#add-class").addEventListener("change", syncOptionFields);
    $("#add-source").addEventListener("change", syncSource);
    $("#add-opt-under").addEventListener("change", () => {
      const inst = (state.snap?.instruments || []).find((i) => i.symbol === $("#add-opt-under").value);
      if (inst) $("#add-opt-strike").value = Number(inst.price.toFixed(2));
    });
    document.querySelectorAll("[data-close]").forEach((b) => {
      b.addEventListener("click", () => { $("#" + b.dataset.close).hidden = true; });
    });
    $("#btn-add-confirm").addEventListener("click", addInstrument);
    $("#btn-lab").addEventListener("click", () => {
      $("#drawer-lab").hidden = false;
      $("#drawer-lab").querySelector("[data-method]").focus();
    });
    document.querySelectorAll("[data-method]").forEach((b) => {
      b.addEventListener("click", () => runLab(b.dataset.method));
    });

    // 图表十字光标
    const canvas = $("#chart");
    canvas.addEventListener("mousemove", (ev) => {
      if (!lastChart) return;
      const rect = canvas.getBoundingClientRect();
      const n = (lastChart.snap.bars || []).length;
      if (!n) return;
      const padL = 8, padR = 62;
      const step = (rect.width - padL - padR) / n;
      const idx = Math.floor((ev.clientX - rect.left - padL) / step);
      const clamped = Math.max(0, Math.min(n - 1, idx));
      if (clamped !== state.crosshair) {
        state.crosshair = clamped;
        drawChart(canvas, lastChart.snap, clamped);
        const b = lastChart.snap.bars[clamped];
        $("#hover").textContent =
          `O ${fmt(b.o, 2)} H ${fmt(b.h, 2)} L ${fmt(b.l, 2)} C ${fmt(b.c, 2)} V ${fmt(b.v, 0)}`;
      }
    });
    canvas.addEventListener("mouseleave", () => {
      state.crosshair = -1;
      $("#hover").textContent = "";
      if (lastChart) drawChart(canvas, lastChart.snap, -1);
    });

    document.addEventListener("keydown", (ev) => {
      const tag = (ev.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      if (ev.code === "Space") { ev.preventDefault(); togglePlay(); }
      if (ev.code === "ArrowRight") { ev.preventDefault(); stepOnce(1); }
    });

    window.addEventListener("resize", () => {
      if (state.snap) {
        drawChart($("#chart"), state.snap, state.crosshair);
        drawEquity($("#equity"), state.snap.account.equity_curve);
      }
    });
  }

  async function stepOnce(n) {
    try {
      await api("/api/step", { n });
      await refresh();
    } catch (e) {
      flash("推进失败：" + e.message);
      stopPlay();
    }
  }

  function togglePlay() {
    if (state.playing) { stopPlay(); return; }
    state.playing = true;
    $("#btn-play").textContent = "⏸ 暂停";
    $("#btn-play").classList.add("on");
    const speed = Number($("#speed").value);
    const loop = async () => {
      if (!state.playing) return;
      if (state.snap && state.snap.finished) { stopPlay(); return; }
      try {
        await api("/api/step", { n: speed });
        await refresh();
      } catch (e) {
        flash("推进失败：" + e.message);
        stopPlay();
        return;
      }
      if (state.playing) state.timer = setTimeout(loop, 320);
    };
    loop();
  }

  function stopPlay() {
    state.playing = false;
    if (state.timer) clearTimeout(state.timer);
    state.timer = null;
    $("#btn-play").textContent = "▶ 播放";
    $("#btn-play").classList.remove("on");
  }

  // ------------------------------------------------------------------ 新增品种
  async function loadDatasets() {
    if (state.datasets.length) return;
    try {
      const data = await api("/api/datasets");
      state.datasets = data.items || [];
      const sel = $("#add-dataset");
      sel.innerHTML = state.datasets.length
        ? state.datasets.slice(0, 300).map((d) =>
            `<option value="${esc(d.name)}">${esc(d.name)} · ${d.rows}行 · ${esc(d.start)}→${esc(d.end)}</option>`).join("")
        : '<option value="">（未找到本地 CSV）</option>';
      $("#add-dataset-hint").textContent = state.datasets.length
        ? `共 ${state.datasets.length} 个数据集，搜索路径：${(data.roots || []).join(" , ")}`
        : "未找到 CSV：设置环境变量 KAIROS_TERMINAL_DATA，或把行情放到 ./data、../kairos-data/data";
    } catch (e) {
      $("#add-dataset-hint").textContent = "数据集加载失败：" + e.message;
    }
  }

  function syncSource() {
    const isData = $("#add-source").value === "dataset";
    $("#add-dataset-box").hidden = !isData;
    $("#add-proc").closest(".field").hidden = isData;
    $("#add-params").hidden = isData;
    $("#add-s0-field").hidden = isData;
  }

  function syncOptionFields() {
    const isOption = $("#add-class").value === "option";
    $("#add-option-box").hidden = !isOption;
    $("#add-proc-box").hidden = isOption;
    $("#add-s0-field").hidden = isOption;
    $("#add-mult-field").hidden = isOption;
    $("#add-margin-field").hidden = isOption;
    if (isOption) {
      const unders = (state.snap?.instruments || []).filter((i) => !i.option);
      $("#add-opt-under").innerHTML = unders.map((i) =>
        `<option value="${esc(i.symbol)}">${esc(i.symbol)} ${esc(i.name)}</option>`).join("");
      const inst = unders[0];
      if (inst) $("#add-opt-strike").value = Number(inst.price.toFixed(2));
    } else {
      syncSource();
    }
  }

  function openModal() {
    $("#modal-add").hidden = false;
    const meta = processMeta($("#add-proc").value);
    if (meta) renderParams($("#add-params"), meta, meta.defaults);
    syncOptionFields();
    loadDatasets();
    $("#add-symbol").focus();
  }

  async function addInstrument() {
    const assetClass = $("#add-class").value;
    const symbol = $("#add-symbol").value.trim();
    const payload = {
      action: "add",
      symbol,
      name: $("#add-name").value.trim() || symbol,
      asset_class: assetClass,
      multiplier: Number($("#add-mult").value),
      margin_rate: Number($("#add-margin").value),
      allow_short: $("#add-short").value === "true",
      tick_size: Number($("#add-tick").value),
    };

    if (assetClass === "option") {
      const spec = {
        underlying: $("#add-opt-under").value,
        strike: Number($("#add-opt-strike").value),
        kind: $("#add-opt-kind").value,
        expiry_index: Number($("#add-opt-expiry").value),
        iv: Number($("#add-opt-iv").value),
        multiplier: Number($("#add-opt-mult").value),
        allow_short: true,
      };
      payload.option = spec;
      payload.multiplier = spec.multiplier;
      payload.allow_short = true;
    } else if ($("#add-source").value === "dataset") {
      const dataset = $("#add-dataset").value;
      if (!dataset) { flash("没有可用的本地数据集"); return; }
      payload.data_path = dataset;
      payload.block = Number($("#add-block").value);
      payload.process = "bootstrap";
      payload.s0 = null;
    } else {
      payload.process = $("#add-proc").value;
      payload.params = readParams($("#add-params"));
      payload.s0 = Number($("#add-s0").value);
    }

    try {
      await api("/api/instrument", payload);
      $("#modal-add").hidden = true;
      state.procKey = null;
      await refresh();
      flash(`已创建品种 ${payload.symbol}`);
    } catch (e) { flash("创建失败：" + e.message); }
  }

  // ------------------------------------------------------------------ 定价实验室
  function labPayload(method) {
    return {
      method,
      symbol: $("#lab-symbol").value,
      kind: $("#lab-kind").value,
      spot: Number($("#lab-spot").value),
      strike: Number($("#lab-strike").value || $("#lab-spot").value),
      maturity: Number($("#lab-maturity").value),
      rate: Number($("#lab-rate").value),
      vol: Number($("#lab-vol").value),
      hedge_vol: Number($("#lab-hvol").value),
      price: Number($("#lab-price").value),
      n_paths: Number($("#lab-paths").value),
      n_steps: Number($("#lab-steps").value),
      payoff: "vanilla",
      steps: 300,
      steps_binomial: 300,
    };
  }

  const LAB_LABELS = {
    price: "价格", stderr: "标准误", ci_low: "95%CI 下界", ci_high: "95%CI 上界",
    black_scholes: "Black-Scholes", binomial: "二叉树", monte_carlo: "蒙特卡洛",
    mc_stderr: "MC 标准误", delta: "Delta", gamma: "Gamma", vega: "Vega",
    theta: "Theta", rho: "Rho", implied_vol: "隐含波动率",
    premium: "理论权利金", mean_pnl: "对冲 PnL 均值", std_pnl: "对冲 PnL 标准差",
    prob_loss: "亏损概率", hedge_vol: "对冲波动率", d1: "d1", d2: "d2",
    terminal_mean: "到期价均值", intrinsic: "内在价值", model_process: "过程",
  };

  async function runLab(method) {
    const box = $("#lab-result");
    box.innerHTML = "计算中…";
    try {
      let payload = labPayload(method);
      if (method === "implied_vol" && !(payload.price > 0)) {
        // 没填市价时，用当前波动率算出的 BS 价做「往返验证」
        const bs = await api("/api/price", { ...payload, method: "bs" });
        payload.price = bs.price;
        $("#lab-price").value = Number(bs.price.toFixed(4));
      }
      const data = await api("/api/price", payload);
      const rows = Object.entries(data)
        .filter(([k, v]) => ["number", "string"].includes(typeof v) && k !== "kind"
          && k !== "payoff" && k !== "model_process")
        .map(([k, v]) => `<tr><td>${esc(LAB_LABELS[k] || k)}</td><td>${typeof v === "number" ? fmt(v, 6) : esc(v)}</td></tr>`)
        .join("");
      let extra = "";
      if (data.pnl_percentiles) {
        extra = `<h4>PnL 分位数 (%)</h4><table><tr><td>1%</td><td>5%</td><td>25%</td><td>50%</td><td>75%</td><td>95%</td><td>99%</td></tr>
          <tr>${data.pnl_percentiles.map((v) => `<td>${fmt(v, 2)}</td>`).join("")}</tr></table>`;
      }
      if (data.mc_ci) {
        extra += `<h4>MC 置信区间</h4><div>[${fmt(data.mc_ci[0], 6)}, ${fmt(data.mc_ci[1], 6)}]</div>`;
      }
      if (data.terminal_percentiles) {
        extra += `<h4>到期价分位数</h4><div>${data.terminal_percentiles.map((v) => fmt(v, 2)).join(" · ")}</div>`;
      }
      box.innerHTML = `<table>${rows}</table>${extra}`;
    } catch (e) {
      box.innerHTML = `<span class="up">计算失败：${esc(e.message)}</span>`;
    }
  }

  // ------------------------------------------------------------------ 启动
  async function init() {
    bindTicket();
    bindControls();
    // 支持 ?symbol=510300C4 深链，直接打开某个品种
    const deepLink = new URLSearchParams(location.search).get("symbol");
    if (deepLink) state.symbol = deepLink;
    try {
      await loadMeta();
      await refresh();
    } catch (e) {
      document.body.insertAdjacentHTML("afterbegin",
        `<div style="padding:12px;color:#f6465d">初始化失败：${esc(e.message)}</div>`);
      return;
    }
    // 初始化实验室默认值
    $("#lab-strike").value = $("#lab-spot").value;
  }

  init();
})();