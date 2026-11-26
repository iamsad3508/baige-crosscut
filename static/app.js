'use strict';
/* Copyright (C) 2026 iamsad3508
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Part of Baige (百格附着力分析). Commercial licence: see COMMERCIAL.md */
/* 百格分析 — 前端
   坐标：canvas 屏幕坐标 <-> 分析图像素坐标（图像像素 = 服务端 work 图，长边 res 指定）
   所有像素级算法在服务端（复用 crosscut.py 纯函数），前端只负责交互与显示。 */

/* 一次性/带变量的文案：T(中文, English)。静态 HTML 用 data-i18n，这里用 T 少加一堆 key。 */
const T = (window.T || ((zh, en) => (en != null ? zh : zh)));

const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

const S = {
  id: null, name: '', ow: 0, oh: 0, w: 0, h: 0, res: 'full', rot: 0,
  img: null,
  fit: 1, zoom: 1, zoomMode: 'fit', panX: 0, panY: 0, cw: 1, ch: 1, dpr: 1,
  mode: 'rect',
  rois: [], sel: -1,
  samples: [], selSample: -1,
  tol: 25, purity: 0.5, line: 1, exmode: 'width', widthTouched: false, suggRetry: false, lineSrc: 'default',
  autoTol: true, semi: true, detail: false, showOverlay: true, softOn: false, soft: null, kneeWarn: '',
  curRid: (() => { try { return localStorage.getItem('baige.curRid') || null; } catch (_) { return null; } })(),
  roiNative: true, analysis: null,
  overlayUrl: null, overlayImg: null, contourUrl: null, contourImg: null,
  rows: null, total: null, lines: [], suggest: null, suggestSrc: '',
  span: null,
  rulerHist: [], rulerFull: [], rulerLast: null, rulerSeg: null, rulerDone: null, rulerFadeT: null,
  suggestHold: null,
  drag: null, busy: false, seq: 0, autoAbort: null,
  annot: { tool: 1, r: 15, strokes: [], cur: null, res: null },
  blocks: null, blocksOn: true, blocksTimer: null, ants: 0,
  blocksSeq: 0,
  grid: null, gridCells: false,
  gm: null, gmPitch: 0, gmArea: false, gmOff: { ox: 0, oy: 0, dx: [], dy: [] }, gmDrag: null,
  // 格网里「人改过」的标记：改了就不再让自动值覆盖（用户明确要求保留手改）
  gmColsManual: false, gmRowsManual: false, gmAngManual: false,
  gmCountsSig: '', gmAutoSig: '', samplesManual: false,
};

/* 顶栏那些复选框：一律记住上次的选择。用户预期是"取消勾 → 刷新也还是取消"，
   以前只记住了少数几个，半自动/诊断明细每次刷新都被强制回默认。 */
const PREFKEY = { semi: 'baige.semi', detail: 'baige.detail', overlay: 'baige.overlay', roiNative: 'baige.roiNative' };
function prefGet(k, d) {
  try { const v = localStorage.getItem(PREFKEY[k]); return v === null ? d : v === '1'; } catch (_) { return d; }
}
function prefSet(k, v) { try { localStorage.setItem(PREFKEY[k], v ? '1' : '0'); } catch (_) {} }

/* ------------------------------------------------------------------ 工具 */
function post(path, body, method, signal) {
  const opt = { method: method || 'POST', headers: { 'Content-Type': 'application/json', 'X-Baige-Lang': I18N.lang } };
  if (body !== undefined) opt.body = JSON.stringify(body);
  if (signal) opt.signal = signal;
  return fetch(path, opt).then(async r => {
    if (r.status === 404) throw new Error(T('图片已过期（分析缓存被换出），请重新「打开图片」上传一次', 'The image expired (analysis cache evicted) — upload it again with “Open image”'));
    if (!r.ok) throw new Error((await r.text()).slice(0, 200) || ('HTTP ' + r.status));
    return r.json();
  });
}
function fmt(n, d = 2) { return (Math.round(n * Math.pow(10, d)) / Math.pow(10, d)).toFixed(d); }
function debounce(fn, ms) {
  let t;
  const f = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  f.cancel = () => clearTimeout(t);          // 双指缩放时要把待发的重算掐掉
  return f;
}
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/* ------------------------------------------------------------------ 画布几何 */
const cv = $('#cv'), ctx = cv.getContext('2d');
const wrap = $('#wrap');

function scale() { return S.fit * S.zoom; }
function toImg(ex, ey) { const s = scale(); return { x: (ex - S.panX) / s, y: (ey - S.panY) / s }; }
function toScreen(x, y) { const s = scale(); return { x: S.panX + x * s, y: S.panY + y * s }; }

/* 指针 → 画布逻辑坐标：用实时 rect 并按 CSS/位图比例换算，
   这样侧栏变化导致画布尺寸变动时坐标也不会跳 */
function evPos(e) {
  const r = cv.getBoundingClientRect();
  const kx = S.cw / (r.width || 1), ky = S.ch / (r.height || 1);
  return { x: (e.clientX - r.left) * kx, y: (e.clientY - r.top) * ky };
}

let resizePending = null;
/* 尺寸变化入口：必须【同步】处理。
   若推迟到 rAF，会有一帧「CSS 尺寸已变、canvas 位图还是旧尺寸」，
   浏览器把旧位图拉伸铺满 → 就是看到的跳动/重影。 */
function scheduleResize() {
  resizeCanvas();
}

function resizeCanvas() {
  if (S.drag) { resizePending = true; return; }   // 拖拽中不改几何，避免画布突然缩放
  const r = wrap.getBoundingClientRect();
  const nw = Math.max(80, Math.ceil(r.width));
  const nh = Math.max(80, Math.ceil(r.height));
  if (nw === S.cw && nh === S.ch && cv.width) return;   // 尺寸没变：不动位图、不重算，避免无谓闪烁
  const anchor = S.img ? toImg(S.cw / 2, S.ch / 2) : null;
  const zoomed = S.zoom > 1.001;
  S.dpr = Math.min(2, window.devicePixelRatio || 1);
  S.cw = nw; S.ch = nh;
  cv.width = Math.round(S.cw * S.dpr);
  cv.height = Math.round(S.ch * S.dpr);
  /* 显式像素尺寸：CSS 盒宽高与位图逻辑宽高严格相等。用 100% 时窗口宽度出现小数
     （如 1048.6px）会让浏览器每次重采样，看起来发虚、像有重影 */
  cv.style.width = S.cw + 'px';
  cv.style.height = S.ch + 'px';
  if (S.img) {
    S.fit = Math.min(S.cw / S.w, S.ch / S.h);
    if (S.zoomMode === 'px1') S.zoom = 1 / S.fit;      // 1:1 永远是 1 图像像素 = 1 屏幕像素
    else if (S.zoomMode === 'fit') S.zoom = 1;
    if (zoomed && anchor) {                // 缩放状态：保持"画布中心那个像素"不动，视觉上不跳
      const s = S.fit * S.zoom;
      S.panX = S.cw / 2 - anchor.x * s;
      S.panY = S.ch / 2 - anchor.y * s;
    }
  }
  clampPan();
  draw();
}


function fitView() {
  if (!S.img) return;
  S.fit = Math.min(S.cw / S.w, S.ch / S.h);
  S.zoom = 1; S.zoomMode = 'fit';
  S.panX = (S.cw - S.w * S.fit) / 2;
  S.panY = (S.ch - S.h * S.fit) / 2;
  draw();
}
function zoomOne() {
  if (!S.img) return;
  const c = toImg(S.cw / 2, S.ch / 2);
  S.zoom = 1 / S.fit; S.zoomMode = 'px1';
  S.panX = S.cw / 2 - c.x * scale();
  S.panY = S.ch / 2 - c.y * scale();
  clampPan(); draw();
}
function zoomAt(factor, ax, ay) {
  if (!S.img) return;
  const p = toImg(ax, ay);
  S.zoom = clamp(S.zoom * factor, 0.05, 40); S.zoomMode = 'free';
  const s = scale();
  S.panX = ax - p.x * s;
  S.panY = ay - p.y * s;
  clampPan(); draw();
}
function clampPan() {
  if (!S.img) return;
  const s = scale(), dw = S.w * s, dh = S.h * s, m = 60;
  if (dw <= S.cw) S.panX = (S.cw - dw) / 2; else S.panX = clamp(S.panX, S.cw - dw - m, m);
  if (dh <= S.ch) S.panY = (S.ch - dh) / 2; else S.panY = clamp(S.panY, S.ch - dh - m, m);
}

/* ------------------------------------------------------------------ 旋转与手柄 */
function boxCenter(r) { return { x: (r.box[0] + r.box[2]) / 2, y: (r.box[1] + r.box[3]) / 2 }; }
function rot(p, c, deg) {
  const t = deg * Math.PI / 180, dx = p.x - c.x, dy = p.y - c.y;
  return { x: c.x + dx * Math.cos(t) - dy * Math.sin(t), y: c.y + dx * Math.sin(t) + dy * Math.cos(t) };
}
function unrot(p, c, deg) {
  const t = deg * Math.PI / 180, dx = p.x - c.x, dy = p.y - c.y;
  return { x: c.x + dx * Math.cos(t) + dy * Math.sin(t), y: c.y - dx * Math.sin(t) + dy * Math.cos(t) };
}
function cornersOf(r) {
  const c = boxCenter(r), hw = (r.box[2] - r.box[0]) / 2, hh = (r.box[3] - r.box[1]) / 2;
  const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  return local.map(([dx, dy]) => rot({ x: c.x + dx, y: c.y + dy }, c, r.ang || 0));
}
/* 8 个缩放手柄（sx,sy ∈ {-1,0,1}） */
function handleDefs(r) {
  const c = boxCenter(r), hw = (r.box[2] - r.box[0]) / 2, hh = (r.box[3] - r.box[1]) / 2;
  const defs = [];
  for (const sx of [-1, 0, 1]) for (const sy of [-1, 0, 1]) {
    if (!sx && !sy) continue;
    defs.push({ sx, sy, p: rot({ x: c.x + sx * hw, y: c.y + sy * hh }, c, r.ang || 0) });
  }
  return defs;
}
function rotateHandlePoint(r) {
  const c = boxCenter(r), hh = (r.box[3] - r.box[1]) / 2;
  const top = rot({ x: c.x, y: c.y - hh }, c, r.ang || 0);
  const cs = toScreen(c.x, c.y), ts = toScreen(top.x, top.y);
  let dx = ts.x - cs.x, dy = ts.y - cs.y;
  const d = Math.hypot(dx, dy) || 1;
  return { x: ts.x + dx / d * 24, y: ts.y + dy / d * 24 };
}

/* ------------------------------------------------------------------ 绘制 */
let lastZoomLabel = '', drawQueued = false;
/* 拖动过程里用这个：一帧最多重绘一次。直接每次 pointermove 都全量重绘，
   高刷新率鼠标会把主线程塞满，画面反而落后指针、一顿一顿地追 */
function drawSoon() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

function draw() {
  ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  ctx.clearRect(0, 0, S.cw, S.ch);
  if (!S.img) return;
  const s = scale(), dw = S.w * s, dh = S.h * s;
  ctx.imageSmoothingEnabled = s < 2;
  ctx.drawImage(S.img, S.panX, S.panY, dw, dh);
  if (S.showOverlay && S.overlayImg) {
    ctx.globalAlpha = 0.96;
    ctx.drawImage(S.overlayImg, S.panX, S.panY, dw, dh);
    ctx.globalAlpha = 1;
  }

  drawAnnotations();
  drawBlockAnts();
  drawBlocks();
  drawRois();
  drawGrid();
  drawModelGrid();
  drawRuler();
  if (S.drag && S.drag.kind === 'new') {
    const { a, b } = S.drag;
    const p0 = toScreen(a.x, a.y), p1 = toScreen(b.x, b.y);
    ctx.save();
    const rw = Math.abs(p1.x - p0.x), rh = Math.abs(p1.y - p0.y);
    const rx = Math.min(p0.x, p1.x), ry = Math.min(p0.y, p1.y);
    ctx.setLineDash([]);                       // 深色底：任何底色上都看得见
    ctx.strokeStyle = 'rgba(0,0,0,.6)';
    ctx.lineWidth = 4.5;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = '#ffe14d';
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();
  }
  const zl = Math.round(s * 100) + '%';
  if (zl !== lastZoomLabel) { lastZoomLabel = zl; $('#zoomLabel').textContent = zl; }
}

/* --- 格网（实验）：把自动铺出来的格子画上去，按脱落率上色 --- */
function gridColor(pct) {
  // 0% 绿 → 35% 黄 → 65%+ 红，和结果区的分级口径一致
  const t = Math.max(0, Math.min(1, pct / 65));
  const r = Math.round(60 + t * 195), g = Math.round(200 - t * 150), b = Math.round(90 - t * 40);
  return `rgba(${r},${g},${b},`;
}

function drawGrid() {
  if (!S.grid) return;
  const s = scale();
  ctx.save();
  // 格线：只当参考线（细、半透明），不再是“每格一个等级”
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255,62,165,.45)';
  for (const x of S.grid.xs) {
    const p0 = toScreen(x, 0), p1 = toScreen(x, S.h);
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
  }
  for (const y of S.grid.ys) {
    const p0 = toScreen(0, y), p1 = toScreen(S.w, y);
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
  }
  // 整片视场一个框 + 一个数
  const b = S.grid.box;
  const a0 = toScreen(b[0], b[1]), a1 = toScreen(b[2], b[3]);
  const w = a1.x - a0.x, h = a1.y - a0.y;
  const col = gridColor(S.grid.total.pct);
  ctx.fillStyle = col + '0.10)';
  ctx.fillRect(a0.x, a0.y, w, h);
  ctx.strokeStyle = col + '0.95)';
  ctx.lineWidth = 2;
  ctx.strokeRect(a0.x, a0.y, w, h);
  if (S.gridCells) {               // 诊断：每格上色 + 百分比（默认关）
    for (const c of S.grid.cells) {
      const p0 = toScreen(c.box[0], c.box[1]), p1 = toScreen(c.box[2], c.box[3]);
      const cw = p1.x - p0.x, ch = p1.y - p0.y;
      ctx.fillStyle = gridColor(c.pct) + '0.18)';
      ctx.fillRect(p0.x, p0.y, cw, ch);
      if (cw > 40 && ch > 24) {
        ctx.font = '600 11px system-ui, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const txt = c.pct.toFixed(1) + '%';
        ctx.fillStyle = 'rgba(0,0,0,.55)';
        ctx.fillText(txt, p0.x + cw / 2 + 1, p0.y + ch / 2 + 1);
        ctx.fillStyle = '#fff';
        ctx.fillText(txt, p0.x + cw / 2, p0.y + ch / 2);
      }
    }
  }
  // 总量写在框的上沿
  ctx.font = '700 15px system-ui, sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
  const lab = T('视场脱落 ', 'field removal ') + S.grid.total.pct.toFixed(2) + '%';
  const tw = ctx.measureText(lab).width;
  ctx.fillStyle = 'rgba(0,0,0,.66)';
  ctx.fillRect(a0.x, a0.y - 24, tw + 16, 22);
  ctx.fillStyle = '#fff';
  ctx.fillText(lab, a0.x + 8, a0.y - 4);
  ctx.restore();
}

/* --- 模型格网（旧图/翻拍图）：刀路按人给的宽度画，逐格统计格内剩余漆膜面积 --- */
function gmAreaText(px2) {
  const g = S.gm;
  if (S.gmPitch > 0 && g && g.pitch_x > 0) {          // 给了格距就换算成 mm²
    const f = S.gmPitch / g.pitch_x;
    return fmt(px2 * f * f, 2) + ' mm²';
  }
  if (px2 >= 1e6) return fmt(px2 / 1e6, 2) + 'M px²';
  if (px2 >= 1e3) return fmt(px2 / 1e3, 1) + 'k px²';
  return Math.round(px2) + ' px²';
}

function gmPath(q) {
  ctx.beginPath();
  q.forEach((p, i) => {
    const s = toScreen(p[0], p[1]);
    if (i) ctx.lineTo(s.x, s.y); else ctx.moveTo(s.x, s.y);
  });
  ctx.closePath();
}

function drawModelGrid() {
  const g = S.gm;
  // 注意：没有格子（比如还没吸色、或格子全落在刀带里）也必须继续画刀带和框，
  // 否则"点了铺格什么都没出现"——几何跟分类无关，该画就得画
  if (!g) return;
  ctx.save();
  const dg = S.gmDrag || {};
  const gx = dg.kind === 'move' ? dg.gdx : 0, gy = dg.kind === 'move' ? dg.gdy : 0;
  const sh = q => (gx || gy) ? q.map(p => [p[0] + gx, p[1] + gy]) : q;
  for (const b of (g.bands || [])) {                  // 刀缝带：给定宽度，所见即所扣
    let q = b.quad || b;
    const dragging = dg.kind === 'line' && dg.axis === b.axis && dg.idx === b.idx;
    if (dragging) q = q.map(p => [p[0] + dg.ldx, p[1] + dg.ldy]);
    gmPath(sh(q));
    ctx.fillStyle = dragging ? 'rgba(255,190,70,.55)' : 'rgba(255,150,40,.30)';
    ctx.fill();
    ctx.strokeStyle = dragging ? '#fff3c4' : 'rgba(255,205,120,.9)';
    ctx.lineWidth = dragging ? 2 : 1;
    ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(70,225,255,.55)';
  ctx.lineWidth = 1;
  for (const c of g.cells) { gmPath(sh(c.quad)); ctx.stroke(); }

  if (S.gmArea) {
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const c of g.cells) {
      const cx2 = (c.quad[0][0] + c.quad[2][0]) / 2 + gx, cy2 = (c.quad[0][1] + c.quad[2][1]) / 2 + gy;
      const s = toScreen(cx2, cy2);
      const p0 = toScreen(c.quad[0][0] + gx, c.quad[0][1] + gy), p1 = toScreen(c.quad[2][0] + gx, c.quad[2][1] + gy);
      const cw = Math.abs(p1.x - p0.x), chh = Math.abs(p1.y - p0.y);
      if (cw < 34 || chh < 20) continue;
      if (c.pct == null) continue;                    // 没吸色就没有百分比，别印 null
      const t1 = Math.round(100 - c.pct) + '%';       // 剩余漆膜占比
      const t2 = gmAreaText(c.kept);                  // 剩余漆膜面积
      ctx.font = '700 12px system-ui, sans-serif';
      const w1 = ctx.measureText(t1).width;
      ctx.font = '600 10px system-ui, sans-serif';
      const w2 = ctx.measureText(t2).width;
      const bw = Math.max(w1, w2) + 8;
      const two = chh > 40;
      ctx.fillStyle = 'rgba(8,10,16,.70)';
      ctx.fillRect(s.x - bw / 2, s.y - (two ? 15 : 9), bw, two ? 30 : 18);
      ctx.font = '700 12px system-ui, sans-serif';
      ctx.fillStyle = c.pct <= 10 ? '#8dffa8' : (c.pct <= 60 ? '#ffe36e' : '#ff8b7d');
      ctx.fillText(t1, s.x, s.y - (two ? 7 : 0));
      if (two) {
        ctx.font = '600 10px system-ui, sans-serif';
        ctx.fillStyle = 'rgba(230,240,255,.92)';
        ctx.fillText(t2, s.x, s.y + 7);
      }
    }
  }

  const b = g.box, a0 = toScreen(b[0] + gx, b[1] + gy);
  const noSmp = g.total.pct == null;
  const main = noSmp
    ? T('还没吸色：点一下漆膜、再点一下露底', 'No samples yet: click coating, then bare')
    : T('框选区脱落 ', 'region removal ') + fmt(g.total.pct, 1) + '%';
  const sub = (noSmp ? T('格子已按你的列/行铺好', 'Grid laid out from your columns/rows')
    : gmAreaText(g.total.kept) + T(' / 合计 ', ' of ') + gmAreaText(g.total.area))
    + ' · ' + g.cols + '×' + g.rows + ' · ' + T('刀宽 ', 'blade ') + Math.round(g.k) + 'px';
  ctx.font = '800 20px system-ui, sans-serif';
  const w1 = ctx.measureText(main).width;
  ctx.font = '600 12px system-ui, sans-serif';
  const w2 = ctx.measureText(sub).width;
  const bw2 = Math.max(w1, w2) + 18;
  ctx.fillStyle = 'rgba(8,10,16,.88)';
  ctx.fillRect(a0.x, a0.y - 52, bw2, 48);
  ctx.fillStyle = noSmp ? '#ffd88a' : gridColor(g.total.pct) + '1)';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.font = '800 20px system-ui, sans-serif';
  ctx.fillText(main, a0.x + 9, a0.y - 50);
  ctx.font = '600 12px system-ui, sans-serif';
  ctx.fillStyle = 'rgba(225,235,250,.85)';
  ctx.fillText(sub, a0.x + 9, a0.y - 25);
  ctx.restore();
}

function distSeg(px, py, a, b) {
  const vx = b.x - a.x, vy = b.y - a.y, L2 = vx * vx + vy * vy;
  let t = L2 ? ((px - a.x) * vx + (py - a.y) * vy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (a.x + t * vx), py - (a.y + t * vy));
}

/* 画布上抓格网：靠近某根刀缝带 → 拖这一根；在框内空白处 → 整体平移 */
function gmHitAt(ex, ey) {
  const g = S.gm;
  if (!g || !g.bands) return null;
  for (const b of g.bands) {
    const q = (b.quad || b).map(p => toScreen(p[0], p[1]));
    /* quad 顺序：0 左上、1 右上、2 右下、3 左下。
       竖线的长轴是「上边中点 → 下边中点」，横线是「左边中点 → 右边中点」——
       以前两种都用后者，竖线那段只有 k px 长，点哪儿都够不着，于是全都落进"整体平移"。 */
    const a = b.axis === 'x'
      ? { x: (q[0].x + q[1].x) / 2, y: (q[0].y + q[1].y) / 2 }
      : { x: (q[0].x + q[3].x) / 2, y: (q[0].y + q[3].y) / 2 };
    const c = b.axis === 'x'
      ? { x: (q[3].x + q[2].x) / 2, y: (q[3].y + q[2].y) / 2 }
      : { x: (q[1].x + q[2].x) / 2, y: (q[1].y + q[2].y) / 2 };
    if (distSeg(ex, ey, a, c) < 10) return { axis: b.axis, idx: b.idx, n: b.n || [1, 0] };
  }
  const bx = g.box;
  if (bx) {
    const p0 = toScreen(bx[0], bx[1]), p1 = toScreen(bx[2], bx[3]);
    if (ex > Math.min(p0.x, p1.x) && ex < Math.max(p0.x, p1.x)
      && ey > Math.min(p0.y, p1.y) && ey < Math.max(p0.y, p1.y)) return { move: true };
  }
  return null;
}

function gmBody(extra) {
  return Object.assign({
    // 格网一律用 full（= 画布这张 wire 图）算：预览分辨率下裁出来的内容与画布上的框对不上，
    // 认缝会跑空（实测同一张图 full 认 11 条、prev 只认 1 条）
    id: S.id, res: 'full', rois: S.rois.map(r => ({ box: r.box, ang: r.ang })),
    samples: S.samples.map(x => ({ rgb: x.rgb, role: x.role })),
    tol: S.tol, purity: S.purity, line_px: S.line, mode: S.exmode,
    loc_tile: 64, loc_factor: 1.6, roi_native: S.roiNative,
    model: true,
    // 没手改过的那一边 → 让服务端自己数；两边都手改过就完全不认（永远以人为准）
    auto_cols: !S.gmColsManual && S.gmCountsSig !== gmRoiSig(),
    auto_rows: !S.gmRowsManual && S.gmCountsSig !== gmRoiSig(),
    auto_angle: !S.gmAngManual,          // 框完就自己估歪角（只回建议，人改过就不再动）
    roles_auto: !S.samplesManual,        // 颜色是自动分的 → 允许服务端按格内面积纠正角色
    cols: +$('#gmCols').value || 0, rows: +$('#gmRows').value || 0,
    // 刀宽与「忽略细线宽度」合成一个数（S.line）：同一个物理量，两个框只会互相打架
    angle: +$('#gmAng').value || 0, k_band: S.line || 0,
    ox: S.gmOff.ox, oy: S.gmOff.oy,
    line_dx: S.gmOff.dx.slice(), line_dy: S.gmOff.dy.slice(),
    overlay: !!S.showOverlay,
  }, extra || {});
}

/* 选区的“位置签名”：框挪了才重新认格数/重新分色，参数一动就重来只会让数字跳 */
function gmRoiSig() {
  return JSON.stringify(S.rois.map(r => r.box.map(v => Math.round(v))));
}

/* 刀缝分离：用上面那套线位把「刀缝 / 刀口崩边 / 格内脱落」分成三层各给一个数。
   几何（列数/行数/歪角/刀宽/拖线/对线）全部来自模型格网，这里只多做一件事——
   把缝外 EDGE 宽度内比漆膜更暗的那圈单独计数，不和格内脱落揉在一起。 */
async function runSep() {
  if (!S.id) { setStatus(T('先打开一张图', 'Open an image first')); return; }
  if (!S.rois.length) { setStatus(T('先在图上框住整片百格', 'Draw a region around the grid first')); return; }
  if (!(+$('#gmCols').value >= 2) || !(+$('#gmRows').value >= 2)) {
    setStatus(T('先在格网里填列数/行数', 'Set columns/rows first')); return;
  }
  if (S.busy) return;
  setBusy(true);
  setStatus(T('正在分离刀缝…', 'Separating the cuts…'));
  const out = $('#sepOut');
  try {
    if (!S.samples.length) { S.gmAuto = false; await gmAutoSample(true); }
    const b = gmBody({
      edge_w: Math.max(0, +$('#sepEdge').value || 0),
      k_mult: Math.max(1.05, +$('#sepK').value || 1.5),
      overlay: true,
    });
    const r = await post('api/separate', b);
    if (!r.ok) throw new Error(r.reason || T('分离失败', 'separation failed'));
    if (r.overlay_url) { S.showOverlay = true; if ($('#chkOverlay')) $('#chkOverlay').checked = true; loadOverlay(r.overlay_url); }
    const pct = r.flare_pct + r.blk_pct;
    out.innerHTML = T(
      `刀缝带 <b>${fmt(r.seam_pct, 1)}%</b> 的选区（扣掉）　｜　刀口崩边 <b>${fmt(r.flare_pct, 2)}%</b>　｜　格内成块露底 <b>${fmt(r.blk_pct, 2)}%</b><br>`
      + `→ 按「崩边+格内」算 <b>${fmt(pct, 2)}%</b> · ${r.astm}　｜　k=${r.k_mult} 时阈值随 k 变：`
      + r.sweep.map(x => `${x.k}→${x.pct}%`).join(' / '),
      `cut band <b>${fmt(r.seam_pct, 1)}%</b> of region (excluded) | edge flaking <b>${fmt(r.flare_pct, 2)}%</b> | flakes inside cells <b>${fmt(r.blk_pct, 2)}%</b><br>`
      + `→ counting flaking only: <b>${fmt(pct, 2)}%</b> · ${r.astm} | sweep: `
      + r.sweep.map(x => `${x.k}→${x.pct}%`).join(' / '));
    setStatus(T('分离完成：蓝=刀缝、橙=崩边、红=格内成块露底',
                'Done — blue = cut, orange = edge flaking, red = flakes inside cells'));
  } catch (e) {
    out.textContent = T('分离失败：', 'Failed: ') + (e.message || e);
    setStatus(String(e.message || e));
  } finally { setBusy(false); }
}

/* 列/行填的是「格子」数，刀线 = 格子数 + 1；把这层换算直接写在界面上，免得填反 */
function updateGmLines() {
  const el = $('#gmLines');
  if (!el) return;
  const c = +($('#gmCols') ? $('#gmCols').value : 0) || 0;
  const r = +($('#gmRows') ? $('#gmRows').value : 0) || 0;
  if (c < 2 || r < 2) { el.textContent = ''; return; }
  el.textContent = T(`→ 竖刀线 ${c + 1} 条 ／ 横刀线 ${r + 1} 条（刀线比格子多 1）`,
                     `→ ${c + 1} vertical cuts / ${r + 1} horizontal cuts (one more than cells)`);
}

function gmSummary(g) {
  const n = g.cells.length;
  const intact = g.cells.filter(c => c.pct <= 10).length;
  const gone = g.cells.filter(c => c.pct > 60).length;
  if (g.total.pct == null) {
    return T('格子已铺好（' + g.cols + '×' + g.rows + '），但还没吸色：图上点一下完好漆膜、再点一下露底，'
      + '或按「自动分色」', 'Grid laid out (' + g.cols + '×' + g.rows + ') but no samples: '
      + 'click coating then bare, or press "Auto colours"');
  }
  return T('框选区脱落 ', 'removal ') + fmt(g.total.pct, 1) + '% · '
    + T('未脱落 ', 'left ') + gmAreaText(g.total.kept) + ' / ' + gmAreaText(g.total.area)
    + ' · ' + g.cols + '×' + g.rows + ' · ' + T('刀宽 ', 'blade ') + Math.round(g.k) + 'px'
    + ' · ' + T('格距 ', 'pitch ') + Math.round(g.pitch_x) + '×' + Math.round(g.pitch_y) + 'px'
    + ' · ' + T('完整格 ' + intact + '/' + n, 'intact ' + intact + '/' + n)
    + (gone ? ' · ' + T('基本掉光 ' + gone, 'gone ' + gone) : '');
}

/* 自动分色：按亮度把选区像素分两簇给一对色样。只是起点，图上点一下可改 */
async function gmAutoSample(silent) {
  if (!S.id || !S.rois.length) return false;
  if (S.samplesManual) return false;          // 人定过的颜色：换框/重算都不重分
  try {
    const r = await post('api/autosample', {
      id: S.id, res: S.res, rois: S.rois.map(x => ({ box: x.box, ang: x.ang || 0 })),
    });
    if (!r.ok) { if (!silent) setStatus(r.reason || T('自动分色失败', 'Auto colours failed')); return false; }
    S.samples = r.samples.map(x => ({ rgb: x.rgb.slice(), role: x.role }));
    S.selSample = -1; S.span = r.span; S.gmAuto = true; S.samplesManual = false;
    if (S.autoTol && typeof r.auto_tol === 'number') S.tol = r.auto_tol;
    syncSampleUI(); updateSpan(); draw();
    if (r.warn) setStatus(r.warn);
    return true;
  } catch (e) {
    if (!silent) setStatus(T('自动分色失败：', 'Auto colours failed: ') + e.message);
    return false;
  }
}

/* 判法=格网 时的半自动：框完/改参数就自己跑一遍（合并连点，别每次拖动都发请求） */
let gmTimer = null, gmPending = false;
function scheduleGm(ms) {
  if (S.exmode !== 'grid' || !S.id || !S.rois.length) return;
  gmPending = true;
  if (gmTimer) clearTimeout(gmTimer);
  gmTimer = setTimeout(pumpGm, ms || 700);
}
/* 忙的时候不能把请求丢掉：改刀宽会同时触发 analyze()，analyze() 一进来就 busy，
   老代码在定时器到点那一刻直接 return（且不重试）→ 刀宽/「计算」永远不生效，
   只有「自动定角」那种 await 完再直调的路径能跑。这里改成等它闲下来再跑。 */
function pumpGm() {
  gmTimer = null;
  if (!gmPending) return;
  if (S.busy) { gmTimer = setTimeout(pumpGm, 180); return; }
  gmPending = false;
  runModelGrid();
}
/* 刀宽与「忽略细线宽度」合并成一个数之后，格网块里那行只显示“沿用 = N px” */
function refreshGmK() {
  const on = S.exmode === 'grid';
  const row = $('#gmKRow'), inh = $('#gmKInheritRow');
  if (row) row.hidden = on;
  if (inh) inh.hidden = !on;
  const t = $('#gmKInherit');
  if (t) {
    t.textContent = on
      ? T(`沿用「忽略细线宽度」= ${S.line} px（在「量尺」页改这个数）`,
          `same as the cut threshold = ${S.line} px (change it on the ruler tab)`)
      : '';
  }
}

/* 判法=格网 时，结果卡直接显示格网口径的数（框选区百分比 + 等级），
   不再让主卡显示像素判法那个数——两个数并排只会让人不知道信哪个 */
function setGridCard(g) {
  const pct = +((g.total || {}).pct || 0);
  const grd = g.grade || '—';
  const head = $('#summary');
  if (head) {
    head.textContent = `${T('框选区', 'Region')} ${pct.toFixed(2)}%  ${grd}`
      + `（${g.cols}×${g.rows}，${T('格网口径', 'grid method')}）`;
    head.className = 'summary' + ((grd === '0B' || grd === '1B') ? ' bad' : '');
  }
  const ss = $('#sideSummary');
  if (ss) ss.textContent = `${pct.toFixed(2)}% ${grd}`;
  const bd = $('#resBadge');
  if (bd) { bd.textContent = grd; bd.className = 'rg ' + gcls(grd); bd.title = ''; }
  const pc = $('#resCardPct');
  if (pc) pc.textContent = `${pct.toFixed(2)}% ${T('脱落（框选区·格网）', 'removal (region, grid)')}`;
  const box = $('#resCardRows');
  if (box) {
    box.innerHTML = `<div class="rrow"><span>${g.cols}×${g.rows}</span><span>${pct.toFixed(2)}%</span>`
      + `<span class="rg ${gcls(grd)}">${grd}</span></div>`;
  }
}

async function runModelGrid(extra) {
  if (!S.id) { setStatus(T('先打开一张图', 'Open an image first')); return; }
  if (!S.rois.length) { setStatus(T('先在图上拖一个选区，把整片百格框住', 'Draw a region around the grid first')); return; }
  if (S.busy) { scheduleGm(200); return; }        // 分析还没跑完：排队，别丢
  setBusy(true);                            // 必须走 setBusy：直接改 S.busy 不会刷新「计算」按钮
  setStatus(T('正在铺格…', 'Laying out the grid…'));
  const rsig0 = gmRoiSig();
  const wantC = !S.gmColsManual && S.gmCountsSig !== rsig0;
  const wantR = !S.gmRowsManual && S.gmCountsSig !== rsig0;
  try {
    // 框完就自动分色：没有色样，或上次那对是自动分的而框已经挪了 → 重新分一对。
    // 人自己点过的颜色（samplesManual）永远不动——包括换框。
    if (!S.samples.length || (!S.samplesManual && S.gmAutoSig !== rsig0)) {
      if (await gmAutoSample(true)) S.gmAutoSig = rsig0;
    }
    const g = await post('api/grid', gmBody(extra));
    // 选区在请求途中被删/移动过 → 这次结果作废（否则删完选区，格网又被旧响应贴回来）
    if (rsig0 !== gmRoiSig()) return;
    // 新图默认刀宽是 1 px，格网里刀宽=扣掉的刀缝带：用实测缝宽当起点，再重算一次。
    // （格网判法不再跑像素分析，就没有人给刀宽兜底了 —— 1 px 会把整条缝都算成格内脱落。）
    if (!S.lineManual && (S.lineSrc === 'default' || S.lineSrc === 'keep')
        && g.ok && g.trace && g.trace.w_x > 1.5 && Math.abs(g.trace.w_x - S.line) > 0.5) {
      S.line = Math.max(1, Math.round(g.trace.w_x));
      markLineSrc('auto');
      if ($('#rngLine')) $('#rngLine').value = Math.min(25, S.line);
      if ($('#numLine')) $('#numLine').value = S.line;
      setStatus(T('刀宽按实测缝宽起算 ' + S.line + ' px（刀缝带扣掉的宽度）；偏小会把缝边算成脱落，可在 k 曲线里挑一档',
        `Blade width starts from the measured cut width, ${S.line} px — pick another row in the k curve if it is too narrow`));
      scheduleGm(30);
      return;
    }
    if (!g.ok) {
      S.gm = null;
      const msg = g.reason || T('铺格失败：先在图上框住整片百格，再填列数/行数', 'Grid failed — frame the whole grid first');
      $('#gmInfo').textContent = msg; setStatus(msg); draw(); return;
    }
    S.gm = g;
    // ── 格数 / 歪角写回输入框（自动认出来的，人改过就不再覆盖）─────────────────
    let autoNote = '';
    if (g.roles_flipped && !S.samplesManual && S.samples.length >= 2) {
      // 服务端按「格子内部谁占多数」替我们把漆膜/基材换回来了：本地色样跟着换，
      // 否则图上的色块/百分比和色样面板对不上。
      S.samples = S.samples.map(s => ({ rgb: s.rgb,
        role: s.role === 'coat' ? 'sub' : (s.role === 'sub' ? 'coat' : s.role) }));
      try { syncSampleUI(); } catch (_) {}
      autoNote = T('漆膜/基材已按格内面积对调（刀缝露白面积大时“面积大=漆膜”会判反）',
                   'Coating/substrate swapped by in-cell area (a wide exposed cut can invert the area rule)');
    }
    if (g.counts_est && (wantC || wantR)) {
      // 逐轴写回：只改没被手改过的那一边（用户只改行时，列照样跟着自动认）
      if (wantC) $('#gmCols').value = g.cols;
      if (wantR) $('#gmRows').value = g.rows;
      S.gmCountsSig = rsig0; updateGmLines();
      const which = (wantC && wantR) ? `${g.cols}×${g.rows}` : (wantC ? T('列 ' + g.cols, 'cols ' + g.cols) : T('行 ' + g.rows, 'rows ' + g.rows));
      const weak = (g.counts_est.conf !== 'trace');
      autoNote += (autoNote ? '　·　' : '') + (weak
        ? T(`格数按格距算到 ${which}（${Math.round(g.pitch_x)}×${Math.round(g.pitch_y)} px，刀缝没认全）；不对就改，改了就按你的`,
            `Cells from the measured pitch: ${which} (${Math.round(g.pitch_x)}×${Math.round(g.pitch_y)} px; cuts not fully traced) — edit if wrong`)
        : T(`格数自动认到 ${which}（格距 ${Math.round(g.pitch_x)}×${Math.round(g.pitch_y)} px）；不对就改，改了就按你的，重新框选才会再认`,
            `Cells auto-counted as ${which} (pitch ${Math.round(g.pitch_x)}×${Math.round(g.pitch_y)} px); edit if wrong — yours stays until you frame again`));
    } else if ((wantC || wantR) && !g.counts_est) {
      autoNote += (autoNote ? '　·　' : '') + T(`格数没认出来（周期不规整），先按 ${g.cols}×${g.rows} 铺；不对就把框收紧一点或改数字`,
                   `Could not count cells (irregular pitch) — using ${g.cols}×${g.rows}; tighten the frame or edit the numbers`);
    }
    if (!S.gmAngManual && typeof g.angle === 'number' && Math.abs(g.angle - (+$('#gmAng').value || 0)) > 0.05) {
      $('#gmAng').value = g.angle;
      autoNote += (autoNote ? '　·　' : '') + T(`歪角自动估到 ${g.angle}°`, `Tilt auto-estimated at ${g.angle}°`);
    } else if (!S.gmAngManual && g.angle_est != null && Math.abs(g.angle_est) < 0.2) {
      // 估出来就是正的：也要说一句，否则用户以为「定角没生效」
      autoNote += (autoNote ? '　·　' : '') + T('歪角估到 0°（格阵是正的，不用转）',
        'Tilt estimated 0° — the grid is already square');
    }
    // 刀宽还是没给（1 px = 新图默认）：刀缝带整条算成脱落，数一定偏大，得说清楚
    if (S.line <= 1 && !S.lineManual) {
      autoNote += (autoNote ? '　·　' : '') + T('刀宽还没量到，现在按 1 px 算，数会偏大——到量尺页量一条刀缝，或点下面 k 曲线挑一档',
        'No blade width measured yet (using 1 px), so numbers run high — measure a cut on the ruler tab, or pick a row in the k curve');
    }
    if (S.gmOff.dx.length !== Math.max(0, g.cols - 1) || S.gmOff.dy.length !== Math.max(0, g.rows - 1)) {
      S.gmOff.dx = new Array(Math.max(0, g.cols - 1)).fill(0);
      S.gmOff.dy = new Array(Math.max(0, g.rows - 1)).fill(0);
    }
    // 服务端回传的是「相对理想等分的总偏移」，吸附过之后必须同步回来，否则下次拖线会叠错
    if (g.line_dx) S.gmOff.dx = g.line_dx.slice();
    if (g.line_dy) S.gmOff.dy = g.line_dy.slice();
    S.gmSnap = !!(extra && extra.snap);          // 记录里存“这次是不是对过线”，重开时照原样复原
    if (g.overlay_url) loadOverlay(g.overlay_url);
    if (S.gmAuto) g.auto_samples = true;
    S.gmPitch = $('#gmPitch') ? (+$('#gmPitch').value || 0) : 0;   // 格距( mm )输入已撤，面积一律 px²
    $('#gmInfo').textContent = gmSummary(g) + (autoNote ? '　·　' + autoNote : '');
    if (S.exmode === 'grid') { try { setGridCard(g); } catch (_) {} }
    // 用 shift_*（最终位置比等分挪了多少）提示；line_* 现在只回显用户自己的拖拽量
    const snapD = [].concat(g.shift_dx || g.line_dx || [], g.shift_dy || g.line_dy || []).map(Math.abs);
    const snapMax = snapD.length ? Math.max(...snapD) : 0;
    const tr = g.trace || null;
    const trOk = tr && (tr.used_x || tr.used_y);
    const trNote = tr
      ? (trOk
        ? T(`刀缝追踪：竖 ${tr.n_x || 0} 条 / 横 ${tr.n_y || 0} 条（间距 ${Math.round(tr.pitch_x || 0)}×${Math.round(tr.pitch_y || 0)} px）`,
            `Seams traced: ${tr.n_x || 0} vertical / ${tr.n_y || 0} horizontal (pitch ${Math.round(tr.pitch_x || 0)}×${Math.round(tr.pitch_y || 0)} px)`)
        : T(`没认全刀缝（竖 ${tr.n_x || 0}/${S.gm.cols + 1}、横 ${tr.n_y || 0}/${S.gm.rows + 1}）→ 这一版先按等分铺`
            // 只在“几乎认全、只差一两条”时提示格数可能填错了；只认出 4 条那种是彻底没认出来，别误导
            + ((tr.n_x >= S.gm.cols - 1 && tr.n_x !== S.gm.cols + 1) ? `｜竖着认到 ${tr.n_x} 条缝，按它算应是 ${tr.n_x - 1} 列` : '')
            + ((tr.n_y >= S.gm.rows - 1 && tr.n_y !== S.gm.rows + 1) ? `、横着 ${tr.n_y} 条 → ${tr.n_y - 1} 行` : ''),
            `Could not trace all cuts (${tr.n_x || 0}/${S.gm.cols + 1} and ${tr.n_y || 0}/${S.gm.rows + 1}) — fell back to even division`))
      : '';
    // 刀宽比实测缝宽小太多时提醒：缝里露白的部分会被算进“格内脱落”（这张二值化图实测缝宽 24px，
    // 填 3px 会把大半个缝算成脱落，读数虚高）
    let wNote = '';
    if (tr && tr.w_x && S.gm && S.line > 0 && S.line < 0.6 * tr.w_x) {
      wNote = T(`｜刀缝实测宽 ${Math.round(tr.w_x)} px，你现在填 ${S.line} px，缝里露白的部分会被算成脱落`,
                ` | detected cut width ${Math.round(tr.w_x)} px vs your ${S.line} px`);
    }
    if (trNote || wNote) $('#gmTrace').textContent = trNote + wNote;
    setStatus((extra && extra.snap)
      ? (snapMax >= 1
        ? T('对线完成：线最多挪了 ' + snapMax.toFixed(1) + ' px；挪太多说明格数/框位不对',
            'Snapped — largest line shift ' + snapMax.toFixed(1) + ' px')
        : T('对线完成：线本来就贴住了，没挪（数字不变是对的）',
            'Snapped — nothing moved, the lines were already on the cuts'))
      : (trOk ? trNote : T('格网已铺好：橙带=你设的刀宽（扣掉不算），逐格是剩余漆膜；线不齐可直接拖那一根',
          'Grid laid out — orange = the blade width you excluded')));
    draw();
    // 格网判法下 k 曲线**不自动扫**：刀宽是用户自己给的参数，自动扫一遍既慢（4 秒）又容易被当成
    // “程序自己定了刀宽”。要看得点面板上那行「扫一遍 k」。
    try { updateKneeHint(); } catch (_) {}
  } catch (e) {
    setStatus(String(e.message || e));
  } finally {
    setBusy(false);
  }
}

async function gmAutoAngle() {
  if (!S.id || !S.rois.length) { setStatus(T('先框住整片百格', 'Frame the grid first')); return; }
  setStatus(T('正在估歪角…', 'Estimating tilt…'));
  try {
    const g = await post('api/grid', gmBody({ auto_angle: true, cols: 2, rows: 2 }));
    const a = (g && g.angle_est);
    if (a === undefined || a === null) { setStatus(T('估不出角度', 'Could not estimate')); return; }
    $('#gmAng').value = a;
    setStatus(T('估到歪角 ' + a + '°，可以再手调', 'Tilt ≈ ' + a + '° — adjust if needed'));
    await runModelGrid();
  } catch (e) { setStatus(String(e.message || e)); }
}

function gridSummary(g) {
  const n = g.cells.length;
  if (!n) return T('没找到格子', 'No cell found');
  const cols = new Set(g.cells.map(c => c.i)).size, rows = new Set(g.cells.map(c => c.j)).size;
  const worst = g.cells.reduce((a, b) => (b.pct > a.pct ? b : a));
  let s = T(`视场 ${cols} 列 × ${rows} 行 · 周期 ${Math.round(g.period_x)}px · 槽宽 ${Math.round(g.k_x)}px · 脱落 `,
            `field ${cols} × ${rows} · pitch ${Math.round(g.period_x)}px · groove ${Math.round(g.k_x)}px · removal `)
    + g.total.pct.toFixed(2) + '%';
  s += ' · ' + T('格内最多 ', 'worst cell ') + worst.pct.toFixed(1) + '%';
  return s;
}

async function runGrid() {
  if (!S.id) { setStatus(T('先打开一张图', 'Open an image first')); return; }
  if (!S.rois.length) { setStatus(T('先在图上拖一个选区，再自动找格子', 'Draw a region first')); return; }
  if (S.busy) return;
  setBusy(true);
  setStatus(T('正在找格子…', 'Fitting the grid…'));
  try {
    const g = await post('api/grid', {
      id: S.id, res: S.res, rois: S.rois.map(r => ({ box: r.box, ang: r.ang })),
      samples: S.samples.map(x => ({ rgb: x.rgb, role: x.role })),
      tol: S.tol, purity: S.purity, line_px: S.line, mode: S.exmode,
      loc_tile: 64, loc_factor: 1.6, roi_native: S.roiNative, inset: 0,
    });
    if (!g.ok) {
      S.grid = null;
      const msg = g.warn === 'grid.tilted'
        ? T(`照片歪了约 ${Math.abs(g.angle).toFixed(1)}°，先把图转正再找：工具-旋转`, `Image is tilted ~${Math.abs(g.angle).toFixed(1)}° — rotate it first (Tools · Rotate)`)
        : T('没找到规则的格线：确认选区盖住了整片格子，且刀缝/烧蚀线在画面里看得清',
            'No regular grid found: make sure the region covers the whole grid and the grooves are visible');
      $('#gridInfo').textContent = msg;
      setStatus(msg);
      draw();
      return;
    }
    S.grid = g;
    let info = gridSummary(g);
    if (S.purity < 0.70 && g.total.pct > 2) {
      // 亮面/高光多的涂层：白色高光会被算成露底，纯度提到 0.75 基本就干净了（实测 2.56% → 0.18%）
      info += ' · ' + T('高光可能被当成露底了：把「露底纯度」提到 0.75 再找一次',
                        'Highlights may read as bare substrate — raise “bare purity” to 0.75 and refit');
    }
    $('#gridInfo').textContent = info;
    setStatus(T('格网已铺好：每格单独算脱落率，可对照图上的百分比', 'Grid fitted — each cell is graded on its own'));
    draw();
  } catch (e) {
    setStatus(String(e.message || e));
  } finally {
    setBusy(false);
  }
}

/* --- 手绘三色笔：红=确认脱落，蓝=确认刀缝，黄=无法判断 --- */
const ANNOT_COLOR = { 1: 'rgba(255,70,70,.42)', 2: 'rgba(58,168,255,.42)', 3: 'rgba(255,205,60,.42)' };
const ANNOT_NAME = { 1: '确认脱落', 2: '确认刀缝', 3: '无法判断' };

function drawAnnotations() {
  const list = S.annot.strokes.length ? (S.annot.cur ? S.annot.strokes.concat([S.annot.cur]) : S.annot.strokes) : (S.annot.cur ? [S.annot.cur] : []);
  if (!list.length) return;
  const s = scale();
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (const k of list) {
    const col = ANNOT_COLOR[k.t] || 'rgba(255,255,255,.35)';
    const w = Math.max(1, k.r * 2 * s);
    const strokes = () => {
      ctx.beginPath();
      if (k.pts.length === 1) {
        const p = toScreen(k.pts[0][0], k.pts[0][1]);
        ctx.arc(p.x, p.y, Math.max(1, k.r * s), 0, Math.PI * 2);
      } else {
        for (let i = 0; i < k.pts.length; i++) {
          const p = toScreen(k.pts[i][0], k.pts[i][1]);
          if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
      }
      ctx.stroke();
    };
    // 先描一道浅白边，笔迹压在红/黄漆面上才不会糊成一片
    ctx.strokeStyle = 'rgba(255,255,255,.5)';
    ctx.lineWidth = w + 2.5;
    strokes();
    ctx.strokeStyle = col;
    ctx.lineWidth = w;
    strokes();
  }
  ctx.restore();
}

/* 逐块确认：在每块露底的中心画编号圆点，点一下就排除 */

/* 块轮廓：1 px 蚂蚁线（和 PS 选框一个意思）。
   用矢量路径描边，线宽永远是 1 屏幕像素——位图轮廓一放大就变成和像素一样粗，细不下去。 */
function drawBlockAnts() {
  const B = S.blocks;
  if (!B || !B.rois || !S.blocksOn) return;
  ctx.save();
  ctx.globalAlpha = (B.sig === blocksSig()) ? 1 : 0.32;   // 参数刚改、新块还没回来：淡一点，别让它看着像结果
  ctx.lineWidth = 1;
  ctx.lineJoin = 'round';
  ctx.setLineDash([4, 4]);
  B.rois.forEach((r, ri) => {
    r.blocks.forEach(b => {
      const ex = B.excluded[ri].has(b.i);
      // 外轮廓 + 每个孔：只画外圈的话，成片脱落里那些“没掉干净的岛”一个都不描（用户说的"覆盖不到洞"）
      const polys = [b.outline].concat(b.holes || []);
      polys.forEach(ol => {
      if (!ol || ol.length < 2) return;
      const sp = ol.map(p => toScreen(p[0], p[1]));
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const q of sp) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
      ctx.beginPath();
      if (x1 - x0 < 9 || y1 - y0 < 9) {
        // 小块/细条在屏幕上只有几个像素：4-4 的虚线根本画不出来，看着就像“没有蚂蚁线”。
        // 这种情况至少给一个 9 屏幕像素的方框，保证眼睛能看到它在哪。
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        const hw = Math.max(9, x1 - x0) / 2, hh = Math.max(9, y1 - y0) / 2;
        ctx.rect(cx - hw, cy - hh, hw * 2, hh * 2);
      } else {
        sp.forEach((q, k) => { if (k) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y); });
        ctx.closePath();
      }
      // 黑一段 + 白一段交替：任何底色上都读得出，还能看出“在动”
      ctx.strokeStyle = ex ? 'rgba(120,120,128,.85)' : 'rgba(0,0,0,.88)';
      ctx.lineDashOffset = -S.ants;
      ctx.stroke();
      ctx.strokeStyle = ex ? 'rgba(200,200,206,.55)' : 'rgba(255,255,255,.95)';
      ctx.lineDashOffset = -S.ants + 4;
      ctx.stroke();
      });                                    // polys.forEach
    });
  });
  ctx.restore();
}

let antsTimer = null;
function startAnts() {
  if (antsTimer) return;
  let last = 0;
  const tick = (t) => {
    if (t - last > 70) {                       // ~14 fps 足够，别把主线程占满
      last = t;
      if (S.blocks && S.blocks.rois && S.blocksOn) { S.ants = (S.ants + 1) % 8; draw(); }
    }
    antsTimer = requestAnimationFrame(tick);
  };
  antsTimer = requestAnimationFrame(tick);
}

function drawBlocks() {
  const B = S.blocks;
  if (!B || !B.rois || !S.blocksOn) return;   // 换判法/换图时会置空，别当成有块
  ctx.save();
  ctx.globalAlpha = (B.sig === blocksSig()) ? 1 : 0.32;
  ctx.font = '600 10px ui-monospace,Menlo,Consolas,monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // 轮廓由服务端的 contour 层画（细线只描边），这里只标块号：
  // 黑描边 + 白字，任何底色都读得出，又不遮住块内的像素。
  B.rois.forEach((r, ri) => {
    r.blocks.forEach(b => {
      const ex = B.excluded[ri].has(b.i);
      const p = toScreen(b.cx, b.cy);
      if (p.x < -24 || p.y < -24 || p.x > S.cw + 24 || p.y > S.ch + 24) return;
      const s = String(b.i + 1);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,.72)';
      ctx.strokeText(s, p.x, p.y + 0.5);
      ctx.fillStyle = ex ? 'rgba(215,215,220,.95)' : '#fff';
      ctx.fillText(s, p.x, p.y + 0.5);
      if (ex) {
        ctx.beginPath();
        ctx.moveTo(p.x - 8, p.y - 8); ctx.lineTo(p.x + 8, p.y + 8);
        ctx.moveTo(p.x + 8, p.y - 8); ctx.lineTo(p.x - 8, p.y + 8);
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = 'rgba(215,215,220,.85)';
        ctx.stroke();
      }
    });
  });
  ctx.restore();
}

function blockHit(ex, ey) {
  const B = S.blocks;
  if (!B) return null;
  let best = null, bd = 16;
  B.rois.forEach((r, ri) => {
    r.blocks.forEach(b => {
      const p = toScreen(b.cx, b.cy);
      const d = Math.hypot(p.x - ex, p.y - ey);
      if (d < bd) { bd = d; best = { ri, i: b.i }; }
    });
  });
  return best;
}

function drawRois() {
  const boxes = [], texts = [];
  S.rois.forEach((r, i) => {
    const cps = cornersOf(r).map(p => toScreen(p.x, p.y));
    const sel = i === S.sel;
    ctx.beginPath();
    ctx.moveTo(cps[0].x, cps[0].y);
    for (let k = 1; k < 4; k++) ctx.lineTo(cps[k].x, cps[k].y);
    ctx.closePath();
    // 先粗黑底、再亮线：黄漆膜、蓝漆膜、白纸上都能看清
    ctx.save();
    ctx.setLineDash([]);
    ctx.lineWidth = sel ? 6.5 : 5;
    ctx.strokeStyle = 'rgba(0,0,0,.62)';
    ctx.stroke();
    ctx.setLineDash(sel ? [] : [8, 5]);        // 选中=实线，未选中=虚线（不靠颜色区分）
    ctx.lineWidth = sel ? 2.8 : 2;
    ctx.strokeStyle = sel ? '#ffd400' : '#ffffff';
    ctx.stroke();
    ctx.restore();
    if (sel) {
      const sq = (x, y, s) => {
        ctx.fillStyle = 'rgba(0,0,0,.75)'; ctx.fillRect(x - s - 1.5, y - s - 1.5, (s + 1.5) * 2, (s + 1.5) * 2);
        ctx.fillStyle = '#ffd400'; ctx.fillRect(x - s, y - s, s * 2, s * 2);
      };
      for (const p of cps) sq(p.x, p.y, 3.5);
      const defs = handleDefs(r);
      for (const d of defs) {
        if (d.sx && d.sy) continue;            // 四角已画
        const p = toScreen(d.p.x, d.p.y);
        sq(p.x, p.y, 3);
      }
      const rp = rotateHandlePoint(r);
      ctx.beginPath(); ctx.arc(rp.x, rp.y, 7.5, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,0,0,.65)'; ctx.lineWidth = 3.5; ctx.stroke();
      ctx.beginPath(); ctx.arc(rp.x, rp.y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 1.8; ctx.stroke();
      ctx.beginPath(); ctx.arc(rp.x, rp.y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffd400'; ctx.fill();
    }
    const xs = cps.map(p => p.x), ys = cps.map(p => p.y);
    const bb = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    boxes.push(bb);
    let t = 'R' + (i + 1);
    if (S.rows && S.rows.length === S.rois.length) {
      t += '  ' + fmt(S.rows[i].pct, 1) + '%  ' + S.rows[i].astm;
      if (Math.abs(r.ang) > 0.5) t += '  ' + (r.ang > 0 ? '+' : '') + Math.round(r.ang) + '°';
    }
    texts.push(t);
  });
  if (!boxes.length) return;
  const placed = placeLabels(boxes, texts);
  ctx.font = '600 12px ui-monospace,Menlo,Consolas,monospace';
  ctx.textBaseline = 'middle';
  placed.forEach((p, i) => {
    ctx.fillStyle = 'rgba(12,13,16,.88)';
    ctx.strokeStyle = '#7a7a7a';
    ctx.lineWidth = 1;
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.strokeRect(p.x + .5, p.y + .5, p.w - 1, p.h - 1);
    ctx.fillStyle = '#ffd54a';
    ctx.fillText(texts[i], p.x + 6, p.y + p.h / 2);
  });
}

function placeLabels(boxes, texts) {
  ctx.font = '600 12px ui-monospace,Menlo,Consolas,monospace';
  const placed = [], out = [];
  const h = 18, pad = 4;
  boxes.forEach((b, i) => {
    const w = ctx.measureText(texts[i]).width + 12;
    const cands = [
      [b[0], b[1] - h - pad], [b[2] - w, b[1] - h - pad],
      [b[0], b[3] + pad], [b[2] - w, b[3] + pad],
      [b[2] + pad, b[1]], [b[0] - w - pad, b[1]],
      [b[2] + pad, b[3] - h], [b[0] - w - pad, b[3] - h],
    ];
    let best = null;
    for (let [x, y] of cands) {
      x = clamp(x, 1, S.cw - w - 1); y = clamp(y, 1, S.ch - h - 1);
      const r = [x, y, x + w, y + h];
      const hit = placed.some(q => !(r[2] < q[0] || r[0] > q[2] || r[3] < q[1] || r[1] > q[3]));
      if (!hit) { best = r; break; }
    }
    if (!best) {
      let [x, y] = cands[0];
      x = clamp(x, 1, S.cw - w - 1); y = clamp(y, 1, S.ch - h - 1);
      best = [x, y, x + w, y + h];
    }
    placed.push(best);
    out.push({ x: best[0], y: best[1], w, h });
  });
  return out;
}

/* 量完的线放 6 秒就自己隐去（数值、均值、用测量值都还在） */
function rulerFadeSoon() {
  clearTimeout(S.rulerFadeT);
  S.rulerFadeT = setTimeout(() => {
    S.rulerDone = null; S.rulerSeg = null;     // 不管当前在哪个页签，到点就收线（数值仍保留）
    draw();
  }, 6000);
}

function drawRuler() {
  for (const [seg, color] of [[S.rulerDone, '#ff66ff'], [S.rulerSeg, '#ffffff']]) {
    if (!seg) continue;
    const a = toScreen(seg[0], seg[1]), b = toScreen(seg[2], seg[3]);
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(a.x - 4, a.y); ctx.lineTo(a.x + 4, a.y);
    ctx.moveTo(b.x - 4, b.y); ctx.lineTo(b.x + 4, b.y);
    ctx.stroke();
  }
  if (S.rulerDone && S.rulerLast) {
    const a = toScreen(S.rulerDone[0], S.rulerDone[1]);
    const b = toScreen(S.rulerDone[2], S.rulerDone[3]);
    const txt = fmt(S.rulerLast.width_full, 1) + T(' px 原图', ' px original');
    ctx.font = '600 12px ui-monospace,Menlo,Consolas,monospace';
    const w = ctx.measureText(txt).width + 10;
    const x = (a.x + b.x) / 2 + 8, y = (a.y + b.y) / 2 - 22;
    ctx.fillStyle = 'rgba(12,13,16,.88)'; ctx.fillRect(x, y, w, 18);
    ctx.fillStyle = '#ff9cff'; ctx.textBaseline = 'middle';
    ctx.fillText(txt, x + 5, y + 9);
  }
}

/* ------------------------------------------------------------------ 命中测试 */
function poi(r, x, y) {
  // 图像坐标 x,y 是否在（可旋转的）框内
  const c = boxCenter(r);
  const l = unrot({ x, y }, c, r.ang || 0);
  const hw = (r.box[2] - r.box[0]) / 2, hh = (r.box[3] - r.box[1]) / 2;
  return Math.abs(l.x - c.x) <= hw && Math.abs(l.y - c.y) <= hh;
}

function hitAt(ex, ey) {
  const p = toImg(ex, ey);
  // 1) 选中框的旋转手柄 / 缩放手柄（屏幕坐标，容差 8px）
  if (S.sel >= 0 && S.rois[S.sel]) {
    const r = S.rois[S.sel];
    const rp = rotateHandlePoint(r);
    if (Math.hypot(rp.x - ex, rp.y - ey) <= 9) return { i: S.sel, part: 'rotate' };
    for (const d of handleDefs(r)) {
      const q = toScreen(d.p.x, d.p.y);
      if (Math.abs(q.x - ex) <= 7 && Math.abs(q.y - ey) <= 7) {
        return { i: S.sel, part: 'resize', sx: d.sx, sy: d.sy };
      }
    }
  }
  // 2) 内部 → 平移（从上层往下找）
  for (let i = S.rois.length - 1; i >= 0; i--) {
    if (poi(S.rois[i], p.x, p.y)) return { i, part: 'move' };
  }
  return null;
}

/* ------------------------------------------------------------------ 指针交互 */
let spaceDown = false;

/* 双指缩放/平移：Pointer Events 在 iOS/Android 都是一套，不需要额外手势库 */
const ptrs = new Map();
let pinch = null;
/* 单指“点”的候选：手指落下时先不执行动作，等抬手确认这是点、不是双指缩放或拖动。
   手机上双指缩放时第一根手指已经落在图上，立刻响应就等于把“缩放”当成“点了块/取了色”。 */
let tapPending = null;
let pinchSeen = false;      // 这次手势里出现过双指
const TAP_SLOP = 8;         // 抬手时位移超过这么多像素就不算“点”

function pinchStart() {
  pinchSeen = true;
  tapPending = null;                                   // 双指了，任何“点”都作废
  if (S.dragSnap) {                                    // 第一根手指可能已经把选区挪了一点，退回去
    S.rois = S.dragSnap; S.dragSnap = null;
    refreshRoiList(); syncRoiAngle(); draw();
  }
  S.annot.cur = null;                                  // 注释笔画也别留在半途
  if (S.rulerSeg) { S.rulerSeg = null; draw(); }
  if (debouncedAnalyze && debouncedAnalyze.cancel) debouncedAnalyze.cancel();
  const p = [...ptrs.values()];
  if (p.length < 2) return;
  const mid = { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
  pinch = {
    d0: Math.max(1, Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y)),
    zoom0: S.zoom,
    anchor: toImg(mid.x, mid.y),
  };
  S.drag = null; pinch._seen = true;
}
function pinchMove() {
  const p = [...ptrs.values()];
  if (p.length < 2 || !pinch) return;
  const mid = { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
  const d = Math.max(1, Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y));
  S.zoom = clamp(pinch.zoom0 * (d / pinch.d0), 0.05, 40);
  S.zoomMode = 'free';
  const s = scale();
  S.panX = mid.x - pinch.anchor.x * s;
  S.panY = mid.y - pinch.anchor.y * s;
  clampPan();
  drawSoon();
}

cv.addEventListener('pointerdown', e => {
  if (!S.img) return;
  e.preventDefault();                       // 关键：阻止浏览器默认的文本选择/图片拖拽
  document.body.classList.add('dragging');
  cv.setPointerCapture(e.pointerId);
  const { x: ex, y: ey } = evPos(e);
  ptrs.set(e.pointerId, { x: ex, y: ey });
  if (ptrs.size === 2) { pinchStart(); return; }
  if (e.button === 1 || spaceDown || (e.button === 0 && e.altKey)) {
    S.drag = { kind: 'pan', ex, ey, px: S.panX, py: S.panY };
    cv.style.cursor = 'grabbing';
    return;
  }
  if (e.button !== 0) return;

  if (S.mode === 'blocks') {
    const hit = blockHit(ex, ey);
    if (hit) tapPending = { kind: 'block', ri: hit.ri, i: hit.i, x: ex, y: ey };
    return;                        // 点在空白处什么都不做，免得误拖出选区
  }
  if (S.mode === 'annot') {
    const p = toImg(ex, ey);
    if (p.x < 0 || p.y < 0 || p.x >= S.w || p.y >= S.h) return;
    S.annot.cur = { t: S.annot.tool, r: S.annot.r, pts: [[p.x, p.y]] };
    S.drag = { kind: 'annot' };
    draw();
    return;
  }
  if (S.mode === 'pick') { tapPending = { kind: 'pick', x: ex, y: ey }; return; }
  // 模型格网：只在「量尺」页才抓线/整体拖。框选页要能安心编辑选区（以前刀缝会抢点击），
  // 量尺页则相反——只跟量尺和格网打交道。
  if (S.mode === 'ruler' && S.gm) {
    const gh = gmHitAt(ex, ey);
    if (gh) {
      const pi = toImg(ex, ey);
      S.drag = gh.move
        ? { kind: 'gmmove', p0x: pi.x, p0y: pi.y }
        : { kind: 'gmline', axis: gh.axis, idx: gh.idx, n: gh.n, p0x: pi.x, p0y: pi.y, off: 0 };
      cv.style.cursor = 'move';
      return;
    }
  }
  if (S.mode === 'ruler') {
    const p = toImg(ex, ey);
    S.drag = { kind: 'ruler', a: p, b: p };
    clearTimeout(S.rulerFadeT);
    S.rulerSeg = [p.x, p.y, p.x, p.y];
    return;
  }
  if (S.mode !== 'rect') return;          // 其余页签不碰选区
  // rect 模式
  const hit = hitAt(ex, ey);
  const p = toImg(ex, ey);
  if (hit && hit.part === 'rotate') {
    S.dragSnap = JSON.parse(JSON.stringify(S.rois));
    const r = S.rois[hit.i], c = boxCenter(r);
    const cs = toScreen(c.x, c.y);
    S.drag = {
      kind: 'rotate', i: hit.i, p0: p,
      startPointer: Math.atan2(ey - cs.y, ex - cs.x) * 180 / Math.PI,
      startAng: r.ang || 0,
    };
    return;
  }
  if (hit && hit.part === 'resize') {
    S.dragSnap = JSON.parse(JSON.stringify(S.rois));
    const r = S.rois[hit.i];
    S.drag = {
      kind: 'resize', i: hit.i, sx: hit.sx, sy: hit.sy,
      p0: p, c0: boxCenter(r), ang: r.ang || 0,
      hw0: (r.box[2] - r.box[0]) / 2, hh0: (r.box[3] - r.box[1]) / 2, moved: false,
    };
    return;
  }
  if (hit && hit.part === 'move') {
    S.dragSnap = JSON.parse(JSON.stringify(S.rois));
    S.sel = hit.i;
    S.drag = { kind: 'move', i: hit.i, p0: p, box0: S.rois[hit.i].box.slice(), moved: false };
    refreshRoiList(); syncRoiAngle(); draw();
    return;
  }
  S.drag = { kind: 'new', a: p, b: p };
  S.sel = -1;
  draw();
});

cv.addEventListener('pointermove', e => {
  if (!S.img) return;
  const { x: ex, y: ey } = evPos(e);
  if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, { x: ex, y: ey });
  if (pinch && ptrs.size >= 2) { e.preventDefault(); pinchMove(); return; }
  if (tapPending && Math.hypot(ex - tapPending.x, ey - tapPending.y) > TAP_SLOP) tapPending = null;
  if (!S.drag) {
    if (S.mode === 'pan' || spaceDown) cv.style.cursor = 'grab';
    else if (S.mode === 'pick') cv.style.cursor = 'crosshair';
    else if (S.mode === 'ruler') cv.style.cursor = 'cell';
    else if (S.mode === 'annot') cv.style.cursor = 'crosshair';
    else if (S.mode === 'blocks') cv.style.cursor = blockHit(ex, ey) ? 'pointer' : 'default';
    else {
      const hit = hitAt(ex, ey);
      cv.style.cursor = !hit ? 'crosshair'
        : hit.part === 'move' ? 'move'
          : hit.part === 'rotate' ? 'alias' : 'nwse-resize';
    }
    return;
  }
  const d = S.drag;
  if (d.kind === 'pan') { S.panX = d.px + (ex - d.ex); S.panY = d.py + (ey - d.ey); clampPan(); drawSoon(); return; }
  const p = toImg(ex, ey);

  if (d.kind === 'gmline') {
    const off = (p.x - d.p0x) * d.n[0] + (p.y - d.p0y) * d.n[1];   // 沿这根线的法线投位移
    d.off = off;
    S.gmDrag = { kind: 'line', axis: d.axis, idx: d.idx, ldx: off * d.n[0], ldy: off * d.n[1] };
    drawSoon(); return;
  }
  if (d.kind === 'gmmove') {
    S.gmDrag = { kind: 'move', gdx: p.x - d.p0x, gdy: p.y - d.p0y };
    drawSoon(); return;
  }

  if (d.kind === 'annot') {
    const c = S.annot.cur;
    if (!c) { S.drag = null; return; }
    const last = c.pts[c.pts.length - 1];
    if (Math.hypot(p.x - last[0], p.y - last[1]) >= Math.max(1, c.r * 0.3)) {
      c.pts.push([Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10]);
      drawSoon();
    }
    return;
  }

  if (d.kind === 'new') { d.b = p; drawSoon(); return; }

  if (d.kind === 'ruler') {
    d.b = p;
    S.rulerSeg = [d.a.x, d.a.y, p.x, p.y];
    drawSoon(); return;
  }

  if (d.kind === 'move') {
    const r = S.rois[d.i];
    if (!r) { S.drag = null; return; }
    /* 位移一律相对“按下那一刻”算，并且先过 4 屏幕像素的死区：
       手指轻点选区的抖动（1~3 px）不该被当成拖动，否则一碰框就重算。 */
    const dx = p.x - d.p0.x, dy = p.y - d.p0.y;
    if (!d.moved && Math.hypot(dx, dy) * scale() < 4) return;
    d.moved = true;
    const bw = d.box0[2] - d.box0[0], bh = d.box0[3] - d.box0[1];
    const x0 = clamp(d.box0[0] + dx, 0, S.w - bw), y0 = clamp(d.box0[1] + dy, 0, S.h - bh);
    r.box = [x0, y0, x0 + bw, y0 + bh];
    drawSoon(); return;
  }

  if (d.kind === 'rotate') {
    const r = S.rois[d.i];
    if (!r) { S.drag = null; return; }
    if (!d.moved && Math.hypot((p.x - d.p0.x) * scale(), (p.y - d.p0.y) * scale()) < 4) return;
    const c = boxCenter(r), cs = toScreen(c.x, c.y);
    const cur = Math.atan2(ey - cs.y, ex - cs.x) * 180 / Math.PI;
    let ang = d.startAng + (cur - d.startPointer);
    if (e.shiftKey) ang = Math.round(ang / 5) * 5;
    r.ang = Math.round(ang * 10) / 10;
    d.moved = true;
    syncRoiAngle(); drawSoon(); return;      // 侧栏等松手后统一刷新，避免每帧重排
  }

  if (d.kind === 'resize') {
    const r = S.rois[d.i];
    if (!r) { S.drag = null; return; }
    if (!d.moved && Math.hypot((p.x - d.p0.x) * scale(), (p.y - d.p0.y) * scale()) < 4) return;
    const ang = d.ang;
    /* 指针位移换算到框的局部坐标系（相对按下那一刻），永远从初始状态重算：
       ① 抓住手柄时不会把边缘“吸附”到指点处（不再有起步一跳）
       ② 不会累积误差、不会来回振荡 */
    const lp = unrot(p, d.c0, ang), lp0 = unrot(d.p0, d.c0, ang);
    const dlx = lp.x - lp0.x, dly = lp.y - lp0.y;
    /* 关键：拖动的是「一条边」，而半宽 = (另一条边到这条边的距离)/2，
       所以半宽只增加位移的一半、中心也只移一半；否则边会以两倍速度跑开（看起来就是跳） */
    const hw = d.sx ? Math.max(3, d.hw0 + d.sx * dlx / 2) : d.hw0;
    const hh = d.sy ? Math.max(3, d.hh0 + d.sy * dly / 2) : d.hh0;
    const t = ang * Math.PI / 180, cs = Math.cos(t), sn = Math.sin(t);
    const mvx = d.sx ? (hw - d.hw0) * d.sx : 0, mvy = d.sy ? (hh - d.hh0) * d.sy : 0;
    const c1 = { x: d.c0.x + mvx * cs - mvy * sn, y: d.c0.y + mvx * sn + mvy * cs };
    r.box = [c1.x - hw, c1.y - hh, c1.x + hw, c1.y + hh];
    d.moved = true;
    drawSoon(); return;
  }
});

cv.addEventListener('pointerup', e => {
  ptrs.delete(e.pointerId);
  if (ptrs.size < 2) pinch = null;
  const wasPinch = pinchSeen;
  if (ptrs.size === 0) pinchSeen = false;
  S.dragSnap = null;
  /* 抬手才落地“点”：双指缩放结束后第一根手指也会走这里，所以必须看 wasPinch */
  if (tapPending) {
    const t = tapPending; tapPending = null;
    if (!wasPinch) {
      if (t.kind === 'block') toggleBlock(t.ri, t.i);
      else if (t.kind === 'pick') { pickAt(t.x, t.y); return; }
    }
  }
  const d = S.drag;
  S.drag = null;
  document.body.classList.remove('dragging');
  if (resizePending) { resizePending = null; resizeCanvas(); }
  cv.style.cursor = 'crosshair';
  if (!d) { draw(); return; }
  if (d.kind === 'pan') { draw(); return; }
  if (d.kind === 'gmline') {
    S.gmDrag = null;
    if (!d.off) { draw(); return; }
    if (d.axis === 'x') S.gmOff.dx[d.idx - 1] = (S.gmOff.dx[d.idx - 1] || 0) + d.off;
    else S.gmOff.dy[d.idx - 1] = (S.gmOff.dy[d.idx - 1] || 0) + d.off;
    runModelGrid(); return;
  }
  if (d.kind === 'gmmove') {
    const dd = S.gmDrag || { gdx: 0, gdy: 0 };
    S.gmDrag = null;
    if (!dd.gdx && !dd.gdy) { draw(); return; }
    S.gmOff.ox += dd.gdx; S.gmOff.oy += dd.gdy;
    runModelGrid(); return;
  }
  if (d.kind === 'annot') {
    const c = S.annot.cur;
    S.annot.cur = null;
    if (c && c.pts.length) { S.annot.strokes.push(c); debouncedAnnot(); }
    draw();
    return;
  }
  if (d.kind === 'new') {
    const a = d.a, b = d.b;
    const x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y);
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    if (x1 - x0 < 6 || y1 - y0 < 6) {
      const q = evPos(e);
      const hit = hitAt(q.x, q.y);
      S.sel = hit ? hit.i : -1;
      refreshRoiList(); syncRoiAngle(); draw();
      return;
    }
    S.rois.push({ box: [x0, y0, x1, y1], ang: 0 });
    S.sel = S.rois.length - 1;
    // 重新框选 = 换了一片样品：行列/歪角的手改标记作废，让程序按新框再认一次
    // （拖已有框的边只算微调，不清标记——否则刚手填的列数会被自己挪一下就覆盖掉）
    S.gmColsManual = false; S.gmRowsManual = false; S.gmAngManual = false;
    S.gmCountsSig = ''; S.gmAutoSig = '';
    dropOverlay();                              // 旧覆盖层是按上一组选区算的
    refreshRoiList(); syncRoiAngle(); draw();
    if (S.exmode === 'grid') scheduleGm();            // 判法=格网：只跑格网，不跑像素分析/扫 k
    else if (S.semi) scheduleSemi();
    else analyze();                                    // 连续拖框时合并成一次计算
    return;
  }
  if (d.kind === 'ruler') {
    const a = d.a, b = d.b;
    S.rulerSeg = null;
    if (Math.hypot(b.x - a.x, b.y - a.y) >= 4) measureGroove(a, b);
    draw(); return;
  }
  if (d.moved) {
    refreshRoiList();
    if (S.exmode === 'grid') scheduleGm(); else { analyze(); scheduleGm(); }
  }
  draw();
});

cv.addEventListener('dblclick', () => fitView());
cv.addEventListener('pointercancel', e => {
  if (e && e.pointerId != null) ptrs.delete(e.pointerId);
  if (ptrs.size < 2) pinch = null;
  if (ptrs.size === 0) pinchSeen = false;
  tapPending = null; S.dragSnap = null;
  S.drag = null; document.body.classList.remove('dragging');
  if (resizePending) { resizePending = null; resizeCanvas(); }
});
cv.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (!S.img) return;
  const q = evPos(e), p = toImg(q.x, q.y);
  for (let i = S.rois.length - 1; i >= 0; i--) {
    if (poi(S.rois[i], p.x, p.y)) { deleteRoi(i); return; }
  }
});
cv.addEventListener('wheel', e => {
  if (!S.img) return;
  e.preventDefault();
  const q = evPos(e);
  zoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, q.x, q.y);
}, { passive: false });

window.addEventListener('keydown', e => {
  if (e.code === 'Space') { spaceDown = true; }
  if (e.target.matches('input,select,textarea')) return;
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (S.sel >= 0) { e.preventDefault(); deleteRoi(S.sel); }
  } else if (e.key.startsWith('Arrow')) {
    if (S.sel < 0) return;
    e.preventDefault();
    const step = e.shiftKey ? 10 : 1;
    const r = S.rois[S.sel];
    const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0;
    const dy = e.key === 'ArrowUp' ? -step : e.key === 'ArrowDown' ? step : 0;
    const bw = r.box[2] - r.box[0], bh = r.box[3] - r.box[1];
    const x0 = clamp(r.box[0] + dx, 0, S.w - bw), y0 = clamp(r.box[1] + dy, 0, S.h - bh);
    r.box = [x0, y0, x0 + bw, y0 + bh];
    dropOverlay();
    draw();
    debouncedAnalyze();
  } else if ((e.ctrlKey || e.metaKey) && e.key === '0') { e.preventDefault(); fitView(); }
  else if ((e.ctrlKey || e.metaKey) && e.key === '1') { e.preventDefault(); zoomOne(); }
});
window.addEventListener('keyup', e => { if (e.code === 'Space') spaceDown = false; });

/* 画布尺寸跟着容器走：侧栏内容变化（列表变长、滚动条出现）时位图不会和 CSS 尺寸脱节，
   否则“画布逻辑像素 ≠ 屏幕像素”，拖动就会一格一格地跳 */
if (window.ResizeObserver) {
  const ro = new ResizeObserver(() => {
    const r = wrap.getBoundingClientRect();
    if (Math.abs(Math.round(r.width) - S.cw) > 0.5 || Math.abs(Math.round(r.height) - S.ch) > 0.5) {
      if (S.drag) resizePending = true; else scheduleResize();
    }
  });
  ro.observe(wrap);
}
window.addEventListener('resize', scheduleResize);

/* ------------------------------------------------------------------ 图片装载 */
function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => { S.img = im; res(im); };
    im.onerror = () => rej(new Error(T('图片加载失败', 'Failed to load the image')));
    im.src = url;
  });
}

/* 带字节进度的图片加载（打开记录时用：WAN 上要几秒，得让人看见在动） */
async function loadImageProgress(url, onProg, signal) {
  const resp = await fetch(url, { signal, cache: 'force-cache' });
  if (!resp.ok) throw new Error('HTTP ' + resp.status);
  const total = +(resp.headers.get('content-length') || 0);
  const ct = resp.headers.get('content-type') || 'image/jpeg';
  if (!resp.body || !resp.body.getReader) {
    const b = await resp.blob();
    return await loadImage(URL.createObjectURL(b));
  }
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (onProg) onProg(got, total);
  }
  return await loadImage(URL.createObjectURL(new Blob(chunks, { type: ct })));
}

/* ---------------- 上传进度 ---------------- */
function upShow(title) {
  const w = $('#upWrap');
  if (!w) return;
  $('#upTitle').textContent = title;
  upSet(0, T('准备中…', 'Preparing…'));
  w.classList.remove('hide');
}
function upHide() { const w = $('#upWrap'); if (w) w.classList.add('hide'); }

/* 不知道还要多久的阶段，让进度条慢慢爬，别停在原地像卡死 */
function creepStart(from, to, sub, ms) {
  let v = from;
  upSet(v, sub);
  const t = setInterval(() => { v = Math.min(to, v + 1.2); upSet(v, sub); }, ms || 400);
  return () => clearInterval(t);
}
function upSet(pct, sub) {
  const f = $('#upFill');
  if (f) f.style.width = Math.max(0, Math.min(100, pct)) + '%';
  const s = $('#upSub');
  if (s) s.textContent = sub;
}

let upXhr = null;          // 当前上传请求，供右上角 × 取消
document.addEventListener('DOMContentLoaded', () => {
  startAnts();
  // 分页用法说明：× 关掉后记住（localStorage），下次进来不再显示
  [['guideRuler', 'guideRulerX', 'baige.guide.ruler'],
   ['guideAnnot', 'guideAnnotX', 'baige.guide.annot']].forEach(([boxId, btnId, key]) => {
    const box = $('#' + boxId), btn = $('#' + btnId);
    let off = false;
    try { off = localStorage.getItem(key) === '1'; } catch (_) { off = false; }
    if (box && off) box.classList.add('hide');
    if (btn) btn.onclick = e => {
      e.preventDefault(); e.stopPropagation();
      if (box) box.classList.add('hide');
      try { localStorage.setItem(key, '1'); } catch (_) {}
    };
  });
  const x = $('#upCancel');
  if (x) x.onclick = e => {
    e.preventDefault(); e.stopPropagation();
    if (upXhr) { setStatus(T('取消中…', 'Cancelling…')); upXhr.abort(); }
    else if (S.openAbort) { try { S.openAbort.abort(); } catch (_) {} S.openAbort = null; upHide(); setStatus(T('已取消', 'Cancelled')); }
  };
});

function uploadXHR(fd, onProg) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    upXhr = xhr;
    const t0 = performance.now();
    xhr.open('POST', 'api/upload');
    xhr.setRequestHeader('X-Baige-Lang', I18N.lang);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch (e) { reject(new Error(T('响应解析失败', 'Malformed server response'))); }
      } else {
        reject(new Error((xhr.responseText || ('HTTP ' + xhr.status)).slice(0, 200)));
      }
    };
    xhr.onerror = () => reject(new Error(T('网络中断（检查网络或代理）', 'Network interrupted (check your connection or proxy)')));
    xhr.onabort = () => reject(new Error(T('已取消', 'Cancelled')));
    xhr.onloadend = () => { if (upXhr === xhr) upXhr = null; };
    if (xhr.upload) {
      xhr.upload.onprogress = e => {
        if (!e.lengthComputable) return;
        onProg(e.loaded, e.total, performance.now() - t0);
      };
    }
    xhr.send(fd);
  });
}

/* 外层反代对 >1 MB 的请求体直接 500（nginx+oauth2 那层），所以大图走分片：
   切成 400 KB 小块逐个 POST，服务端拼回同一份字节（sha1 不变，判重口径不受影响）。 */
const CHUNK_BYTES = 400 * 1024;
const CHUNK_OVER = 900 * 1024;          // 超过这个大小就改走分片

function chunkXHR(url, body, isJSON) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    upXhr = xhr;
    xhr.open('POST', url);
    if (isJSON) xhr.setRequestHeader('Content-Type', 'application/json');
    else xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText || '{}')); }
        catch (e) { reject(new Error(T('响应解析失败', 'Malformed server response'))); }
      } else {
        reject(new Error((xhr.responseText || ('HTTP ' + xhr.status)).slice(0, 200)));
      }
    };
    xhr.onerror = () => reject(new Error(T('网络中断（检查网络或代理）', 'Network interrupted (check your connection or proxy)')));
    xhr.onabort = () => reject(new Error(T('已取消', 'Cancelled')));
    xhr.onloadend = () => { if (upXhr === xhr) upXhr = null; };
    xhr.send(body);
  });
}

async function uploadChunked(file, onProg) {
  const uid = 'u' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const n = Math.max(1, Math.ceil(file.size / CHUNK_BYTES));
  const t0 = performance.now();
  let sent = 0;
  for (let i = 0; i < n; i++) {
    const blob = file.slice(i * CHUNK_BYTES, Math.min(file.size, (i + 1) * CHUNK_BYTES));
    await chunkXHR(`api/upload/chunk?uid=${encodeURIComponent(uid)}&idx=${i}&total=${n}`
                   + `&name=${encodeURIComponent(file.name || '')}`, blob, false);
    sent += blob.size;
    onProg(sent, file.size, performance.now() - t0, i + 1, n);
  }
  return chunkXHR('api/upload/finish', JSON.stringify({ uid, name: file.name || '' }), true);
}

async function upload(file) {
  if (!file) return;
  setStatus(t('res.uploading'));
  const fd = new FormData();
  fd.append('file', file);
  const mb = (file.size || 0) / 1e6;
  upShow(T(`上传 ${file.name || '图片'}（${mb.toFixed(1)} MB）`,
      `Uploading ${file.name || 'image'} (${mb.toFixed(1)} MB)`));
  try {
    const onProg = (loaded, total, ms, part, parts) => {
      const pct = total ? 100 * loaded / total : 0;
      const secs = ms / 1000;
      const spd = secs > 0.35 ? (loaded / 1e6) / secs : 0;
      const tag = parts ? T(`　分片 ${part}/${parts}`, ` · chunk ${part}/${parts}`) : '';
      if (loaded >= total) {
        upSet(100, T(`上传完成 ${mb.toFixed(1)} MB${tag}　服务器解码中…`,
      `Uploaded ${mb.toFixed(1)} MB${tag} — server decoding…`));
      } else {
        upSet(pct, `${pct.toFixed(0)}%　${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
          + (spd ? `　${spd.toFixed(1)} MB/s` : '') + tag);
      }
    };
    let meta;
    try {
      meta = await uploadXHR(fd, onProg);          // 正常就一次整传
    } catch (err) {
      // 外层反代的 body 上限（曾把 >1 MB 回成 500）偶尔复发：大图自动退到分片再试一次
      if (file.size > CHUNK_OVER && !/已取消/.test(err.message || '')) {
        upSet(0, T('整传被反代拦下，改用分片重传…', 'The proxy rejected the single-shot upload — retrying in chunks…'));
        meta = await uploadChunked(file, onProg);
      } else {
        throw err;
      }
    }
    S.id = meta.id; S.name = meta.name; S.ow = meta.ow; S.oh = meta.oh;
    S.shrunkNote = meta.shrunk_note || ''; S.origW = meta.orig_w || meta.ow; S.origH = meta.orig_h || meta.oh;
    S.res = meta.default_res || 'full'; S.rot = 0;
    await refetchImage();
    S.rois = []; S.sel = -1; S.samples = []; S.selSample = -1;
    S.overlayUrl = null; S.overlayImg = null; S.contourUrl = null; S.contourImg = null;
    S.annot.strokes = []; S.annot.cur = null; S.annot.res = null;
    S.tol = 25; S.purity = 0.5;              // 容差/纯度跟着图走，重置
    S.widthTouched = false; S.suggRetry = false;
    // 刀缝宽度也不沿用：上一张手设的 k 是那张图的（不同相机/倍率下同一个数不是同一段宽度），
    // 新图一律回到"没量过"的初值，交给判法重定——半自动会带 r.line_px 回来，
    // 或用「建议值」按钮按露底厚度算。手改标记一并作废，否则自动值会被 lineManual 挡住。
    S.line = 1; S.lineSrc = 'default'; S.lineManual = false;
    S.gmColsManual = false; S.gmRowsManual = false; S.gmAngManual = false;
    S.gmCountsSig = ''; S.gmAutoSig = ''; S.samplesManual = false;   // 新图：格数/歪角/分色都可以重新自动
    S.suggest = null; S.suggestSrc = ''; S.suggestHold = null; S.knee = null; S.kneeWarn = '';
    clearImageState();                      // 结果卡/胶囊/建议阈值/诊断全部清空
    $('#chkAutoTol').checked = true; S.autoTol = true;
    syncParams(); syncSampleUI(); refreshRoiList();
    $('#fileName').textContent = meta.name || T('未命名', 'untitled');
    $('#drop').classList.add('hide');
    fitView();
    const resTxt = (meta.default_res === 'full') ? T('全画质', 'full resolution') : (T('长边 ', 'long edge ') + meta.default_res);
    setResult(T(`已打开：${meta.name}\n原图 ${meta.ow}×${meta.oh} → 分析 ${meta.w}×${meta.h}（${resTxt}）\n` +
      (meta.dng ? `DNG：${meta.dng}\n` : '') +
      `\n下一步：切到「吸色」页 → 在完好漆膜上点一下，在露底基材上点一下 → 回「框选」页框出划格区。`,
      `Opened: ${meta.name}\noriginal ${meta.ow}×${meta.oh} → analysed ${meta.w}×${meta.h} (${resTxt})\n` +
      (meta.dng ? `DNG: ${meta.dng}\n` : '') +
      `\nNext: go to “Pick” → click the intact coating, click the exposed substrate → back to “Frame” and frame the cross-hatch area.`));
    updateFileInfo();
    setStatus('');
    upHide();
  } catch (err) {
    upHide();
    setStatus('');
    if (err.message === T('已取消', 'Cancelled')) setResult(T('已取消这次上传，可以重新拖一张进来。', 'Upload cancelled — drop another image when you are ready.'));
    else setResult(T('打开失败：', 'Failed to open: ') + err.message);
  }
}

async function refetchImage(onProg, signal) {
  const meta = await post('api/load', { id: S.id, res: S.res }, 'POST', signal);
  S.w = meta.w; S.h = meta.h; S.note = meta.note || '';
  S.shrunkNote = meta.shrunk_note || S.shrunkNote || '';
  const url = 'api/image/' + S.id + '?res=' + encodeURIComponent(S.res) + '&v=' + Date.now();
  if (onProg) await loadImageProgress(url, onProg, signal);
  else await loadImage(url);
  S.fit = Math.min(S.cw / S.w, S.ch / S.h);
  updateFileInfo();
}

function updateFileInfo() {
  if (!S.id) return;
  const k = (S.ow / S.w);
  const an = S.analysis;
  const extra = (S.roiNative && an && an.roi_native && an.w)
    ? ' · ' + tf('file.roinative', an.w, an.h, an.scale.toFixed(2))
    : '';
  $('#fileInfo').textContent = (S.shrunkNote ? `⚠ ${noteText(S.shrunkNote)} · ` : '') +
    tf('file.info', S.ow, S.oh, S.w, S.h, k.toFixed(2)) +
    extra +
    (Math.abs(S.rot) > 1e-6 ? ' · ' + tf('file.rotated', S.rot.toFixed(1)) : '') +
    (S.note ? ` · ${noteText(S.note)}` : '');
}


/* 服务端给的是中文提示（第二阶段才做服务端 i18n）：英文界面按已知形状就地翻译 */
function noteText(s) {
  if (!s || I18N.lang !== 'en') return s || '';
  let m;
  if ((m = s.match(/^已按内存上限（([\d.]+) MP）降采样分析$/)))
    return `downscaled to the ${m[1]} MP analysis cap`;
  if ((m = s.match(/^已启用「选区原生像素」：只分析框内 ([\d.]+) MP/)))
    return `native ROI: analysing ${m[1]} MP inside the box, no downscaling (k stays in canvas units)`;
  if ((m = s.match(/^已解码 DNG：点白平衡 \+ 曝光对齐（×([\d.]+)），未做锐化$/)))
    return `DNG decoded: white balance + exposure aligned (×${m[1]}), not sharpened`;
  if ((m = s.match(/^原图 (\d+)×(\d+)，上传时超过 (\d+) MP 上限，存档已缩到 (\d+)×(\d+)（(\d+)%）$/)))
    return `original ${m[1]}×${m[2]} exceeded the ${m[3]} MP cap; stored at ${m[4]}×${m[5]} (${m[6]}%)`;
  return s;
}

/* ------------------------------------------------------------------ 分析 */
/* 判法 = auto 时只是“先自动定个 k”，真正算的时候仍按宽度判法 */
function srvMode() { return (S.exmode === 'auto' || S.exmode === 'auto_blk' || S.exmode === 'grid') ? 'width' : S.exmode; }
function payload() {
  return {
    id: S.id, res: S.res,
    rois: S.rois.map(r => ({ box: r.box, ang: r.ang || 0 })),
    samples: S.samples.map(s => ({ rgb: s.rgb, role: s.role })),
    tol: S.tol, purity: S.purity, line_px: S.line, mode: srvMode(),
    want_hist: true, detail: S.detail, overlay: true,
    ruler_hist: S.rulerHist, ruler_full: S.rulerFull,
    soft: false,                      // 软权重已撤下，不再参与计算
    roi_native: !!S.roiNative,
    want_suggest: (S.exmode === 'width'),    // 按块数回落时 k 由判法定，别让服务器再猜一遍
  };
}

/* 按钮的“计算中…”要盖住两段：分析 + 列块/画蚂蚁线。只盖分析的话，
   分析一结束按钮就变回“计算”，但蚂蚁线还是淡色（块还没回来），看着像算完了其实没有。 */
function refreshCalcBtn() {
  const b = !!(S.busy || S.blocksBusy || S.blocksDue);
  $('#btnCalc').disabled = b;
  $('#btnCalc').textContent = b ? t('res.calculating') : t('btn.calc');
}
function setBusy(b) { S.busy = b; refreshCalcBtn(); }

/* 刀缝值从哪来：自动算的 / 沿用的 / 手设的 / 量尺量的 / 记录里的。
   “沿用”是最容易被误当成结果的一种——算不出 k 时它就是上一次的值。 */
function markLineSrc(src) {
  if (src) S.lineSrc = src;
  const el = $('#lineSrc'); if (!el) return;
  const map = {
    auto:   ['自动算出', 'derived'],
    keep:   ['沿用·未算出', 'inherited (not derived)'],
    manual: ['手设', 'set by hand'],
    suggest:['建议值·按露底', 'suggested (bare thickness)'],
    fallback:['给不出阈值·用 1 px', 'no threshold, using 1 px'],
    ruler:  ['量尺量的', 'measured'],
    record: ['记录里的', 'from record'],
    default:['默认值·未量', 'default (never measured)'],
  };
  const x = map[S.lineSrc] || map.default;
  el.textContent = '（' + T(x[0], x[1]) + '）';
  el.style.color = (S.lineSrc === 'keep' || S.lineSrc === 'default' || S.lineSrc === 'fallback') ? '#d29922' : '';
  el.title = (S.lineSrc === 'keep' || S.lineSrc === 'default' || S.lineSrc === 'fallback')
    ? T('这个宽度不是算法算出来的，请到量尺页量一条刀缝，或在 k 曲线里点一档',
        'This width was not derived — measure a cut on the ruler tab, or click a row in the k curve')
    : '';
}
function setBlocksBusy(b) { S.blocksBusy = b; refreshCalcBtn(); }

/* 自动定刀缝：扫 k 曲线，取“再降 1 px 露底就突增”的上侧那一档 */
let kneeBusy = false;
function kneeSig() {
  return JSON.stringify({
    r: S.rois.map(r => [Math.round(r.box[0]), Math.round(r.box[1]), Math.round(r.box[2]), Math.round(r.box[3]), r.ang || 0]),
    s: S.samples.map(s => s.rgb.join(',') + s.role),
    t: S.tol, p: S.purity, n: S.roiNative ? 1 : 0, res: S.res,
  });
}
function renderKneeWarn() {
  const el = $('#kneeWarn');
  const t = (S.kneeWarn || '').trim();   // 判法无关：没检出露底 / 两种判据都没有台阶时，任何判法都要显示
  if (el) { el.textContent = t; el.style.display = t ? 'block' : 'none'; }
  // 顶栏那个常驻小黄标：判法在「量尺」标签页里，平时看不见，别让人漏掉
  const pill = $('#warnPill');
  if (pill) {
    pill.style.display = t ? '' : 'none';
    pill.title = t;
    pill.onclick = () => {
      const b = $$('.tabs button').find(x => x.dataset.tab === 'ruler');
      if (b) b.click();
    };
  }
}

function goRulerTab() {
  const b = $$('.tabs button').find(x => x.dataset.tab === 'ruler');
  if (b) b.click();
}

function renderKnee() {
  const el = $('#kneeInfo'); if (!el) return;
  if (S.exmode === 'width') {
    // 按露底分布：不铺整张表，但必须说清「扫了没有 / 定出没有」——否则用户看到 k=1 会以为算出来了
    const ok = S.knee && S.knee.rows && S.knee.rows.length && S.knee.rule === 'area';
    el.innerHTML = ok
      ? '<b>' + T('k 曲线', 'k curve') + '</b>' + T('（已扫 k=1…16，按露底分布）',
          ' (scanned k=1…16, by bare area)') + '：'
        + (S.knee.derived
            ? T('定出 k=' + S.knee.k + '，当前用的就是它', `derived k=${S.knee.k}; that is what is being used`)
            : '<span style="color:#d29922">' + T('没定出 k（曲线一路只升不降，最大档只是扫描上限）——当前 k='
                + S.line + ' px 是沿用的值', 'no k derived (the curve only rises, so its largest step is just the '
                + 'scan limit) — the current ' + S.line + ' px is inherited') + '</span>')
        + T('。要看整张表或手点一档，切到「按块数回落」。', '. Switch to “by block count” to see the whole table '
            + 'or pick a row by hand.')
      : T('按露底分布这条判法不扫 k 曲线：'
          + (S.lineSrc === 'suggest' ? '刀缝用服务器按露底厚度给的建议值 ' + S.line + ' px（可自己改）'
             : S.lineSrc === 'fallback' ? '服务器给不出可信阈值（照建议会把真实脱落删掉），先用 ' + S.line + ' px'
             : S.lineSrc === 'manual' ? '刀缝用你手设的 ' + S.line + ' px'
             : S.lineSrc === 'ruler' ? '刀缝用你量尺量的 ' + S.line + ' px'
             : '刀缝是 ' + S.line + ' px（没量过、也不是算出来的）')
          + '；要看整张表或手点一档，切到「按块数回落」。',
          '“By bare area” does not scan the k curve: '
          + (S.lineSrc === 'suggest' ? 'the cut width is the server suggestion from bare thickness, '
                + S.line + ' px (you can change it)'
             : S.lineSrc === 'fallback' ? 'no trustworthy threshold could be derived (following it would delete '
                + 'real removal), so ' + S.line + ' px is used'
             : S.lineSrc === 'manual' ? 'the cut width is your hand-set ' + S.line + ' px'
             : S.lineSrc === 'ruler' ? 'the cut width is your measured ' + S.line + ' px'
             : 'the cut width is ' + S.line + ' px (never measured, not derived)')
          + '. Switch to “by block count” to see the whole table or pick a row by hand.');
    try { renderKneeWarn(); } catch (_) {}
    return;
  }
  if (!S.knee || !S.knee.rows) { el.textContent = ''; return; }
  const autoNow = (S.exmode === 'auto' || S.exmode === 'auto_blk');
  const isGrid = (S.exmode === 'grid');
  // 每一行都能点：点哪档就把刀缝设成哪档，边看蚂蚁线边挑
  const rows = S.knee.rows.map(r => {
    const on = r.k === S.line;
    return '<div class="krow' + (on ? ' on' : '') + '" data-k="' + r.k + '" style="cursor:pointer;padding:1px 4px;border-radius:3px;'
      + (on ? 'background:rgba(88,166,255,.18);color:#8cc8ff;font-weight:600' : '') + '">'
      + (on ? '▶ ' : '&nbsp;&nbsp;') + 'k=' + String(r.k).padStart(2) + T('   块 ', '  blocks ')
      + String(r.n).padStart(3) + T('   露底 ', '  bare ') + r.pct.toFixed(2) + '%</div>';
  }).join('');
  const label = isGrid ? T('k 曲线（刀宽）', 'k curve (blade)')
    : (S.exmode === 'auto_blk') ? T('按块数回落', 'By block count') : T('按露底分布', 'By bare area');
  el.innerHTML = (isGrid
      ? '<b>' + T('k 曲线（当前刀宽 ' + S.line + ' px）', 'k curve (blade ' + S.line + ' px)') + '</b>'
        + T('：每行的 k 就是刀缝阈值＝刀宽，点一行就把它设成刀宽（橙色带随之变宽），格网立刻重算。'
            + (S.knee.derived ? '自动判法建议 k=' + S.knee.k + '。' : '自动判法没定出 k，自己挑一档。'),
            ': each row is a cut threshold ＝ blade width — click one to use it; the grid recomputes. '
            + (S.knee.derived ? 'The automatic rule suggests k=' + S.knee.k + '.'
                              : 'The automatic rule derived no k; pick a row yourself.'))
      : autoNow
      ? (S.knee.derived
          ? '<b>' + label + T('：k=', ': k=') + S.knee.k + '</b>' + T('（', ' (') + (S.knee.why || '') + T('）', ')')
          : '<b style="color:#d29922">' + label + T('：没定出 k</b>', ': no k could be derived</b>')
            + T('（', ' (') + (S.knee.why || '') + T('）—— 下面 ▶ 是沿用的 ' + S.line + ' px，不是算出来的',
                ') — the ▶ row below is the inherited ' + S.line + ' px, not a derived value'))
      : '<b>' + T('k 曲线', 'k curve') + '</b>' + T('（自动值 k=' + (S.knee.k || '—') + '，当前按你设的宽度 ' + S.line + ' 算；点下面任意一档就改用它）',
          ` (auto k=${S.knee.k || '—'}, currently using your width ${S.line}; click any row below to switch to it)`))
    + '<div style="margin-top:6px;font-size:11.5px;line-height:1.5">' + rows + '</div>'
    + '<div style="opacity:.65;font-size:11px;margin-top:2px">'
    + T('点任意一行 = 用该档当刀缝（▶ 是当前用的）', 'Click any row = use that width as the cut threshold (▶ marks the current one)')
    + '</div>';
  el.querySelectorAll('.krow').forEach(d => {
    d.onclick = () => { setLine(+d.dataset.k); renderKnee(); };
  });
  try { renderKneeWarn(); } catch (_) {}
}

/* 没有 k 曲线时给一句话，别让面板空着让人以为功能没了 */
function updateKneeHint() {
  const el = $('#kneeInfo'); if (!el) return;
  if (S.knee && S.knee.rows && S.knee.rows.length) { renderKnee(); return; }
  if (S.exmode === 'width') { renderKnee(); return; }             // 按露底分布：统一由 renderKnee 出一句“扫没扫/定没定”
  if (S.exmode === 'grid') {
    // 格网判法下 k 就是刀宽：不主动扫，给一行可点的入口（用户点了才花那 4 秒）
    el.innerHTML = '<a href="#" id="kneeScanLink" style="opacity:.8">'
      + T('k 曲线（= 刀宽）：点这里扫一遍 k=1…16（约 4 秒）', 'k curve (= blade width): click to scan k=1…16 (~4 s)')
      + '</a>';
    const a = $('#kneeScanLink');
    if (a) a.onclick = ev => { ev.preventDefault(); ensureKnee(true).catch(() => {}); };
    return;
  }
  el.innerHTML = '<span style="opacity:.65">'
    + T('k 曲线：框好选区、吸完两个颜色（或打开已有记录）就会自动扫一次（约 4 秒）；'
      + '这里会列出 k=1…16 的块数和露底，点任意一档可直接用它。',
      'k curve: once a region is framed and two colours are sampled (or a record is opened) it is scanned '
      + 'automatically (~4 s). It lists block count and bare area for k=1…16; click a row to use it.')
    + '</span>';
}

async function ensureKnee(force) {
  if (!S.id || !S.rois.length || S.samples.length < 2) return false;
  const sig = kneeSig();
  const rule = (S.exmode === 'auto_blk' || S.exmode === 'grid') ? 'count' : 'area';
  const autoNow = (S.exmode === 'auto' || S.exmode === 'auto_blk');
  if (!force && S.knee && S.knee.sig === sig && S.knee.rule === rule) {
    // 切回自动判法：即便曲线是缓存的，也要把刀缝重新按判法对齐（用户手改过的除外）
    if (autoNow && !S.lineManual && S.knee.k && S.line !== S.knee.k) {
      S.line = S.knee.k;
      if ($('#rngLine')) $('#rngLine').value = Math.min(25, S.line);
      if ($('#numLine')) $('#numLine').value = S.line;
      setStatus(T('按块数回落：k=' + S.knee.k + '（已按判法重设刀缝）',
        `By block count: k=${S.knee.k} (cut threshold reset by the method)`));
    }
    renderKnee();
    return true;
  }
  if (kneeBusy) return false;
  kneeBusy = true;
  try {
    setStatus(T('自动定刀缝：正在扫 k=1…15（约 4 秒）…', 'Auto cut width: scanning k=1…15 (~4 s)…'));
    const r = await post('api/knee', { ...payload(), mode: 'width', k_max: 16, rule });
    if (!r.ok) { setStatus(r.reason || T('自动定刀缝失败', 'Automatic cut width failed')); return false; }
    // derived=false 表示算法没定出 k：这时表里 ▶ 那档只是"当前沿用"的宽度，
    // 别把它印成"按块数回落：k=8"，否则看着像算出来的（用户就是被这个骗到）。
    S.knee = { sig, rule, k: r.k_wire || null, derived: !!r.k_wire, rows: r.rows || [], why: r.why || '',
               k_area: r.k_area, k_count: r.k_count, why_area: r.why_area, why_count: r.why_count,
               warn: r.warn || '', no_bare: !!r.no_bare };
    const autoMode = (S.exmode === 'auto' || S.exmode === 'auto_blk');
    if (!S.lineManual) markLineSrc(r.k_wire ? 'auto' : 'keep');
    if (r.k_wire && !S.lineManual && autoMode) {
      S.line = r.k_wire;
      if ($('#rngLine')) $('#rngLine').value = Math.min(25, S.line);
      if ($('#numLine')) $('#numLine').value = S.line;
    }
    // 判法没定出 k：留一条常驻黄条（不是状态栏那种一眨眼就没的提示）
    if (autoMode && rule === 'count' && r.k_from === 'suggest' && r.k_wire) {
      const rows0 = (r.rows || []).filter(x => x && typeof x.n === 'number');
      let pk0 = rows0[0]; for (const x of rows0) if (pk0 && x.n > pk0.n) pk0 = x;
      const last0 = rows0[rows0.length - 1];
      const hit0 = rows0.find(x => x.k === r.k_wire) || {};
      S.kneeWarn = T('⚠ 按块数回落没找到台阶（块数峰值 ' + (pk0 ? pk0.n : '?') + '，到 k=' + (last0 ? last0.k : '?')
        + ' 还有 ' + (last0 ? last0.n : '?') + ' 块）→ 已改用露底分布的建议 k=' + r.k_wire
        + (hit0.pct != null ? '（露底 ' + hit0.pct.toFixed(2) + '%）' : '')
        + '。这是兜底值，请在下面曲线点一档确认。',
        '⚠ No step found by block count (peak ' + (pk0 ? pk0.n : '?') + ' blocks, still '
        + (last0 ? last0.n : '?') + ' blocks at k=' + (last0 ? last0.k : '?') + ') → fell back to the '
        + 'bare-distribution suggestion k=' + r.k_wire
        + (hit0.pct != null ? ' (bare ' + hit0.pct.toFixed(2) + '%)' : '')
        + '. That is a fallback: confirm it by clicking a row in the curve below.');
    } else if (autoMode && rule === 'count' && !r.k_wire) {
      const rows = (r.rows || []).filter(x => x && typeof x.n === 'number');
      if (rows.length) {
        let pk = rows[0]; for (const x of rows) if (x.n > pk.n) pk = x;
        const last = rows[rows.length - 1];
        const thr = Math.round(pk.n * 0.25);
        S.kneeWarn = T('⚠ 按块数回落没找到台阶：块数峰值 ' + pk.n + '（k=' + pk.k + '），到 k=' + last.k
          + ' 还有 ' + last.n + ' 块（要 ≤' + thr + ' 才算回落）。现在沿用 ' + S.line + ' px —— '
          + '请在下面曲线里点一档确认。',
          '⚠ No step found by block count: peak ' + pk.n + ' blocks (k=' + pk.k + '), still ' + last.n
          + ' blocks at k=' + last.k + ' (needs ≤' + thr + ' to count as a fall). Keeping ' + S.line
          + ' px — confirm by clicking a row in the curve below.');
      } else {
        S.kneeWarn = T('⚠ 按块数回落没定出 k，现在沿用 ' + S.line + ' px，建议手挑一档。',
        '⚠ Block count did not produce a k; keeping ' + S.line + ' px — pick a row by hand.');
      }
    } else if (r.k_wire) {
      S.kneeWarn = '';
    }
    // 服务端现在会给一条「与判法无关」的告警：没检出露底 / 两种判据都没定出 k。
    // 以前只有 rule=count 才会写提示，判法=按露底分布（rule=area）时一句都没有。
    if (r.warn) S.kneeWarn = r.warn;
    renderKnee();
    setStatus(!autoMode
      ? T('k 曲线已扫（自动值 k=' + (r.k_wire || '—') + '，当前按你设的宽度 ' + S.line + ' 算；点曲线任意一档可改用它）',
          `k curve scanned (auto k=${r.k_wire || '—'}; currently using your width ${S.line}; click any row to switch)`)
      : (r.k_wire
         ? (S.lineManual
            ? T('按块数回落：k=' + r.k_wire + '；你已手设刀缝 ' + S.line + '，按你的算（重选「按块数回落」可恢复）',
                `By block count: k=${r.k_wire}; you set ${S.line} by hand, so yours is used (re-select “By block count” to restore)`)
            : T('按块数回落：k=' + r.k_wire + '（' + (r.why || '') + '）', `By block count: k=${r.k_wire} (${r.why || ''})`))
         : T('按块数回落：没找到明显台阶，仍用当前值', 'By block count: no clear step, keeping the current value')));
    return true;
  } finally { kneeBusy = false; }
}

async function analyze() {
  if (!S.id) return;
  if (!S.rois.length) {                       // 选区全删了：不许拿整图凑数（会得 0B）
    S.rows = null; S.total = null; S.autoTotal = null; S.autoRows = null;
    S.gm = null; S.gmDrag = null; S.gmOff = { ox: 0, oy: 0, dx: [], dy: [] };   // 格网也跟着走
    if ($('#gmInfo')) $('#gmInfo').textContent = '';
    if ($('#gmTrace')) $('#gmTrace').textContent = '';
    S.lines = []; S.soft = null;
    setResult(t('res.noRoi'));
    const sum0 = $('#summary');
    if (sum0) { sum0.textContent = '—'; sum0.className = 'summary'; }
    const bd = $('#resBadge');
    if (bd) { bd.textContent = '—'; bd.className = 'rg'; }
    const pc0 = $('#resCardPct');
    if (pc0) pc0.textContent = '';
    const rb0 = $('#resCardRows');
    if (rb0) rb0.innerHTML = '<div class="rrow"><span>—</span></div>';
    const ss0 = $('#sideSummary');
    if (ss0) ss0.textContent = '—';
    renderDiag();
    dropOverlay();
    draw();
    return;
  }
  if (!S.samples.length) {
    setResult(t('res.needColors'));
    return;
  }
  if (S.samples.length < 2) {
    setResult(T('只吸了一个颜色。还需要另一个（漆膜 / 基材），否则没法算占比。', 'Only one colour sampled. Sample the other one (coating / substrate) — without both there is no ratio to compute.'));
    return;
  }
  // 两种判法各走各的：按块数回落 → 先扫 k；按宽度 → 只等服务器的露底分布建议值，不扫 k（免得两套规则重复算）
  const my = ++S.seq;
  setBusy(true);
  try {
    // 扫 k 曲线（按块数回落）也要算进"计算中"：这一趟要 4 秒左右，之前按钮在这段时间还写着
    // "计算"，而后台其实已经在忙，看着像没点动。按露底分布不扫 k（只用建议值），所以那边切换及时。
    if (S.exmode === 'auto' || S.exmode === 'auto_blk') await ensureKnee();
    const r = await post('api/analyze', payload());
    if (my !== S.seq) return;
    if (!r.ok) { setResult(r.reason || T('分析失败', 'Analysis failed')); return; }
    S.rows = r.rows; S.total = r.total; S.lines = r.lines;
    // 存档值 vs 当前算法：改了吸色/判级规则以后，重开旧记录的数字会漂，必须当场说，别让人以为记错了
    if (!S.driftChecked && S.recPct != null && Math.abs(r.total.pct - S.recPct) > 0.05) {
      const _d = r.total.pct - S.recPct;
      const _f = (x) => (x >= 0 ? '+' : '') + x.toFixed(2);
      S.lines.unshift(T(`· 存档 ${S.recPct.toFixed(2)}%（算法 ${S.recAlgo || '旧版'}）≠ 现在 ${r.total.pct.toFixed(2)}%（算法 ${r.total.algo || '?'}），差 ${_f(_d)} pp —— 算法或参数改过了，以现在为准`,
        `· Archived ${S.recPct.toFixed(2)}% (algorithm ${S.recAlgo || 'older'}) ≠ now ${r.total.pct.toFixed(2)}% `
        + `(algorithm ${r.total.algo || '?'}), Δ ${_f(_d)} pp — the algorithm or parameters changed; the current value wins`));
      setStatus(T('注意：这条记录的存档值和现在的算法不一致（见结果栏第一行）',
        'Note: this record was saved with a different algorithm — see the first line of the result panel'));
    }
    S.driftChecked = true;
    // 每次算完都刷一次 k 面板：按露底分布这条判法不进 ensureKnee()，
    // 不刷的话面板上留的是上一次（可能是另一种判法）的说明。
    try { renderKnee(); } catch (_) {}
    S.analysis = r.analysis || null; updateFileInfo();
    S.autoTotal = { ...r.total, worst: r.total.worst ? { ...r.total.worst } : null };
    S.autoRows = r.rows.map(x => ({ ...x }));
    S.autoLines = r.lines.slice();
    S.soft = r.soft || null;
    // 开着的时候绝不把块清空：清了蚂蚁线会空一段（后端列块要 1~3 s，用户看到的就是“线消失了”）。
    // 旧块留着半透明继续画，新结果一到就替换；只有“已关闭”状态下才按签名清缓存。
    if (S.blocks && S.blocks.sig !== blocksSig() && !S.blocksOn) { S.blocks = null; renderBlocks(); }
    else if (S.blocks) applyBlocks();
    if (S.blocksOn) {
      clearTimeout(S.blocksTimer);
      S.blocksDue = true; refreshCalcBtn();                     // 分析刚结束、块还没开始算的 450ms 空档也要显示“计算中”
      S.blocksTimer = setTimeout(() => listBlocks(true), 450);   // 拖框/调 k 时合并成一次
    }
    if (S.suggestHold == null) { S.suggest = r.suggest_px; S.suggestSrc = r.suggest_src; }
    // 按宽度模式：服务器给的建议值（按露底厚度猜的刀缝宽度）当这张图的起点
    if (S.exmode === 'width' && !S.widthTouched && !S.suggRetry) {
      const hasSug = (typeof r.suggest_px === 'number' && r.suggest_px >= 1);
      const want = hasSug ? clamp(Math.round(r.suggest_px), 1, 60) : 1;   // 拒绝给建议 → 用 1 px
      if (!S.lineManual) { markLineSrc(hasSug ? 'suggest' : 'fallback'); try { renderKnee(); } catch (_) {} }
      if (want !== S.line) {
        S.suggRetry = true;
        S.line = want;
        $('#rngLine').value = Math.min(25, S.line); $('#numLine').value = S.line;
        try { localStorage.setItem('baige.line', String(S.line)); } catch (_) {}
        setStatus(hasSug
          ? T('按露底分布：已用建议值 ' + S.line + ' px 起算（' + (r.suggest_src || '按露底厚度') + '，可自己改）',
              'By bare distribution: starting from the suggested ' + S.line + ' px (' + (r.suggest_src || 'bare thickness') + ', you can change it)')
          : T('按露底分布：给不出可信阈值（照建议会把真实脱落删掉），先用 1 px；可到 k 曲线里手挑一档',
              'By bare distribution: no trustworthy threshold (following the suggestion deletes real removal) — starting at 1 px; pick a row in the k table instead'));
        { const _my = my; setTimeout(() => { if (S.seq === _my) analyze(); }, 0); return; }
      }
      S.suggRetry = true;
    }
    S.span = r.span;
    updateTolHint(r);
    renderResult(); refreshRoiList(); updateSuggest(); updateSpan();
    if (r.overlay_url) loadOverlay(r.overlay_url); else { S.overlayImg = null; }
    draw();
  } catch (err) {
    setResult(T('分析失败：', 'Analysis failed: ') + err.message);
  } finally {
    if (my === S.seq) {
      setBusy(false);
      // 格网判法：像素分析跑完再铺格（以前 setLine→scheduleGm 在 busy 时被丢掉，刀宽看着“不生效”），
      // 并顺手把 k 曲线扫出来——这条判法下 k 就是刀宽，表要能看、能点。
      if (S.exmode === 'grid') {
        scheduleGm(30);
        if (S.samples.length >= 2) ensureKnee().catch(() => {});
      }
    }
  }
}
const debouncedAnalyze = debounce(analyze, 220);

const THR = [0.5, 5, 15, 35, 65];
const GRADE_ABOVE = ['4B', '3B', '2B', '1B', '0B'];

/* 复刻 crosscut.margin_hint 的文案，用于“手动排除块之后”改写那一行 */
function marginHint(pct) {
  const L = [0.5, 5.0, 15.0, 35.0, 65.0];
  const en = I18N.lang === 'en';
  const parts = [en ? `Margin: ${pct.toFixed(2)}% is ${gradeOf(pct)}`
                    : `判定余量：${pct.toFixed(2)}% 属 ${gradeOf(pct)}`];
  for (const lim of L) {
    if (pct < lim - 1e-9) {
      parts.push(en ? `+${(lim - pct).toFixed(2)} pp (to ${lim.toFixed(1)}%) becomes ${gradeOf(lim + 1e-6)}`
                    : `再多 ${(lim - pct).toFixed(2)} pp（到 ${lim.toFixed(1)}%）即 ${gradeOf(lim + 1e-6)}`);
      break;
    }
  }
  for (const lim of [...L].reverse()) {
    if (pct >= lim - 1e-9) {
      if (lim === 0.5) parts.push(en ? `below 0.5% becomes 5B (-${(pct - 0.5).toFixed(2)} pp)`
                                     : `降到 0.5% 以下即 5B（少 ${(pct - 0.5).toFixed(2)} pp）`);
      else parts.push(en ? `-${(pct - lim).toFixed(2)} pp (to ${lim.toFixed(0)}%) becomes ${gradeOf(lim - 1e-6)}`
                         : `少 ${(pct - lim).toFixed(2)} pp（到 ${lim.toFixed(0)}%）即 ${gradeOf(lim - 1e-6)}`);
      break;
    }
  }
  return parts.join(en ? '; ' : '；');
}

function marginText(pct) {
  let best = null;
  THR.forEach((b, i) => {
    const d = Math.abs(pct - b);
    if (!best || d < best.d) best = { d, b, i };
  });
  if (!best) return '';
  const en = I18N.lang === 'en';
  const dir = pct < best.b
    ? (en ? `${best.d.toFixed(2)} pp up crosses into ${GRADE_ABOVE[best.i]}`
          : `再高 ${best.d.toFixed(2)} pp 就越到 ${GRADE_ABOVE[best.i]}`)
    : (en ? `${best.d.toFixed(2)} pp down crosses into ${best.i === 0 ? '5B' : GRADE_ABOVE[best.i - 1]}`
          : `再低 ${best.d.toFixed(2)} pp 就越到 ${best.i === 0 ? '5B' : GRADE_ABOVE[best.i - 1]}`);
  return en ? `${best.d.toFixed(2)} pp from the ${best.b}% boundary (${dir})`
            : `距分界 ${best.b}% 仅 ${best.d.toFixed(2)} pp（${dir}）`;
}

function renderDiag() {
  const dg = $('#diagx'), v1 = S.total;
  if (!v1 || !S.detail) { dg.innerHTML = ''; return; }   // 诊断明细关着就不铺这些数
  const rows = [
    [T('判级余量', 'Grade margin'), marginText(v1.pct)],
    [T('已剔除', 'Excluded'), v1.groove_px > 0
      ? T(`${v1.groove_px} px（占统计面积 ${fmt(v1.groove_share, 2)}%，刀缝阈值 ${S.line}）`,
          `${v1.groove_px} px (${fmt(v1.groove_share, 2)}% of the counted area, cut threshold ${S.line})`)
      : T(`0 px（刀缝阈值 = ${S.line}，没有剔除任何刀缝）`,
          `0 px (cut threshold = ${S.line}, nothing excluded)`)],
    [T('未分类', 'Unclassified'), `${fmt(v1.unassigned_pct, 2)}%`],
  ];
  dg.innerHTML = rows.map(([k, v]) => `<div class="diagrow"><span class="dk">${k}</span><span>${v}</span></div>`).join('');
}

/* 软权重（实验口径）—— 只追加显示，上面 v1 的数字一个字都不改 */
function softText() {
  const s = S.soft;
  if (!s || !s.ok) return '';
  const base = S.total ? S.total.pct : 0;
  const d = s.pct - base;
  const out = [
    '——— 软权重（实验口径）———',
    `主值 ${fmt(s.pct, 2)}% → ${s.astm} / ${s.iso}　（硬口径 ${fmt(base, 2)}%，差 ${d >= 0 ? '+' : ''}${fmt(d, 2)} pp）`,
    `不确定 ${fmt(s.unc_pct, 1)}%　区间 ${fmt(s.lo, 2)}% ~ ${fmt(s.hi, 2)}%`,
  ];
  if (s.cross) {
    out.push(`⚠ 区间跨档 ${s.lo_cls}~${s.hi_cls}：这张图只够给"临界"，别写成单一等级`);
  } else {
    out.push(`区间落在同一档（${s.astm}），可以按这个数报`);
  }
  return out.join('\n');
}

/* 手绘口径：三色笔 + 纯面积统计，和自动分类完全无关 */
let annotSeq = 0;
async function annotCalc() {
  if (!S.id) return;
  if (!S.annot.strokes.length) { S.annot.res = null; renderAnnot(); if (S.total) renderResult(); return; }
  const my = ++annotSeq;
  try {
    const r = await post('api/annot', {
      id: S.id, res: S.res,
      rois: S.rois.map(x => ({ box: x.box, ang: x.ang || 0 })),
      strokes: S.annot.strokes.map(k => ({ t: k.t, r: k.r, pts: k.pts })),
    });
    if (my !== annotSeq) return;
    S.annot.res = r.ok ? r : null;
  } catch (err) {
    if (my !== annotSeq) return;
    S.annot.res = null;
  }
  renderAnnot();
  if (S.total) renderResult();
}
const debouncedAnnot = debounce(annotCalc, 450);

function annotText() {
  const r = S.annot.res;
  if (!r || !r.ok) return '';
  const out = [
    '——— 手绘口径（你自己涂的）———',
    `脱落 ${fmt(r.d_pct, 2)}%｜刀缝 ${fmt(r.g_pct, 2)}%｜不确定 ${fmt(r.u_pct, 2)}%｜其余 ${fmt(r.bg_pct, 1)}%（完好／没标）`,
    `主值 ${fmt(r.pct, 2)}%（扣刀缝）→ ${r.astm} / ${r.iso}　范围 ${fmt(r.lo, 2)}% ~ ${fmt(r.hi, 2)}%`,
  ];
  out.push(r.cross ? `⚠ 范围跨档 ${r.lo_cls}~${r.hi_cls}：只够给"临界"` : `范围落在同一档（${r.astm}）`);
  return out.join('\n');
}

function renderAnnot() {
  const el = $('#annotInfo');
  if (!el) return;
  const r = S.annot.res;
  if (!r || !r.ok) {
    el.textContent = S.annot.strokes.length
      ? '算不出来：先框一个选区，再涂。'
      : '还没涂。切到这一页后直接在图上按住拖动。';
    return;
  }
  el.textContent = [
    `笔迹 ${r.n_stroke} 笔`,
    `脱落 ${fmt(r.d_pct, 2)}%　刀缝 ${fmt(r.g_pct, 2)}%　不确定 ${fmt(r.u_pct, 2)}%　其余 ${fmt(r.bg_pct, 1)}%（完好漆膜／没标）`,
    `主值 ${fmt(r.pct, 2)}%（扣刀缝）→ ${r.astm} / ${r.iso}`,
    `范围 ${fmt(r.lo, 2)}% ~ ${fmt(r.hi, 2)}%（刀缝不扣 ~ 不确定全算脱落）`,
    r.cross ? `⚠ 范围跨档 ${r.lo_cls}~${r.hi_cls}：只够给"临界"` : `范围同一档（${r.astm}），可以按这个数报`,
  ].join('\n');
}


/* 容差提示：mind 落在 tol 附近的像素才会因为调 tol 改变分类。
   带内占比≈0 就说明这张图调容差不会有任何反应，直接说清楚，别让人在滑块上白试。 */
function updateTolHint(r) {
  const el = $('#tolHint');
  if (!el) return;
  const te = r && r.tol_eff;
  if (!te) { el.style.display = 'none'; return; }
  if (Math.abs(te.delta) >= 0.05) { el.style.display = 'none'; return; }   // 有差就不提示
  // 注意措辞：这是“用不上”，不是“没调好”。容差本来就是干“兜底模糊像素”这一件事的，
  // 图上没有那种像素，它自然闲着；结果完全正常，用户什么都不用做。
  el.style.display = 'block';
  el.style.color = '';
  el.innerHTML = '<b>容差在这张图上用不上，不用调</b>：'
    + (r.tol * 0.75).toFixed(0) + ' / ' + r.tol + ' / ' + (r.tol * 1.25).toFixed(0)
    + ' 算出来都是 ' + te.lo.toFixed(2) + '%'
    + ((r.total && r.total.astm) ? ' · ' + r.total.astm : '') + '。'
    + '图里每个像素都能明确归到漆膜或基材，没有需要容差兜底的模糊像素，所以它闲着不影响结果；'
    + '当前值是按取样间距自动定的，保持自动就行。';
}

function renderResult() {
  // 判法=格网 时结果卡只认格网的数：像素判法算完会再刷一次卡，会把格网的数字盖掉
  // （截图里出现过：卡上 45.62%、画布上 47.5%，两个数并排没法看）
  if (S.exmode === 'grid' && S.gm) { try { setGridCard(S.gm); return; } catch (_) {} }
  if (S.blocks) applyBlocks();
  if (!S.total) return;
  const t = S.total;
  const st = '';                    // 软权重已撤下
  const bt = S.detail ? blocksText() : '';     // 逐块确认摘要归诊断明细
  let lines = S.lines || [];
  if (!S.detail) {                  // 只显示：总判级 + 各选区脱落率/判级 + 合计
    const keep = [];
    for (const ln of lines) {
      if (/^(判定余量|判级余量|露底厚度|—— 诊断明细)/.test(ln)
          || /^(Margin:|Bare-area thickness:|— diagnostics —)/.test(ln)) break;
      keep.push(ln);
    }
    lines = keep;
  }
  const rel = $('#result');
  rel.textContent = lines.join('\n') + '\n' + (bt ? bt + '\n' : '');
  rel.style.display = S.detail ? '' : 'none';     // 关闭诊断明细：卡片已经说清了，不再重复一层文本
  renderDiag();
  const sum = $('#summary');
  const w = t.worst;
  const G = w ? w.astm : t.astm;
  const B = w ? w.iso : t.iso;
  const unrel = (t.credible === false);     // 参数不可信：不给等级（未分类>12% 或漆膜<5%）
  // 顶栏胶囊只放最关键的：最差选区 + 等级；完整句子进 title（避免窄屏被省略号截断）
  const fullTxt = w
    ? (I18N.lang === 'en'
        ? `worst R${w.roi} ${fmt(w.pct, 2)}% → ASTM ${w.astm} / ISO ${w.iso} (total ${fmt(t.pct, 2)}% ${t.astm}, reference only)`
        : `${window.t('res.worst')} R${w.roi}　${window.t('res.dropPct')} ${fmt(w.pct, 2)}%  →  ASTM ${w.astm}  /  ISO ${w.iso}　（${window.t('res.total')} ${fmt(t.pct, 2)}% ${t.astm}，${window.t('res.refOnly')}）`)
    : (I18N.lang === 'en'
        ? `${fmt(t.pct, 2)}% → ASTM ${t.astm} / ISO ${t.iso}`
        : `${window.t('res.dropPct')} ${fmt(t.pct, 2)}%  →  ASTM ${t.astm}  /  ISO ${t.iso}`);
  const baseSum = w
    ? (I18N.lang === 'en' ? `worst R${w.roi} ${fmt(w.pct, 2)}% → ${w.astm}/${w.iso}`
                          : `${window.t('res.worst')} R${w.roi} ${fmt(w.pct, 2)}% → ${w.astm}/${w.iso}`)
    : (I18N.lang === 'en' ? `${fmt(t.pct, 2)}% → ${t.astm}/${t.iso}`
                          : `${fmt(t.pct, 2)}% → ${t.astm}/${t.iso}`);
  sum.textContent = unrel ? T('⚠ 参数不可信（' + fmt(t.pct, 2) + '%，等级仅供参考）',
                              '⚠ Parameters unreliable (' + fmt(t.pct, 2) + '%, grade for reference only)')
                          : baseSum;
  sum.title = (unrel && t.unreliable) ? (t.unreliable + '\n' + fullTxt) : fullTxt;
  sum.className = 'summary' + ((G === '0B' || G === '1B') ? ' bad' : '');
  const ss = $('#sideSummary');
  if (ss) ss.textContent = unrel ? T('参数不可信', 'unreliable')
                                 : (w ? `R${w.roi} ${fmt(w.pct, 2)}% ${G}` : `${fmt(t.pct, 2)}% ${G}`);
  // 保险一：输入不可信（未分类过高 / 漆膜占比过低）→ 徽章不给等级，免得 0B 被当真（unrel 在函数上方声明）
  const badge = $('#resBadge');
  if (badge) {
    if (unrel) {
      badge.textContent = '—';
      badge.className = 'rg gq';
      badge.title = t.unreliable || T('参数不可信', 'unreliable parameters');
    } else {
      badge.textContent = G; badge.className = 'rg ' + gcls(G); badge.title = '';
    }
  }
  const pc = $('#resCardPct');
  if (pc) pc.textContent = (w ? `R${w.roi} ${fmt(w.pct, 2)}%（${window.t('res.worst')}）` : `${fmt(t.pct, 2)}% ${window.t('res.dropPct')}`)
    + (unrel ? T('　· 参数不可信', ' · unreliable parameters') : '');
  const box = $('#resCardRows');
  if (box) {
    const rows = (S.rows || []).map((r, i) =>
      `<div class="rrow"><span>R${i + 1}</span><span>${fmt(r.pct, 2)}%</span>` +
      `<span class="rg ${unrel ? 'gq' : gcls(r.astm)}">${unrel ? '—' : (r.astm || '—')}</span></div>`);
    if (rows.length > 1) {
      rows.push(`<div class="rrow" style="border-top:1px dashed var(--line);margin-top:3px;padding-top:3px">` +
        `<span>${window.t('res.total')}</span><span>${fmt(t.pct, 2)}%</span>` +
        `<span class="rg ${unrel ? 'gq' : gcls(t.astm)}">${unrel ? '—' : (t.astm || '—')}</span></div>`);
    }
    box.innerHTML = rows.join('') || '<div class="rrow"><span>—</span></div>';
  }
  if (S.needRedraw) { S.needRedraw = false; draw(); }   // 让画布上的 R1 标签跟着变
}

function gcls(g) {
  const m = /(\d)B/.exec(g || '');
  return m ? 'g' + m[1] : 'g3';
}

async function copyText(txt) {
  try { await navigator.clipboard.writeText(txt); return true; } catch (_) {}
  try {
    const ta = document.createElement('textarea');
    ta.value = txt;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch (_) { return false; }
}

/* 复制成 TSV（制表符分隔）：就复制眼睛看到的这几行——选区 / 脱落率 / 等级，
   粘进 Excel 自动分列；要看面积/露底/未分类这些明细，用「导出报告」。 */
$('#resCopy').onclick = async () => {
  if (!S.total) { setStatus(T('还没有结果可复制', 'No result to copy yet')); return; }
  const pc = v => (v === undefined || v === null) ? '' : fmt(v, 2) + '%';
  const out = (S.rows || []).map((r, i) => ['R' + (i + 1), pc(r.pct), r.astm || ''].join('\t'));
  out.push([T('合计', 'Total'), pc(S.total.pct), S.total.astm || ''].join('\t'));
  const ok = await copyText(out.join('\r\n'));
  const btn = $('#resCopy');
  if (ok) {
    btn.textContent = T('✓ 已复制', '✓ Copied');
    btn.classList.add('ok');
    clearTimeout(btn._t);
    btn._t = setTimeout(() => { btn.textContent = T('复制', 'Copy'); btn.classList.remove('ok'); }, 1600);
    setStatus(T(`已复制 ${out.length} 行（选区 / 脱落率 / 等级），粘到 Excel 自动分列；明细请用「导出报告」`,
      `Copied ${out.length} rows (region / removal / grade) — paste into Excel; use “Export CSV” for the details.`));
  } else {
    setStatus(T('复制失败：浏览器不让写剪贴板，手动选中结果框复制吧', 'Copy failed: the browser blocked clipboard access — select the result box and copy manually.'));
  }
};

function loadContour(url) {
  if (!url) { S.contourUrl = null; S.contourImg = null; draw(); return; }
  if (S.contourUrl === url && S.contourImg) { draw(); return; }
  const im = new Image();
  im.onload = () => { S.contourUrl = url; S.contourImg = im; draw(); };
  im.src = url;
}

function loadOverlay(url) {
  if (S.overlayUrl === url && S.overlayImg) { draw(); return; }
  const im = new Image();
  im.onload = () => {
    S.overlayUrl = url; S.overlayImg = im;
    if (S.showOverlay) draw();
  };
  im.src = url;
}

/* 连续操作（拖第二个框、删框、调参）时把半自动合并成一次：
   350ms 内的多次触发只跑最后一次，并中断还在飞的上一次请求，避免服务器排队算没人看的结果 */
let semiTimer = null;
function scheduleSemi(delay = 350) {
  if (!S.semi) { if (S.samples.length >= 2 && S.rois.length) analyze(); return; }
  clearTimeout(semiTimer);
  semiTimer = setTimeout(() => { semiTimer = null; runSemi(); }, delay);
}

async function runSemi() {
  if (!S.id) return;
  if (!S.rois.length) { setResult(T('半自动已开：先在图上框出划格区，框完自动出结果。', 'Semi-auto is on: frame the cross-hatch area and the result appears automatically.')); return; }
  const my = ++S.seq;
  if (S.autoAbort) { try { S.autoAbort.abort(); } catch (_) {} }
  const ac = new AbortController(); S.autoAbort = ac;
  setBusy(true);
  try {
    const prevRows = S.autoRows ? S.autoRows.map(x => ({ ...x })) : null;
    const semiBody = { id: S.id, res: S.res, soft: false, roi_native: !!S.roiNative,
                       purity: S.purity,
                       rois: S.rois.map(x => ({ box: x.box, ang: x.ang || 0 })) };
    // 半自动不再自己判 k：第一步只认色，k 交给当前判法定，再回来算一次
    const det = await post('api/autorun', { ...semiBody, detect_only: true }, undefined, ac.signal);
    if (my !== S.seq) return;
    if (!det.ok) throw new Error(det.reason || T('认色失败', 'colour detection failed'));
    if (det.samples && !S.samplesManual) { S.samples = det.samples.map(s => ({ rgb: s.rgb, role: s.role })); S.selSample = -1; }
    if (typeof det.tol === 'number') { S.tol = det.tol; $('#rngTol').value = S.tol; $('#numTol').value = S.tol; }
    if (det.span != null) S.span = det.span;
    syncParams(); syncSampleUI();
    await ensureKnee();                     // 判法定 k（面板里那张表也在这步出来）
    if (my !== S.seq) return;
    // 哪种判法定 k：按块数回落 → 用判法的 k；按宽度 → 不传，让服务器按露底分布猜（就是它原来的做法）
    const autoMode = (S.exmode === 'auto' || S.exmode === 'auto_blk');
    const r = await post('api/autorun', autoMode ? { ...semiBody, line_px: S.line } : semiBody, undefined, ac.signal);
    if (my !== S.seq) return;
    if (!r.ok && r.reason === 'superseded') return;      // 已被更新的请求取代，静默丢掉
    if (!r.ok) {
      setStatus('');
      setResult(T('半自动没认出来：' + (r.reason || '未知原因') + '。\n请取消「半自动」后在图上手动吸两个色。',
                 'Semi-auto could not detect the colours: ' + (r.reason || 'unknown reason') +
                 '.\nTurn off “Semi-auto” and sample the two colours by hand on the image.'));
      $('#chkSemi').checked = false; S.semi = false;
      setBusy(false);
      return;
    }
    if (r.samples && !S.samplesManual) { S.samples = r.samples.map(s => ({ rgb: s.rgb, role: s.role })); S.selSample = -1; }
    if (typeof r.tol === 'number') S.tol = r.tol;
    if (typeof r.line_px === 'number' && !(S.exmode === 'auto' || S.exmode === 'auto_blk')) {
      S.line = r.line_px; S.widthTouched = true;      // 半自动给的就是这张图的宽度，别再被建议值顶掉
      markLineSrc('auto');                             // 半自动按线宽估的，算“算出来的”
      if ($('#rngLine')) $('#rngLine').value = Math.min(25, S.line);
      if ($('#numLine')) $('#numLine').value = S.line;
      try { localStorage.setItem('baige.line', String(S.line)); } catch (_) {}
    }
    syncParams(); syncSampleUI();
    S.rows = r.rows; S.total = r.total;
    S.analysis = r.analysis || null; updateFileInfo();
    S.autoTotal = { ...r.total, worst: r.total.worst ? { ...r.total.worst } : null };
    S.autoRows = r.rows.map(x => ({ ...x }));
    S.autoLines = r.lines.slice();
    S.soft = r.soft || null;
    S.lines = (r.note ? [r.note] : []).concat(r.lines);
    if (S.blocksOn && S.rois.length && S.samples.length >= 2) {
      clearTimeout(S.blocksTimer);
      S.blocksTimer = setTimeout(() => listBlocks(true), 450);
    }
    if (r.ignored && r.ignored.length) {
      const ps = r.ignored.map(o => `${o.pct}%`).join('、');
      const at = r.note ? 1 : 0;
      S.lines.splice(at, 0, T(`· 程序自动忽略了 ${r.ignored.length} 类颜色（占选区 ${ps}）`
        + `——在「吸色」页可改角色或删除`,
        `· ${r.ignored.length} colour cluster(s) were ignored automatically (${ps} of the region)`
        + ` — change the role or delete them on the “Pick” tab`));
    }
    // 新加的框会让整张图重新认色（KMeans 跑全部选区的并集），旧 ROI 的数字可能跟着变——明说，别让它悄悄变
    if (prevRows && prevRows.length && prevRows.length < r.rows.length) {
      const diffs = [];
      prevRows.forEach((o, i) => {
        const n = r.rows[i];
        if (n && Math.abs(n.pct - o.pct) > 0.02) diffs.push(`R${i + 1} ${fmt(o.pct, 2)}% → ${fmt(n.pct, 2)}%`);
      });
      if (diffs.length) {
        const at = r.note ? 1 : 0;
        S.lines.splice(at, 0, T(`· 加了这个选区后重新认色，之前的结果跟着变了：${diffs.join('、')}`
          + `（同一张图只认一套颜色；想固定就先只框一个、确认后再框下一个）`,
          `· Adding this region re-detected the colours, so earlier numbers moved: ${diffs.join(', ')}`
          + ` (one colour set per image; to keep them, frame one region at a time)`));
      }
    }
    if (S.suggestHold == null) { S.suggest = r.suggest_px; S.suggestSrc = r.suggest_src; }
    S.span = r.span;
    renderResult(); refreshRoiList(); updateSuggest(); updateSpan();
    if (S.exmode === 'auto' || S.exmode === 'auto_blk') ensureKnee().then(() => updateKneeHint()).catch(() => {});
    if (r.overlay_url) loadOverlay(r.overlay_url);
    draw();
  } catch (err) {
    if (err && err.name === 'AbortError') return;        // 被更新的请求中断，正常
    setResult(T('半自动失败：', 'Semi-auto failed: ') + err.message);
  } finally {
    if (my === S.seq) setBusy(false);
  }
}

/* ------------------------------------------------------------------ 吸色 / 量尺 */
async function pickAt(ex, ey) {
  if (!S.id) return;
  const p = toImg(ex, ey);
  if (p.x < 0 || p.y < 0 || p.x >= S.w || p.y >= S.h) return;
  try {
    const r = await post('api/pick', {
      id: S.id, res: S.res, x: p.x, y: p.y,
      samples: S.samples.map(s => ({ rgb: s.rgb, role: s.role })),
    });
    if (!r.ok) { setStatus(r.reason); return; }
    S.samples = r.samples;
    S.samplesManual = true;               // 人点过的颜色：自动分色以后不许再覆盖
    S.selSample = S.samples.length - 1;
    S.span = r.span;
    const autoApplied = S.autoTol && typeof r.auto_tol === 'number';
    if (autoApplied) {
      S.tol = r.auto_tol;                   // 服务端给的是整数
      $('#rngTol').value = S.tol; $('#numTol').value = S.tol;
    }
    syncParams(); syncSampleUI(); updateSpan();
    setStatus(T(`取到 ${r.role_name} RGB(${r.rgb.join(',')}) ${r.note}`,
                `Sampled ${r.role_name} RGB(${r.rgb.join(',')}) ${r.note}`) +
      (autoApplied ? T(`　自动容差 → ${S.tol}`, ` · auto tolerance → ${S.tol}`) : ''));
    if (S.samples.length >= 2) analyze(); else setResult(T('再点一下另一个颜色：' +
      (S.samples[0].role === 'coat' ? '露底基材' : '完好漆膜'),
      'Now click the other one: ' + (S.samples[0].role === 'coat' ? 'exposed substrate' : 'intact coating')));
  } catch (err) { setStatus(T('吸色失败：', 'Sampling failed: ') + err.message); }
}

async function measureGroove(a, b) {
  try {
    const r = await post('api/ruler', { id: S.id, res: S.res, p0: [a.x, a.y], p1: [b.x, b.y], on_full: true });
    S.rulerDone = [a.x, a.y, b.x, b.y];
    if (r.ok === false || r.width_full == null) {
      S.rulerLast = null;
      $('#rulerMsg').textContent = r.reason || T('没测到明显的沟槽：让线段垂直于刀缝、长度 30px 以上',
        'No clear groove measured: keep the segment perpendicular to the cut and at least 30 px long');
      draw(); return;
    }
    S.rulerLast = r;
    if (r.reliable) {
      S.rulerHist.push(r.width_analysis);
      if (r.full_analysis) S.rulerFull.push(r.full_analysis);
    }
    const avg = S.rulerHist.length ? S.rulerHist.reduce((x, y) => x + y, 0) / S.rulerHist.length : 0;
    const avgF = S.rulerFull.length ? S.rulerFull.reduce((x, y) => x + y, 0) / S.rulerFull.length : 0;
    const wHalfA = r.width_analysis, wFullA = r.full_analysis || r.width_analysis;
    $('#rulerMsg').textContent = r.reliable
      ? T(`分析图：半高宽 ${fmt(wHalfA, 1)} px（缝心）／全宽 ${fmt(wFullA, 1)} px（含发虚的边）\n` +
        `原图：${fmt(r.width_full, 1)} ／ ${fmt(r.full_full || r.width_full, 1)} px　对比度 ${Math.round(r.contrast)} ✓\n` +
        `已量 ${S.rulerHist.length} 次｜均值 半高 ${fmt(avg, 1)} / 全宽 ${fmt(avgF, 1)}\n` +
        `→ 点「用测量值」会把阈值设成全宽 +1 = ${Math.round(avgF || avg) + 1}（半高宽只是缝心，边缘也得一起排掉）\n` +
        `（这条线 6 秒后自动隐去，数值和均值仍然保留）`,
        `Analysis image: half-max ${fmt(wHalfA, 1)} px (cut centre) / full width ${fmt(wFullA, 1)} px (incl. soft edges)\n` +
        `Original: ${fmt(r.width_full, 1)} / ${fmt(r.full_full || r.width_full, 1)} px  contrast ${Math.round(r.contrast)} ✓\n` +
        `Measured ${S.rulerHist.length}× | mean half-max ${fmt(avg, 1)} / full ${fmt(avgF, 1)}\n` +
        `→ “Use measurement” sets the threshold to full width +1 = ${Math.round(avgF || avg) + 1} (half-max is only the centre; the soft edges must go too)\n` +
        `(the line fades after 6 s; the values and the mean stay)`)
      : T(`分析图：半高宽 ${fmt(wHalfA || 0, 1)} / 全宽 ${fmt(wFullA || 0, 1)} px，但对比度只有 ${Math.round(r.contrast)}，不可靠\n` +
        `换个位置重拖一条，或先点「放大到 100%」`,
        `Analysis image: half-max ${fmt(wHalfA || 0, 1)} / full ${fmt(wFullA || 0, 1)} px, but contrast is only ${Math.round(r.contrast)} — unreliable\n` +
        `Drag again somewhere else, or zoom to 100% first`);
    S.suggestHold = null;                        // 量到新刀缝了，建议重新算
    updateSuggest(); draw();
    rulerFadeSoon();
  } catch (err) { $('#rulerMsg').textContent = T('量尺失败：', 'Measurement failed: ') + err.message; }
}

/* ------------------------------------------------------------------ 分辨率 / 旋转 */
async function changeRes(res) {
  if (!S.id) { S.res = res; return; }
  const oldW = S.w, oldLine = S.line;
  const meta = await post('api/load', { id: S.id, res });
  const f = oldW ? meta.w / oldW : 1;
  S.res = res;
  S.rois = S.rois.map(r => ({ box: r.box.map(v => v * f), ang: r.ang }));
  S.line = oldLine > 1 ? Math.max(1, Math.round(oldLine * f)) : 1;
  if (S.annot.strokes.length && Math.abs(f - 1) > 1e-6) {
    S.annot.strokes.forEach(k => { k.r = k.r * f; k.pts = k.pts.map(([x, y]) => [x * f, y * f]); });
    S.annot.res = null;
  }
  S.overlayUrl = null; S.overlayImg = null; S.contourUrl = null; S.contourImg = null;
  await refetchImage();
  syncParams(); fitView();
  if (S.samples.length) analyze(); else setResult(T('分辨率已切换，选区与刀缝阈值已按比例换算。', 'Resolution changed; regions and cut threshold were rescaled.'));
}

/* ------------------------------------------------------------------ 导出 */
async function download(path, name) {
  const body = Object.assign(payload(), { name: S.name || T('结果', 'result') });
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Baige-Lang': I18N.lang }, body: JSON.stringify(body),
  });
  if (!r.ok) { setStatus(T('导出失败：', 'Export failed: ') + (await r.text()).slice(0, 200)); return; }
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

/* ------------------------------------------------------------------ UI 同步 */
function setResult(text) {
  const el = $('#result'); if (!el) return;
  el.textContent = text; el.style.display = '';   // 提示/报错一律要看得到
}

/* 换图/换记录前，把上一张图的所有派生结果清干净：
   结果卡（大等级、百分比、每选区行）、右上角胶囊、建议刀缝阈值、诊断、块确认、软权重、量尺结果 */
function clearImageState() {
  S.rows = null; S.total = null; S.lines = [];
  S.suggest = null; S.suggestSrc = ''; S.span = null;
  S.autoTotal = null; S.autoRows = null; S.autoLines = null;
  S.blocks = null; S.soft = null;
  S.gm = null; S.grid = null; S.gmDrag = null; S.gmOff = { ox: 0, oy: 0, dx: [], dy: [] };
  S.rulerHist = []; S.rulerFull = []; S.rulerLast = null; S.rulerSeg = null; S.rulerDone = null;
  S.suggestHold = null;
  clearTimeout(S.rulerFadeT);
  setResult(T('打开图片 → 吸两个颜色 → 框出划格区 → 计算\n', 'Open image → sample two colours → frame the cross-hatch area → calculate\n'));
  setStatus('');
  const set = (id, t) => { const e = $(id); if (e) e.textContent = t; };
  const sum = $('#summary');
  if (sum) { sum.textContent = ''; sum.className = 'summary'; }
  const ss = $('#sideSummary');
  if (ss) ss.textContent = '';
  const badge = $('#resBadge');
  if (badge) { badge.textContent = '—'; badge.className = 'rg'; }
  set('#resCardPct', '');
  const rowsBox = $('#resCardRows');
  if (rowsBox) rowsBox.innerHTML = '<div class="rrow"><span>—</span></div>';
  const dx = $('#diagx'); if (dx) dx.innerHTML = '';
  const cm = $('#cmp'); if (cm) cm.innerHTML = '';
  const bl = $('#blockList'); if (bl) bl.innerHTML = '';
  const bi = $('#blockInfo'); if (bi) bi.textContent = '—';
  set('#rulerMsg', '');
  updateSuggest();          // 建议阈值回到占位提示，不再显示上一张图的 n px
  updateKneeHint();
}
function setStatus(text) { $('#hint').textContent = text || ''; }

function syncParams() {
  $('#rngTol').value = S.tol; $('#numTol').value = S.tol;
  $('#rngPurity').value = S.purity; $('#numPurity').value = S.purity;
  $('#rngLine').value = S.line; $('#numLine').value = S.line;
  try { markLineSrc(); } catch (_) {}
  if ($('#selExmode')) $('#selExmode').value = S.exmode;   // 已撤下（只剩固定宽度一种）
  $('#selRes').value = String(S.res);
  $('#chkAutoTol').checked = S.autoTol;
  $('#chkDetail').checked = S.detail;
  $('#chkSemi').checked = S.semi;
  $('#chkOverlay').checked = S.showOverlay;
  updatePurityInfo();
}

/* 与服务端 _auto_tol_of 同一套换算：整数，6~120 */
const autoTolOf = span => (span == null ? null
  : Math.round(Math.max(6, Math.min(120, span * 0.45))));

function updateSpan() {
  const el = $('#spanInfo');
  if (S.span == null) { el.textContent = T('样本间距：—（还缺漆膜或基材）', 'Colour distance: — (coating or substrate missing)'); return; }
  el.textContent = T(`样本间距 ${Math.round(S.span)}（Lab 距离）→ 建议容差 ${autoTolOf(S.span)}`,
                     `Colour distance ${Math.round(S.span)} (Lab) → suggested tolerance ${autoTolOf(S.span)}`) +
    (S.autoTol ? T(`　· 当前自动采用 ${S.tol}`, ` · using ${S.tol} automatically`)
               : T(`　· 手动 ${S.tol}`, ` · manual ${S.tol}`));
}

/* 向服务端要真实的样本间距 / 建议容差；自动容差开着就立刻套用 */
async function refreshSpan() {
  if (!S.samples.length) { S.span = null; updateSpan(); return; }
  try {
    const r = await post('api/samples', {
      samples: S.samples.map(s => ({ rgb: s.rgb, role: s.role })),
    });
    S.span = r.span;
    if (S.autoTol && r.auto_tol != null) {
      S.tol = r.auto_tol;                       // 服务端给的就是整数
      $('#rngTol').value = S.tol;
      $('#numTol').value = S.tol;
    }
  } catch (err) { /* 间距算不出来不影响主流程 */ }
  updateSpan();
}

function updateSuggest() {
  const btn = $('#btnSuggest'), info = $('#suggestInfo');
  if (!btn || !info) return;      // 采用建议按钮与提示行已撤下
  if (S.suggestHold != null) {
    info.textContent = T(`已采用 ${S.suggestHold} px（改容差/纯度/选区/量尺后会重新建议）`,
      `Using ${S.suggestHold} px (re-suggested after changing tolerance, purity, region or the ruler)`);
    btn.disabled = false; btn.textContent = T(`采用 ${S.suggestHold} px`, `Use ${S.suggestHold} px`);
    return;
  }
  if (S.suggest == null) {
    info.textContent = T('建议刀缝阈值：先吸色算一次，或量一条刀缝（成片脱落为主的图只能靠量尺）',
      'Suggested cut threshold: sample the colours and calculate once, or measure a cut (images dominated by sheet removal can only be measured)');
    btn.disabled = true; btn.textContent = T('采用建议', 'Use suggestion');
  } else {
    info.textContent = T(`建议刀缝阈值 ≈ ${S.suggest} px（${S.suggestSrc || '按露底厚度'}）`,
      `Suggested cut threshold ≈ ${S.suggest} px (${S.suggestSrc || 'from bare thickness'})`);
    btn.disabled = false; btn.textContent = T(`采用 ${S.suggest} px`, `Use ${S.suggest} px`);
  }
}

function refreshRoiList() {
  const box = $('#roiList');
  if (!S.rois.length) {
    box.innerHTML = '<div class="empty">' + T('还没有选区：在图上拖出划格区', 'No region yet — drag a box around the cross-hatch area') + '</div>';
    return;
  }
  box.innerHTML = '';
  S.rois.forEach((r, i) => {
    const st = (S.rows && S.rows.length === S.rois.length) ? S.rows[i] : null;
    const el = document.createElement('div');
    el.className = 'roiitem' + (i === S.sel ? ' on' : '');
    el.innerHTML = `<span class="nm">R${i + 1}</span>
      <span class="pc">${st ? fmt(st.pct, 2) + '%' : '—'}</span>
      <span class="cl">${st ? st.astm : '—'}</span>
      <button class="del" title="删除">×</button>`;
    el.onclick = ev => {
      if (ev.target.classList.contains('del')) { deleteRoi(i); return; }
      S.sel = i; refreshRoiList(); syncRoiAngle(); draw();
    };
    box.appendChild(el);
  });
}

function dropOverlay() {
  // 覆盖层是按"当前这组选区"算的；选区一改，旧的那张就是错的，立刻撤掉。
  // seq 一定要跟着 +1：在飞的分析算的是旧的这组选区，回来时不能再往画布上贴图
  // （否则删完框等一两秒，浅色区域会自己"变回来"）。
  S.seq++;
  if (S.autoAbort) { try { S.autoAbort.abort(); } catch (_) {} S.autoAbort = null; }
  setBusy(false);
  if (!S.overlayUrl && !S.overlayImg) return;
  S.overlayUrl = null; S.overlayImg = null; S.contourUrl = null; S.contourImg = null;
}

/* 清空全部选区（按钮以前没接事件，点了没反应） */
function clearAllRois() {
  if (!S.rois.length) return;
  S.rois = []; S.sel = -1;
  S.gm = null; S.gmDrag = null; S.gmOff = { ox: 0, oy: 0, dx: [], dy: [] };   // 没选区了，格网不该还挂在那儿
  if ($('#gmInfo')) $('#gmInfo').textContent = '';
  if ($('#gmTrace')) $('#gmTrace').textContent = '';
  dropOverlay();
  refreshRoiList(); syncRoiAngle(); draw();
  analyze();                       // 没选区 → 走"清空结果"分支
}

function deleteRoi(i) {
  S.rois.splice(i, 1);
  if (S.sel >= S.rois.length) S.sel = S.rois.length - 1;
  // 删掉选区，属于它的块（蚂蚁线/块号）必须跟着消失。原来的 block 数据是按 ROI 下标存的，
  // 不删就会留在画布上；而且 rois 一变 blocksSig() 就对不上 → 整片被画成 32% 透明，
  // 用户看到的就是“块没消失、只是变浅了”。这里连数据和签名一起收拾干净。
  let hadBlocks = false;
  if (S.blocks && Array.isArray(S.blocks.rois)) {
    if (i < S.blocks.rois.length) { S.blocks.rois.splice(i, 1); hadBlocks = true; }
    if (Array.isArray(S.blocks.excluded) && i < S.blocks.excluded.length) S.blocks.excluded.splice(i, 1);
    if (!S.blocks.rois.length) S.blocks = null;          // 一个选区都不剩 → 别再画
    else {
      try { S.blocks.sig = blocksSig(); } catch (_) {}    // 剩下的块仍然有效，别让它们变淡
      try { renderBlocks(); } catch (_) {}
    }
  }
  if (hadBlocks) { try { applyBlocks(); } catch (_) {} }
  // 格网挂在选地上：最后一个选区没了，格网必须先撤掉再重画（否则 draw 会拿旧格网配上已删的选区）
  const lastGone = (S.exmode === 'grid' && !S.rois.length && S.gm);
  if (lastGone) S.gm = null;
  dropOverlay();
  refreshRoiList(); syncRoiAngle(); draw();
  if (S.exmode === 'grid') {
    if (S.rois.length) scheduleGm();
    else if (lastGone) { if ($('#gmInfo')) $('#gmInfo').textContent = ''; if ($('#gmTrace')) $('#gmTrace').textContent = ''; }
  }
  else if (S.samples.length) debouncedAnalyze();
}

function syncSampleUI() {
  updateKneeHint();
  const box = $('#sampleList');
  if (!S.samples.length) {
    box.innerHTML = '<div class="empty">' + T('还没有样本：图上点一下完好漆膜，再点一下露底基材',
      'No colour samples yet — click the intact coating, then the exposed substrate') + '</div>';
    $('#spanInfo').textContent = T('样本间距：—（先吸漆膜和基材）', 'Colour distance: — (sample coating and substrate first)');
    return;
  }
  box.innerHTML = '';
  S.samples.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'sitem' + (i === S.selSample ? ' on' : '');
    el.innerHTML = `<span class="chip" style="background:rgb(${s.rgb.join(',')})"></span>
      <span class="rl">${roleName(s.role)}</span>
      <span class="rgb">RGB ${s.rgb.join(',')}</span>
      <button class="x" title="${T('删除', 'Delete')}">×</button>`;
    el.onclick = ev => {
      if (ev.target.classList.contains('x')) {
        S.samples.splice(i, 1);
        if (S.selSample >= S.samples.length) S.selSample = S.samples.length - 1;
        afterSamplesChanged();
        return;
      }
      S.selSample = i; syncSampleUI();
    };
    box.appendChild(el);
  });
  updateSpan();
}

function roleName(r) { return I18N.lang === 'en' ? (r === 'sub' ? 'substrate' : 'coating')
                                                  : (r === 'sub' ? '基材' : '漆膜'); }   // v1.42：只有两个角色

/* 样本角色改动后的统一收尾：刷新列表 → 重算自动容差 → 重算结果 */
async function afterSamplesChanged() {
  S.overlayUrl = null; S.overlayImg = null; S.contourUrl = null; S.contourImg = null;
  syncSampleUI();
  await refreshSpan();                 // 角色变了间距就变，自动容差跟着走
  const hasCoat = S.samples.some(s => s.role === 'coat');
  const hasSub = S.samples.some(s => s.role === 'sub');
  if (S.samples.length >= 2 && hasCoat && hasSub) {
    if (S.exmode === 'grid') scheduleGm(); else analyze();     // 格网判法只重算格网
  }
  else setResult(T('颜色角色还没齐：至少需要一行「漆膜」和一行「基材」。\n' +
    '在下面的列表里点一行选中，再点上面的「设为漆膜 / 设为基材」按钮改过来。',
    'The colour roles are not complete: you need at least one coating and one substrate row.\n' +
    'Select a row in the list below, then use the “Set as coating / Set as substrate” buttons.'));
}

function setSampleRole(role) {
  if (S.selSample < 0 && S.samples.length) { S.selSample = 0; syncSampleUI(); }
  const i = S.selSample;
  if (i < 0 || !S.samples[i]) { setStatus(T('还没有颜色样本：先在图上点一下漆膜、再点一下基材', 'No colour samples yet — click the coating, then the substrate')); return; }
  if (S.samples[i].role === role) { setStatus(T('这一行已经是「' + roleName(role) + '」了', 'This row is already “' + roleName(role) + '”')); return; }
  S.samples[i].role = role;
  S.samplesManual = true;          // 人改过的颜色：换框、自动分色都不许再动
  setStatus(T(`第 ${i + 1} 个样本（RGB ${S.samples[i].rgb.join(',')}）→ ${roleName(role)}`,
    `Sample ${i + 1} (RGB ${S.samples[i].rgb.join(',')}) → ${roleName(role)}`));
  afterSamplesChanged();
}

/* 已不绑按钮，保留给键盘 Delete 用；样本行右侧的 × 也走这里 */
function delSampleSel() {
  if (S.selSample < 0 && S.samples.length) { S.selSample = 0; syncSampleUI(); }
  const i = S.selSample;
  if (i < 0 || !S.samples[i]) { setStatus(T('还没有颜色样本可删', 'No colour sample to delete')); return; }
  S.samples.splice(i, 1);
  S.samplesManual = true;          // 删掉的也是人定的
  S.selSample = Math.min(S.samples.length - 1, i);
  afterSamplesChanged();
}

function syncRoiAngle() {
  const el = $('#roiAng');           // 数值旋转已从界面撤下（只在画布上拖手柄转）
  if (!el) return;
  if (S.sel >= 0 && S.rois[S.sel]) el.value = (S.rois[S.sel].ang || 0).toFixed(1);
  else el.value = '0';
}

/* ------------------------------------------------------------------ 事件绑定 */
$('#btnOpen').onclick = () => $('#fileInput').click();
$('#fileInput').onchange = e => { upload(e.target.files[0]); e.target.value = ''; };
$('#btnCalc').onclick = () => {
  if (S.exmode === 'grid') { scheduleGm(60); return; }   // 格网判法下「计算」= 重算格网，不跑像素分析
  analyze();
};
$('#btnFit').onclick = fitView;
$('#btnOne').onclick = zoomOne;
$('#btnReport').onclick = () => download('api/export/report', (S.name || T('结果', 'result')).replace(/\.[^.]+$/, '') + T('_百格报告.pdf', '_crosscut_report.pdf'));
$('#btnPng').onclick = () => download('api/export/png', (S.name || T('结果', 'result')).replace(/\.[^.]+$/, '') + T('_标注.png', '_annotated.png'));

/* ---- 结果栏高度：拖顶部把手上下调（往上 = 变高），双击复原 ----
   上下都留位置：最少 120px，上面固定留 110px 给参数区，拉不出视野。 */
(function () {
  const grip = document.getElementById('resGrip');
  const box = document.getElementById('resultBox');
  if (!grip || !box) return;
  const MIN = 120, DEF = 300, PANE_KEEP = 100;   // 上面给参数区至少留 100px（不含页签）
  const tabsH = () => (document.querySelector('.side .tabs') || {}).offsetHeight || 42;
  const maxH = () => Math.max(MIN, (box.parentElement.clientHeight || 600) - tabsH() - PANE_KEEP);
  const apply = h => { box.style.height = Math.min(maxH(), Math.max(MIN, h)) + 'px'; };
  const saved = parseInt(localStorage.getItem('baige.resH'), 10);
  if (saved > 0) apply(saved);

  let sy = 0, sh = 0, on = false;
  grip.addEventListener('pointerdown', e => {
    on = true; sy = e.clientY; sh = box.getBoundingClientRect().height;
    grip.classList.add('drag');
    try { grip.setPointerCapture(e.pointerId); } catch (_) {}
    e.preventDefault();
  });
  grip.addEventListener('pointermove', e => { if (on) apply(sh - (e.clientY - sy)); });
  const end = e => {
    if (!on) return;
    on = false; grip.classList.remove('drag');
    try { grip.releasePointerCapture(e.pointerId); } catch (_) {}
    localStorage.setItem('baige.resH', String(parseInt(box.style.height, 10) || DEF));
  };
  grip.addEventListener('pointerup', end);
  grip.addEventListener('pointercancel', end);
  grip.addEventListener('dblclick', () => { apply(DEF); localStorage.setItem('baige.resH', String(DEF)); });
  window.addEventListener('resize', () => {
    const h = parseInt(box.style.height, 10);
    if (h) apply(h);
  });
})();
$('#chkOverlay').onchange = e => { S.showOverlay = e.target.checked; prefSet('overlay', S.showOverlay); draw(); };
$('#chkSemi').onchange = e => {
  S.semi = e.target.checked;
  prefSet('semi', S.semi);
  if (S.semi) runSemi();
};
$('#chkDetail').onchange = e => {
  S.detail = e.target.checked;
  prefSet('detail', S.detail);
  renderResult();                                  // 先按本地数据立刻切换显示
  if (S.samples.length >= 2) analyze();            // 再让服务器按需补算/省算明细
};
function setSoft(on) {
  S.softOn = on;
  const a = $('#chkSoft'); if (a) a.checked = on;
  const b = $('#chkSoftR'); if (b) b.checked = on;
  try { localStorage.setItem('baige.soft', on ? '1' : '0'); } catch (_) {}
  if (!S.id) return;
  if (on) { analyze(); }
  else { S.soft = null; if (S.total) renderResult(); setStatus(T('已关闭软权重，只显示 v1 口径', 'Soft weighting off — showing the v1 figures only')); }
}
if ($('#chkSoftR')) $('#chkSoftR').onchange = e => setSoft(e.target.checked);   // 开关已撤下

/* 选区原生：只把框里那一点按原图像素算（v1.31，默认关） */
function setRoiNative(on) {
  S.roiNative = !!on;
  const cb = $('#chkRoiNative');
  if (cb) cb.checked = S.roiNative;
  prefSet('roiNative', S.roiNative);          // 记住（以前每次刷新都强制回"开"，改了不生效）
  S.blocks = null;
  try { renderBlocks(); } catch (_) {}
  if (!S.id) return;
  if (!S.rois.length) {
    setStatus(S.roiNative ? T('已开「选区原生」：框完选区再生效（只分析框内那块）',
                              'Native region on: applies once a region is framed (only the box is analysed)')
                          : T('已关「选区原生」：按整图 6MP 分析',
                              'Native region off: the whole image is analysed at 6 MP'));
    return;
  }
  if (S.samples.length >= 2) analyze();
  else setStatus(S.roiNative ? T('已开「选区原生」：吸两个颜色后再计算', 'Native region on: sample two colours to calculate')
                             : T('已关「选区原生」', 'Native region off'));
}

$('#chkRoiNative').onchange = e => setRoiNative(e.target.checked);

/* 刀缝判法只剩两种：按块数回落（默认）/ 按宽度。
   旧记录里存的 shape/local/auto 已停用，打开时一律按「按宽度」+ 记录里的 k 还原，
   并在状态栏说明数字可能与存档不同。*/
const EXMODE_TXT = {
  auto_blk: '按块数回落：自动扫 k=1…16，块数先从低急升到峰值、再回落。①取过峰值后第一档块数 ≤ 峰值 25% 的 k；②若进这一档时露底还在大跌（≥2 个百分点），说明还没走平，继续往下一档；③走平的那一档当刀缝宽度，再按这个宽度算。扫一次约 4 秒；下面的曲线可以点任意一档手动覆盖。',
  width: '按露底分布：看露底厚度分布给一个刀缝阈值（薄结构占主导时才信它），比阈值粗的算脱落、其余全算刀缝。这个值你可以随时手改；换个判法再回来会重算。',
  grid: '按刀缝格网（半自动）：先把每条刀缝认出来、线铺在缝正中，再逐格统计；列/行在「框选」页，刀宽就是上面那个「忽略细线宽度」。框完选区会自动跑完分色→认缝→铺格→统计，不用再点按钮；改列/行/刀宽会立刻重跑。'
};
const EXMODE_TXT_EN = {
  auto_blk: 'By block count: automatically scans k=1…16. The block count rises steeply to a peak, then falls. '
    + '(1) take the first k past the peak whose block count is ≤25% of the peak; (2) if the bare area is still '
    + 'dropping sharply (≥2 pp) at that k, the curve has not levelled off — keep going; (3) the first level k is '
    + 'the cut width, and the result is computed with it. One scan takes ~4 s; click any row of the curve to '
    + 'override it by hand.',
  width: 'By bare distribution: derives a cut threshold from the distribution of bare-pixel thickness (only '
    + 'trustworthy when thin structures dominate). Anything thicker than the threshold counts as removal, the '
    + 'rest as cuts. You can change the value at any time; switching the method and coming back recomputes it.'
};
function exmodeTxt(v) { return I18N.lang === 'en' ? (EXMODE_TXT_EN[v] || '') : (EXMODE_TXT[v] || ''); }
const RETIRED_MODES = { auto: 1, shape: 1, local: 1 };
/* 只同步状态与界面，不触发计算（打开记录时要用它，否则会先按错的判法算一遍） */
function applyExmode(v) {
  const retired = !!RETIRED_MODES[v];
  S.exmode = (v === 'auto_blk' || v === 'grid') ? v : 'width';   // 停用的判法一律回落到按宽度
  S.kneeWarn = '';                                    // 上一条判法的告警不能留到这一条
  try { updateKneeHint(); } catch (_) {}               // 面板立刻换成这一条判法的说明
  const el = $('#selExmode'); if (el) el.value = S.exmode;
  if (retired && S.line > 0) {
    setStatus(T('这条记录原来用的是「' + ({ auto: '按露底突增', shape: '按形状', local: '按局部宽度' }[v])
      + '」（已停用），现按「按露底分布 ' + S.line + ' px」重算，数字可能与存档略有不同',
      'This record used “' + ({ auto: 'by bare jump', shape: 'by shape', local: 'by local width' }[v])
      + '” (retired); it is recomputed with “by bare distribution, ' + S.line + ' px”, so numbers may differ '
      + 'slightly from the archive'));
  }
  try { localStorage.setItem('baige.exmode', S.exmode); } catch (_) {}
  const info = $('#exmodeInfo');
  if (info) info.textContent = exmodeTxt(S.exmode);
  // 判法=刀缝格网 → 格网块（列/行/歪角）直接露出来，这就是半自动的操作面
  try { syncGmWrap(); } catch (_) {}          // 判法=grid 才把格网参数露出来
  try { refreshGmK(); } catch (_) {}
  try { updateKneeHint(); } catch (_) {}     // 换判法立刻刷新/隐藏 k 曲线
}
function setExmode(v) {
  applyExmode(v);
  S.lineManual = false;                                  // 只要切换判法，手改标记一律作废
  S.widthTouched = false; S.suggRetry = false;            // 新判法要能覆盖你之前设的宽度
  S.kneeWarn = '';                                        // 换判法 = 换判据，上一条黄条的理由不一定还成立
  if (v === 'auto' || v === 'auto_blk') S.knee = null;    // 重选自动判法 = 重新扫、重新套用
  else if (v === 'width') setStatus(T('按露底分布：沿用当前 ' + S.line + ' px（手改标记已清；切回按块数回落会重新按判法定 k）',
        `By bare distribution: keeping ${S.line} px (the manual flag is cleared; switching back to block count re-derives k)`));
  else if (v === 'grid') {
    S.knee = null;                                       // 换判据了，上一条判法的曲线不能留着（格网下 k=刀宽，重新扫）
    // 半自动：框选由人做，认缝/分色/铺格/统计由程序做完；列/行/刀宽留给人数值微调
    setStatus(T('按刀缝格网：在图上框住整片百格就自动跑完（列/行在上面，刀宽=量尺页的「忽略细线宽度」）',
      'By cut grid: just frame the whole grid — tracing, colouring and statistics run automatically'));
    scheduleGm(300);
  }
  S.blocks = null;
  // 判法=格网：只跑格网（认缝/铺格/统计），不跑像素分析——否则切过去会莫名其妙扫一遍 k、
  // 顺带跑「按块数回落」，用户看到的就是“选了第三种还在算原来的东西”。
  if (S.exmode === 'grid') {
    try { renderKnee(); } catch (_) {}
    scheduleGm(300);
  } else if (S.id && S.samples.length >= 2 && S.rois.length) analyze();
}
$('#selExmode').onchange = e => setExmode(e.target.value);

/* 点块确认：不用画笔，点一下某块 = 排除 */
function blocksSig() {
  // 判法(mode)、样本颜色也会改变块的分法，必须进签名——否则换了判法还拿旧编号沿用排除，
  // 那就是把“排除第 7 块”套到一个完全不同的块上。
  return JSON.stringify({ k: S.line, tol: S.tol, pur: S.purity, res: S.res, rn: S.roiNative ? 1 : 0,
                          md: S.exmode, lt: S.locTile || null, lf: S.locFactor || null,
                          s: S.samples.map(x => x.rgb.join(',') + x.role),
                          r: S.rois.map(x => x.box.map(v => Math.round(v * 10) / 10).concat([x.ang || 0])) });
}

async function listBlocks(silent) {
  S.blocksDue = false; refreshCalcBtn();
  if (!S.id) return;
  if (!S.samples.length) { if (!silent) setStatus(T('先吸两个颜色再列块', 'Sample two colours before listing blocks')); return; }
  if (!S.rois.length) { if (!silent) setStatus(T('先框一个选区', 'Frame a region first')); return; }
  if (!silent) setStatus(T('列出露底块…', 'Listing bare blocks…'));
  const my = ++S.blocksSeq;
  const sig = blocksSig();
  setBlocksBusy(true);
  const keepEx = (S.blocks && S.blocks.sig === sig) ? S.blocks.excluded : null;
  try {
    const r = await post('api/blocks', {
      id: S.id, res: S.res,
      rois: S.rois.map(x => ({ box: x.box, ang: x.ang || 0 })),
      samples: S.samples.map(s => ({ rgb: s.rgb, role: s.role })),
      tol: S.tol, purity: S.purity, line_px: S.line, mode: srvMode(),
      loc_tile: S.locTile || 64, loc_factor: S.locFactor || 1.6,
      roi_native: !!S.roiNative,
    });
    if (my !== S.blocksSeq) return;                 // 又来了新的一次，这次结果作废
    if (!r.ok) { if (!silent) setStatus(r.reason || T('列块失败', 'Listing blocks failed')); return; }
    const excluded = r.rois.map((x, ri) => {
      const old = keepEx && keepEx[ri];
      if (!old) return new Set();
      return new Set([...old].filter(i => i < x.blocks.length));   // 同参数改回来时保留排除
    });
    S.blocks = { k: r.k, rois: r.rois, excluded, sig, hover: -1 };
    draw(); renderBlocks();
    const n = r.rois.reduce((a, x) => a + x.blocks.length, 0);
    if (!silent) setStatus(T(`列出 ${n} 块`, `${n} blocks listed`));
    else setStatus(tf('status.autoBlocks', n));
  } catch (err) {
    if (!silent) setStatus(T('列块失败：', 'Listing blocks failed: ') + err.message);
  } finally {
    if (my === S.blocksSeq) setBlocksBusy(false);
  }
}

function blocksCalc() {
  const B = S.blocks;
  if (!B || !B.rois || !B.rois.length) return null;
  let num = 0, den = 0, keptAll = 0, exN = 0, exA = 0;
  const per = [];
  B.rois.forEach((r, ri) => {
    const ex = B.excluded[ri];
    let inc = r.kept;
    r.blocks.forEach(b => { if (ex.has(b.i)) { inc -= b.area; exN++; exA += b.area; } });
    per.push({ pct: r.denom ? 100 * inc / r.denom : 0, auto: r.denom ? 100 * r.kept / r.denom : 0 });
    num += inc; den += r.denom; keptAll += r.kept;
  });
  if (!den) return null;
  const pct = 100 * num / den;
  return { pct, lo: 100 * num / B.rois.reduce((a, r) => a + r.A, 0), astm: gradeOf(pct), iso: isoOf(pct),
           auto: 100 * keptAll / den, exN, exA, n: B.rois.reduce((a, r) => a + r.blocks.length, 0), per };
}

function gradeOf(p) {
  if (p <= 0.5) return '5B';
  if (p < 5) return '4B';
  if (p < 15) return '3B';
  if (p < 35) return '2B';
  if (p < 65) return '1B';
  return '0B';
}
function isoOf(p) {
  if (p <= 0.5) return 'Gt0';
  if (p < 5) return 'Gt1';
  if (p < 15) return 'Gt2';
  if (p < 35) return 'Gt3';
  if (p < 65) return 'Gt4';
  return 'Gt5';
}

/* 逐块排除要影响所有显示（右上胶囊、R1 标签、结果区），不能让用户再点一次计算 */
function applyBlocks() {
  if (!S.autoTotal) return;
  const R = blocksCalc();
  const t = S.autoTotal;
  if (!R || R.exN === 0) {
    S.total = { ...t, worst: t.worst ? { ...t.worst } : null };
  } else {
    S.total = { ...t, pct: R.pct, astm: R.astm, iso: R.iso, worst: null, manual: R.exN };
  }
  if (S.autoRows && S.rows && R && R.per.length === S.rows.length) {
    const prev = S.rows.map(x => x.pct);
    S.rows = S.autoRows.map(x => ({ ...x }));
    R.per.forEach((p, i) => {
      S.rows[i].pct = p.pct;
      S.rows[i].astm = gradeOf(p.pct);
      S.rows[i].iso = isoOf(p.pct);
    });
    if (S.rows.some((x, i) => Math.abs((prev[i] ?? -1) - x.pct) > 1e-9)) S.needRedraw = true;
  }
  // 结果区里 "R1  1.96%  4B" 这行也要跟着改，否则跟上面的胶囊对不上
  const base = S.autoLines || S.lines;
  const out = base.slice();
  if (R && R.exN > 0 && S.rows) {
    S.rows.forEach((row, i) => {
      const re = new RegExp('^R' + (i + 1) + '\\s');
      const k = out.findIndex(s => re.test(s));
      if (k >= 0) out[k] = out[k].replace(/(R\d+\s+)[\d.]+%(\s+)\S+/, `$1${fmt(row.pct, 2)}%$2${row.astm}`);
    });
    const mi = out.findIndex(s => s.startsWith('判定余量') || s.startsWith('Margin:'));
    if (mi >= 0) out[mi] = marginHint(R.per.length === 1 ? R.per[0].pct : R.pct);
  }
  S.lines = out;
  // 选区列表（框选页左上「R1 29.89% 3B」）也要跟着手动排除变，否则同一屏上两处数字不一样。
  // 点块排除走的是 applyBlocks → 以前没人刷这个列表。
  try { refreshRoiList(); } catch (_) {}
}

function blocksText() {
  const R = blocksCalc();
  if (!R) return '';
  const d = R.pct - R.auto;
  const dTxt = R.exN ? `${d < 0 ? '−' : '+'}${fmt(Math.abs(d), 2)} pp`
                     : T('未排除任何块', 'no blocks excluded');
  return [
    T('——— 逐块确认（自动重算，不用点「计算」）———', '——— block review (recomputed automatically, no need to press Calculate) ———'),
    T(`共 ${R.n} 块，已排除 ${R.exN} 块（${fmt(R.exA, 0)} px）`,
      `${R.n} blocks, ${R.exN} excluded (${fmt(R.exA, 0)} px)`),
    T(`脱落 ${fmt(R.pct, 2)}% → ${R.astm} / ${R.iso}　${dTxt}（自动 ${fmt(R.auto, 2)}%）`,
      `removal ${fmt(R.pct, 2)}% → ${R.astm} / ${R.iso}  ${dTxt} (auto ${fmt(R.auto, 2)}%)`),
  ].join('\n');
}

function renderBlocks() {
  const box = $('#blockList'), info = $('#blockInfo');
  const B = S.blocks;
  if (!B) {
    box.innerHTML = '<div class="empty">' + T('还没列出。框好选区后会自动列（勾着「自动列出」时）。',
      'Nothing listed yet — with the switch on, blocks appear once a region is framed.') + '</div>';
    info.textContent = '—';
    if (S.total) renderResult();
    return;
  }
  const R = blocksCalc();
  const rows = [];
  if (!S.blocksOn) {
    rows.push('<div class="recnone" style="padding:6px 8px">'
      + T('块显示已关闭：不再计算、图上也不画。已排除的记录（'
          + B.excluded.reduce((a, x) => a + x.size, 0) + ' 个）先留着，参数或判法一改就清空。',
          'Block display is off: nothing is computed or drawn. The ' 
          + B.excluded.reduce((a, x) => a + x.size, 0)
          + ' exclusions are kept until the parameters or method change.')
      + '</div>');
  }
  // 每个选区自己的块号都从 1 开始，必须分组显示，否则 4 个选区就有 4 个“1”，看不出谁是谁
  B.rois.forEach((r, ri) => {
    const sr = (S.rows && S.rows[ri]) || null;
    const exN = B.excluded[ri] ? B.excluded[ri].size : 0;
    // 每个选区一个“独立盒子”：分组头只在自己的盒子里吸顶，滚出这个盒子就跟着走，
    // 否则多个 sticky 头都停在 top:0，滚到后面 R1 R2 R3 R4 会叠在一起。
    rows.push(`<div class="recgrp" data-roi="${ri}"><div class="recg">
      <span class="gname">R${ri + 1}</span>
      <span class="gmeta">${sr ? fmt(sr.pct, 2) + '% · ' + sr.astm : '—'} ｜ ${r.blocks.length}${T(' 块', ' blocks')}${exN ? T(' · 已排除 ' + exN, ' · ' + exN + ' excluded') : ''}</span>
    </div>`);
    if (!r.blocks.length) {
      rows.push('<div class="recnone">' + T('这个选区里没有够大的露底块。', 'No bare block large enough in this region.') + '</div></div>');
      return;
    }
    r.blocks.forEach(b => {
      const ex = B.excluded[ri].has(b.i);
      const share = r.kept ? 100 * b.area / r.kept : 0;
      rows.push(`<div class="reckitem" data-roi="${ri}" data-i="${b.i}">
        <span class="rk">${b.i + 1}</span>
        <span class="rv">${fmt(b.area, 0)} px · ${T('占露底 ', 'of bare ')}${fmt(share, 1)}%</span>
        <button class="mini ${ex ? '' : 'on'}" data-act="toggle" data-roi="${ri}" data-i="${b.i}">${ex ? T('已排除', 'excluded') : T('算脱落', 'counts as removal')}</button>
      </div>`);
    });
    rows.push('</div>');
  });
  box.innerHTML = rows.join('') || ('<div class="empty">' + T('还没有块。框一个选区、或者调一下刀缝阈值就会自动列出来。',
    'No blocks yet — frame a region or change the cut threshold and they are listed automatically.') + '</div>');
  info.textContent = R
    ? [T(`共 ${R.n} 块，已排除 ${R.exN} 块（${fmt(R.exA, 0)} px）`,
         `${R.n} blocks, ${R.exN} excluded (${fmt(R.exA, 0)} px)`),
       T(`脱落 ${fmt(R.pct, 2)}% → ${R.astm} / ${R.iso}`,
         `removal ${fmt(R.pct, 2)}% → ${R.astm} / ${R.iso}`),
       T(`自动口径 ${fmt(R.auto, 2)}%（不减任何块）`,
         `automatic basis ${fmt(R.auto, 2)}% (no blocks excluded)`)].join('\n')
    : '—';
  if (S.total) renderResult();
}

function toggleBlock(ri, i) {
  const B = S.blocks;
  if (!B || !B.rois[ri]) return;
  if (B.sig !== blocksSig()) { setStatus(T('块正在按新参数更新，稍等一下再点', 'Blocks are updating for the new parameters — wait a moment')); return; }
  const ex = B.excluded[ri];
  if (ex.has(i)) ex.delete(i); else ex.add(i);
  draw(); renderBlocks();
}

/* 一个开关管两件事：显示就自动列块（调参数实时更新），不显示就压根不算。
   以前是“自动列出 / 图上显示”两个勾，勾了不显示也照样算一遍，白跑。 */
function setShowBlocks(on) {
  S.blocksOn = !!on;
  const el = $('#chkBlocks'); if (el) el.checked = S.blocksOn;
  try { localStorage.setItem('baige.showBlocks', S.blocksOn ? '1' : '0'); } catch (_) {}
  if (!S.blocksOn) {
    // 只停“算”和“画”：已排除的块留在缓存里，勾回来不用重算也还在。
    // 一旦参数/判法/选区变了，blocksSig() 对不上，下面的分支和 analyze 回来的校验会把它清掉。
    clearTimeout(S.blocksTimer);
    renderBlocks();
  } else if (S.blocks && S.blocks.sig === blocksSig()) {
    draw(); renderBlocks();                              // 缓存还有效：不动后端
  } else {
    if (S.blocks) { S.blocks = null; renderBlocks(); }
    if (S.id && S.rois.length && S.samples.length >= 2) listBlocks(true);
  }
  draw();
}
$('#chkBlocks').onchange = e => {
  setShowBlocks(e.target.checked);
  setStatus(e.target.checked ? T('块已显示（调参数会实时更新）', 'Blocks shown (updates live as you change parameters)')
                             : T('块已隐藏，也不再计算', 'Blocks hidden and no longer computed'));
};
$('#bReset').onclick = () => {
  if (!S.blocks) return;
  S.blocks.excluded = S.blocks.rois.map(() => new Set());
  draw(); renderBlocks();
  setStatus(T('已恢复全部块', 'All blocks restored'));
};


$('#blockList').onclick = e => {
  const b = e.target.closest('[data-act="toggle"]');
  if (!b) return;
  toggleBlock(+b.dataset.roi, +b.dataset.i);
};

/* ------------------------------------------------------------------ 记录（手工保存） */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

async function refreshRecords() {
  const box = $('#recList'), stat = $('#recStat');
  try {
    const r = await post('api/records', undefined, 'GET');
    if (!r.ok) throw new Error(r.reason || T('读不到记录', 'cannot read records'));
    $('#recCount').textContent = r.total;
    const mb = r.bytes / 1e6;
    stat.textContent = T(`${r.total}/${r.max} 条 · ${fmt(mb, 1)} MB`, `${r.total}/${r.max} records · ${fmt(mb, 1)} MB`)
      + (r.free_mb != null ? T(` · 服务器剩 ${fmt(r.free_mb / 1000, 1)} GB`, ` · ${fmt(r.free_mb / 1000, 1)} GB free on the server`) : '');
    if (!r.items.length) {
      box.innerHTML = '<div class="empty">' + T('还没有记录。算完之后点结果栏右上角的「保存到记录」。',
        'No records yet — after an analysis, click “Save to records” at the top right of the result box.') + '</div>';
      return;
    }
    box.innerHTML = r.items.map(it => {
      const grade = it.worst_roi ? (it.worst_pct != null ? fmt(it.worst_pct, 2) : '—') : (it.pct != null ? fmt(it.pct, 2) : '—');
      const g = it.astm || '—';
      const cur = (it.id === S.curRid) ? ' cur' : '';
      return `<div class="recitem${cur}" data-id="${it.id}">
        <img src="api/records/${it.id}/thumb.jpg?v=${Date.now()}" alt="" loading="lazy">
        <div class="recmeta">
          <div class="recname" data-name="${it.id}">${esc(it.name)}</div>
          <div class="recsub">${esc(it.created || '')}　${it.w}×${it.h}　k=${it.line_px}　tol=${it.tol}</div>
          <div class="recsub"><b>${grade}%</b> ${g}${it.manual ? T(`　✎排除${it.manual}块`, ` · ✎${it.manual} excluded`) : ''}${it.worst_roi ? T(`　最差 R${it.worst_roi}`, ` · worst R${it.worst_roi}`) : ''}${T(`　${it.n_roi} 个选区`, ` · ${it.n_roi} region(s)`)}</div>
          ${it.gm ? `<div class="recsub" data-i18n-skip>${T('格网', 'grid')} ${it.gm.cols}×${it.gm.rows}${it.gm.n ? ` · ${it.gm.n} 格` : ''}${it.gm.pct != null ? ` · ${fmt(it.gm.pct, 1)}%` : ''}${it.gm.angle ? ` · ${it.gm.angle}°` : ''}</div>` : ''}

        </div>
        <button class="ren" data-ren="${it.id}" title="${T('重命名', 'Rename')}">✎</button>
        <button class="del" data-del="${it.id}" title="${T('删除（服务器上一起删）', 'Delete (also from the server)')}">×</button>
      </div>`;
    }).join('');
  } catch (e) {
    box.innerHTML = `<div class="empty">${T('读取失败：', 'Load failed: ')}${esc(e.message)}</div>`;
  }
}

/* 重命名：把那行的名字换成输入框，回车保存，Esc 取消；空名字不提交 */
function startRename(rid) {
  const box = document.querySelector(`[data-name="${rid}"]`);
  if (!box || box.querySelector('input')) return;
  const old = box.textContent.trim();
  const inp = document.createElement('input');
  inp.className = 'renbox';
  inp.value = old;
  inp.maxLength = 60;
  box.innerHTML = '';
  box.appendChild(inp);
  inp.focus();
  inp.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const v = inp.value.trim();
    if (!save || !v || v === old) { refreshRecords(); return; }
    try {
      const r = await post('api/records/' + rid + '/rename', { name: v });
      if (!r.ok) throw new Error(r.reason || T('改名失败', 'rename failed'));
      if (rid === S.curRid) { S.name = v; updateFileInfo(); }
      setStatus(T('已改名为「' + v + '」', 'Renamed to “' + v + '”'));
    } catch (err) {
      setStatus(T('改名失败：', 'Rename failed: ') + err.message);
    }
    refreshRecords();
  };
  inp.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') { ev.preventDefault(); finish(true); }
    else if (ev.key === 'Escape') { ev.preventDefault(); finish(false); }
  });
  inp.addEventListener('blur', () => finish(true));
}

function openDrawer() {
  $('#drawer').classList.add('on');
  $('#scrim').classList.add('on');
  refreshRecords();
}
function closeDrawer() {
  $('#drawer').classList.remove('on');
  $('#scrim').classList.remove('on');
}

async function saveRecord() {
  if (!S.id) { setStatus(T('先打开一张图', 'Open an image first')); return; }
  if (!S.rois.length) { setStatus(T('先框一个选区再存', 'Frame a region before saving')); return; }
  const btn = $('#btnSave');
  btn.disabled = true; btn.textContent = T('保存中…', 'Saving…');
  try {
    const B = S.blocks;
    const R = B ? blocksCalc() : null;
    const excl = B ? B.rois.map((r, ri) => [...B.excluded[ri]].sort((a, b) => a - b)) : null;
    const r = await post('api/records/save', {
      id: S.id, name: S.name || T('未命名', 'untitled'), res: S.res,
      rois: S.rois.map(x => ({ box: x.box, ang: x.ang || 0 })),
      samples: S.samples.map(s => ({ rgb: s.rgb, role: s.role })),
      tol: S.tol, purity: S.purity, line_px: S.line, mode: srvMode(),
      pct: (R && R.exN > 0) ? R.pct : null,
      astm: (R && R.exN > 0) ? R.astm : '',
      iso: (R && R.exN > 0) ? R.iso : '',
      excl: (R && R.exN > 0) ? excl : null,
      knee_warn: S.kneeWarn || '', knee_k: (S.knee && S.knee.k) || null,
      gm: S.gm ? {
        cols: S.gm.cols, rows: S.gm.rows,
        // 存界面上的输入值本身（不是回传的换算值）：重开时按同一个数填回去，单位才不会再错一次
        k_wire: S.line || 0, angle: S.gm.angle,   // 刀宽与「忽略细线宽度」已合并成一个数
        ox: (S.gmOff && S.gmOff.ox) || 0, oy: (S.gmOff && S.gmOff.oy) || 0,
        line_dx: S.gm.line_dx || [], line_dy: S.gm.line_dy || [],
        snap: !!S.gmSnap, total_pct: (S.gm.total || {}).pct,
        total_kept: (S.gm.total || {}).kept, total_area: (S.gm.total || {}).area,
        cells: (S.gm.cells || []).map(c => ({ i: c.i, j: c.j, pct: c.pct, kept: c.kept })),
        trace: S.gm.trace || null,        // 存下“这次认缝认到几条”，出问题时能回看
        pitch_x: S.gm.pitch_x, pitch_y: S.gm.pitch_y,
      } : null,
    });
    if (!r.ok) throw new Error(r.reason || T('保存失败', 'save failed'));
    const savedId = r.id || (r.record && r.record.id);
    if (savedId) { S.curRid = savedId; try { localStorage.setItem('baige.curRid', savedId); } catch (_) {} }
    btn.textContent = r.updated ? T('已更新 ✓', 'Updated ✓') : T('已保存 ✓', 'Saved ✓');
    setStatus(r.updated ? T('同一张图同一选区 → 已更新原记录（没有新增）', 'Same image and region — the existing record was updated (nothing added)')
                        : T('已保存到记录', 'Saved to records'));
    refreshRecCount();
    setTimeout(() => { btn.textContent = T('保存到记录', 'Save to records'); btn.disabled = false; }, 1600);
  } catch (e) {
    btn.textContent = T('保存到记录', 'Save to records'); btn.disabled = false;
    setStatus(T('保存失败：', 'Save failed: ') + e.message);
  }
}

async function openRecord(rid) {
  setStatus(T('打开记录…', 'Opening record…'));
  S.openAbort = new AbortController();
  const sig = S.openAbort.signal;
  let creep = null;
  const mb = n => (n / 1048576).toFixed(1) + ' MB';
  upShow(T('正在打开记录…', 'Opening record…'));
  let stopCreep = creepStart(2, 7, T('读取存档、解码原图…（约 1~2 秒）', 'Reading the archive and decoding the original… (~1–2 s)'), 350);
  try {
    const r = await post('api/records/' + rid + '/open', {}, 'POST', sig);
    if (!r.ok) throw new Error(r.reason || T('打开失败', 'open failed'));
    const meta = r.meta, rec = r.record;
    S.curRid = rid;
    try { localStorage.setItem('baige.curRid', rid); } catch (_) {}
    if ($('#drawer').classList.contains('on')) refreshRecords();
    clearImageState();                      // 先清上一张，再按记录恢复
    S.id = meta.id; S.name = rec.name || meta.name || T('记录', 'record');
    S.recAlgo = rec.algo || null;                       // 存档时的算法版本（老记录没有这个字段）
    S.recPct = (rec.result && typeof rec.result.pct === 'number') ? rec.result.pct : null;
    S.driftChecked = false;
    S.ow = meta.ow; S.oh = meta.oh; S.res = meta.default_res || 'full'; S.rot = 0;
    S.shrunkNote = meta.shrunk_note || ''; S.origW = meta.orig_w || meta.ow; S.origH = meta.orig_h || meta.oh;
    if (stopCreep) { stopCreep(); stopCreep = null; }
    upSet(8, T('取图…', 'Fetching image…'));
    await refetchImage((got, total) => {
      if (total) upSet(8 + 62 * (got / total), T(`载入图像 ${mb(got)} / ${mb(total)}　${Math.round(100 * got / total)}%`,
        `Loading image ${mb(got)} / ${mb(total)}  ${Math.round(100 * got / total)}%`));
      else upSet(40, T(`载入图像 ${mb(got)}…`, `Loading image ${mb(got)}…`));
    }, sig);
    S.rois = (rec.rois || []).map(x => ({ box: (x.box || [0, 0, 1, 1]).slice(), ang: x.ang || 0 }));
    S.samples = (rec.samples || []).map(s => ({ rgb: (s.rgb || [128, 128, 128]).slice(), role: s.role || 'coat' }));
    S.sel = -1; S.selSample = -1;
    S.tol = rec.tol != null ? rec.tol : 25;
    S.purity = rec.purity != null ? rec.purity : 0.5;
    S.line = rec.line_px != null ? rec.line_px : 1;
  markLineSrc(rec.line_px != null ? 'record' : 'default');
    S.widthTouched = true; S.suggRetry = false;   // 记录里的宽度就是当时的定论，别被建议值顶掉
    S.kneeWarn = rec.knee_warn || '';             // 存的时候判法没定出 k 的警告，重开还在
    applyExmode(rec.mode || 'width');        // 判法也要还原：不然重开时按当前下拉算，和存下的数字对不上
    try { renderKneeWarn(); } catch (_) {}
    S.roiNative = !!rec.roi_native;          // 记录里存了分析口径，重开必须一致（这里只同步状态，别再触发一次计算）
    $('#chkRoiNative').checked = S.roiNative;
    S.autoTol = false; S.detail = false;
    S.overlayUrl = null; S.overlayImg = null; S.contourUrl = null; S.contourImg = null;
    S.rulerHist = []; S.rulerFull = []; S.rulerLast = null; S.rulerSeg = null; S.rulerDone = null;
    S.annot.strokes = []; S.annot.cur = null; S.annot.res = null;
    syncParams(); syncSampleUI(); refreshRoiList();
    // 记录里存过模型格网 → 把参数与线位先摆回去（真正铺格在 analyze 之后）
    S.gmRec = (rec.gm && rec.gm.cols >= 2 && rec.gm.rows >= 2) ? rec.gm : null;
    if (S.gmRec) {
      if ($('#gmCols')) $('#gmCols').value = S.gmRec.cols;
      if ($('#gmRows')) $('#gmRows').value = S.gmRec.rows;
      if ($('#gmAng')) $('#gmAng').value = S.gmRec.angle || 0;
      S.gmOff = { ox: S.gmRec.ox || 0, oy: S.gmRec.oy || 0,
                  dx: (S.gmRec.line_dx || []).slice(), dy: (S.gmRec.line_dy || []).slice() };
      S.gmSnap = !!S.gmRec.snap;
      // 记录里存的列/行/歪角是当时的人工值：标成手改，别让自动认格数/自动定角把它覆盖掉
      S.gmColsManual = true; S.gmRowsManual = true; S.gmAngManual = true;
      updateGmLines();
    }
    $('#fileName').textContent = S.name;
    $('#drop').classList.add('hide');
    fitView();
    closeDrawer();
    if (S.samples.length >= 2 && S.rois.length) {
      if (stopCreep) { stopCreep(); stopCreep = null; }
      creep = creepStart(72, 93, T('重算中…（约 3 秒）', 'Recomputing… (~3 s)'), 400);
      await analyze();
      if (creep) { creep(); creep = null; }
      // 存档里记的是当时的数；重算如果对不上，必须说出来（口径/版本变过）
      const savedPct = rec.result && rec.result.pct;
      if (savedPct != null && S.total && Math.abs(S.total.pct - savedPct) > 0.05) {
        setStatus(T(`⚠ 重算结果 ${fmt(S.total.pct, 2)}% 与存档 ${fmt(savedPct, 2)}% 不一致（判法/阈值或程序版本变过）`,
          `⚠ Recomputed ${fmt(S.total.pct, 2)}% differs from the archived ${fmt(savedPct, 2)}% (method, threshold or version changed)`));
      }
      upSet(100, T('完成', 'Done'));
      const ex = (rec.result && rec.result.excl) || [];
      const nEx = ex.reduce((a, b) => a + (b ? b.length : 0), 0);
      if (nEx) {
        await listBlocks();
        if (S.blocks) {
          ex.forEach((arr, ri) => {
            if (S.blocks.excluded[ri]) (arr || []).forEach(i => S.blocks.excluded[ri].add(i));
          });
          draw(); renderBlocks();
          setStatus(T(`已按记录复原：手动排除 ${nEx} 块（自动 ${fmt(rec.result.auto_pct, 2)}% → ${fmt(rec.result.pct, 2)}%）`,
            `Restored from the record: ${nEx} blocks excluded by hand (auto ${fmt(rec.result.auto_pct, 2)}% → ${fmt(rec.result.pct, 2)}%)`));
        }
      }
      if (S.gmRec) {
        // 记录里带格网 → 用同一组参数把它铺回去（格数/刀宽/歪角/线位都一致），
        // 否则“打开记录”只回到主分析，格网那层就丢了（用户报的“格子没记下来”）
        try { syncGmWrap(); } catch (_) {}
        await runModelGrid({ snap: false });
        const gp0 = (S.gmRec.total || {}).pct, gp1 = (S.gm && S.gm.total || {}).pct;
        const gmOff = (gp0 != null && gp1 != null && Math.abs(gp0 - gp1) > 0.05);
        setStatus(gmOff
          ? T(`⚠ 格网复原后框选区 ${fmt(gp1, 2)}% 与存档 ${fmt(gp0, 2)}% 不一致（程序版本变过）`,
              `⚠ Grid restored but measures ${fmt(gp1, 2)}% vs archived ${fmt(gp0, 2)}% (version changed)`)
          : T(`记录里带模型格网 ${S.gmRec.cols}×${S.gmRec.rows} · 框选区 ${fmt(gp1, 2)}% → 已复原`,
              `Record had a model grid ${S.gmRec.cols}×${S.gmRec.rows} (${fmt(gp1, 2)}%) — restored`));
      }
    }
    else setResult(T(`已打开记录：${S.name}\n${rec.created || ''}\n参数已恢复（吸色 ${S.samples.length} 个、阈值 ${S.line}、容差 ${S.tol}），点「计算」出结果。`,
      `Record opened: ${S.name}\n${rec.created || ''}\nParameters restored (${S.samples.length} colour samples, threshold ${S.line}, tolerance ${S.tol}). Press “Calculate” for the result.`));
  } catch (e) {
    if (e && e.name === 'AbortError') setStatus(T('已取消', 'Cancelled'));
    else setStatus(T('打开失败：', 'Open failed: ') + e.message);
  } finally {
    if (creep) { creep(); creep = null; }
    if (stopCreep) { stopCreep(); stopCreep = null; }
    S.openAbort = null;
    upHide();
  }
}

async function deleteRecord(rid) {
  if (!confirm(T('从服务器上删除这条记录？（图和参数都会删，无法恢复）', 'Delete this record from the server? The image and parameters go too; this cannot be undone.'))) return;
  try {
    const r = await post('api/records/' + rid, {}, 'DELETE');
    if (!r.ok) throw new Error(r.reason || T('删除失败', 'delete failed'));
    $('#recCount').textContent = r.total;
    refreshRecords();
  } catch (e) {
    setStatus(T('删除失败：', 'Delete failed: ') + e.message);
  }
}

/* 手机上把参数面板收起来，让画布占满 */
function setSideCollapsed(v) {
  $('#side').classList.toggle('collapsed', !!v);
  $('#sideChev').textContent = v ? '▾' : '▴';
  try { localStorage.setItem('baige.side', v ? '1' : '0'); } catch (e) {}
  requestAnimationFrame(() => { resizeCanvas(); });
}
$('#sideHandle').onclick = () => setSideCollapsed(!$('#side').classList.contains('collapsed'));
if (window.matchMedia('(max-width: 860px)').matches) {
  let saved = null;
  try { saved = localStorage.getItem('baige.side'); } catch (e) {}
  setSideCollapsed(saved === null ? true : saved === '1');   // 手机上默认收起，画布占满
} else {
  setSideCollapsed(false);
}

async function refreshRecCount() {
  try {
    const r = await post('api/records/count', undefined, 'GET');
    if (r && r.ok) $('#recCount').textContent = r.total;
  } catch (e) {}
}
$('#btnRec3').onclick = applyRecommended;
if ($('#btnGrid')) $('#btnGrid').onclick = runGrid;
if ($('#btnGridOff')) $('#btnGridOff').onclick = () => { S.grid = null; $('#gridInfo').textContent = ''; draw(); };
/* 模型格网（旧图）：列/行/歪角/刀缝宽都由人给 */
const gmRerun = debounce(() => { if (S.gm) runModelGrid(); }, 420);
if ($('#btnGmRun')) $('#btnGmRun').onclick = () => runModelGrid();
if ($('#btnGmAngle')) $('#btnGmAngle').onclick = gmAutoAngle;
/* 模型格网是实验功能：默认收起来，勾一下才显示（记忆在 localStorage） */
function applyGmShow(on) {
  const w = $('#gmWrap');
  if (w) w.hidden = !on;
  if (!on) {                                  // 收起时两块都要清：自动找格子 + 模型格网
    S.gm = null; S.gmDrag = null;
    S.grid = null; S.gridCells = false;
    if ($('#gmInfo')) $('#gmInfo').textContent = '';
    if ($('#gridInfo')) $('#gridInfo').textContent = '';
    if ($('#chkGridCells')) $('#chkGridCells').checked = false;
  }
  if (typeof draw === 'function') draw();
}
/* 格网块现在挂在「量尺 → 刀缝判法」第三项后面：判法=grid 才出现，没有勾选框了 */
function syncGmWrap() { applyGmShow(S.exmode === 'grid'); }
try { syncGmWrap(); } catch (_) {}
if ($('#btnGmSnap')) $('#btnGmSnap').onclick = () => runModelGrid({ snap: true });
if ($('#btnSep')) $('#btnSep').onclick = () => runSep();
if ($('#btnGmReset')) $('#btnGmReset').onclick = () => {
  S.gmOff = { ox: 0, oy: 0, dx: [], dy: [] };
  if (S.gm) runModelGrid(); else draw();
};
/* 刀宽：滑块和数字框双向同步（手拖不好对小数时直接填） */
function syncGmK(v, rerun) {
  const k = Math.min(80, Math.max(2, Math.round(+v || 17)));
  if ($('#gmK')) $('#gmK').value = k;
  if ($('#gmKn')) $('#gmKn').value = k;
  if ($('#gmKv')) $('#gmKv').textContent = k;
  if (rerun) gmRerun();
}
if ($('#gmK')) $('#gmK').oninput = (e) => syncGmK(e.target.value, true);
if ($('#gmKn')) $('#gmKn').onchange = (e) => syncGmK(e.target.value, true);
const GM_MANUAL = { '#gmCols': 'gmColsManual', '#gmRows': 'gmRowsManual', '#gmAng': 'gmAngManual' };
['#gmCols', '#gmRows', '#gmAng'].forEach(s => {
  if ($(s)) $(s).onchange = () => {
    S[GM_MANUAL[s]] = true;                    // 人改过 → 自动认格数/自动定角不再覆盖（保留手改）
    updateGmLines();
    if (S.gm) gmRerun(); else scheduleGm(300);
  };
});
updateGmLines();
if ($('#chkGmArea')) $('#chkGmArea').onchange = (e) => { S.gmArea = e.target.checked; draw(); };
if ($('#chkGridCells')) $('#chkGridCells').onchange = (e) => { S.gridCells = e.target.checked; draw(); };
async function applyRecommended() {
  if (!S.id) { setStatus(T('先打开图片', 'Open an image first')); return; }
  if (S.samples.length < 2) { setStatus(T('先吸两个颜色：漆膜 + 基材', 'Sample two colours first: coating + substrate')); return; }
  S.autoTol = true; $('#chkAutoTol').checked = true;
  S.purity = 0.75;
  syncParams();
  setStatus(T('按推荐参数：自动容差 / 纯度 0.75 / 刀缝取全宽+1 …', 'Recommended parameters: auto tolerance / purity 0.75 / cut = full width +1 …'));
  await refreshSpan();
  await analyze();
  let k = S.line;
  if (S.rulerFull.length) {
    k = Math.round(S.rulerFull.reduce((a, b) => a + b, 0) / S.rulerFull.length) + 1;
  } else if (S.suggest) {
    k = Math.round(S.suggest);
  }
  const before = S.line;
  S.line = Math.max(1, Math.round(k));
  syncParams();
  if (S.line !== before) await analyze();
  setStatus(T(`推荐参数已生效：容差 ${S.tol}（自动）· 纯度 ${S.purity} · 刀缝阈值 ${S.line}`,
    `Recommended parameters applied: tolerance ${S.tol} (auto) · purity ${S.purity} · cut threshold ${S.line}`));
}

$('#rZoom').onclick = zoomOne;
$('#rFit').onclick = fitView;
refreshRecCount();
$('#btnRec').onclick = openDrawer;
$('#recClose').onclick = closeDrawer;
$('#scrim').onclick = closeDrawer;
$('#btnSave').onclick = saveRecord;
$('#recList').addEventListener('click', e => {
  const ren = e.target.closest('[data-ren]');
  if (ren) { e.stopPropagation(); startRename(ren.getAttribute('data-ren')); return; }
  const del = e.target.closest('[data-del]');
  if (del) { e.stopPropagation(); deleteRecord(del.getAttribute('data-del')); return; }
  const it = e.target.closest('.recitem');
  if (it) openRecord(it.getAttribute('data-id'));
});


$$('.tabs button').forEach(b => {
  b.onclick = () => {
    $$('.tabs button').forEach(x => x.classList.toggle('on', x === b));
    const t = b.dataset.tab;
    $$('.pane').forEach(p => p.classList.toggle('on', p.id === 'pane-' + t));
    S.mode = t === 'roi' ? 'rect' : t === 'pick' ? 'pick' : t === 'ruler' ? 'ruler' : 'blocks';
    setStatus(T({
      rect: '拖动框选；右键删框', pick: '点击取色（第一次漆膜，第二次基材）',
      ruler: '横跨刀缝拖一下', annot: '点「列出露底块」，然后点块排除误判',
    }[t], {
      rect: 'Drag to frame a region; right-click to delete it',
      pick: 'Click to sample a colour (first coating, second substrate)',
      ruler: 'Drag across a cut',
      annot: 'Click a block to exclude a false positive',
    }[t]));
    if (t === 'annot') {
      draw(); renderBlocks();
      if (!S.blocks && S.blocksOn && S.id && S.rois.length && S.samples.length >= 2) listBlocks(true);
    }
  };
});

/* 容差 */
function setTol(v) {
  S.tol = clamp(Math.round(+v), 3, 120);        // 只认整数，和桌面版一致
  const wasAuto = S.autoTol;
  S.autoTol = false;                             // 手动一动就退出自动（同桌面版）
  $('#chkAutoTol').checked = false;
  $('#rngTol').value = S.tol; $('#numTol').value = S.tol;
  if (wasAuto) setStatus(T(`已切到手动容差 ${S.tol}（想恢复自动就勾回「自动容差」）`,
      `Switched to manual tolerance ${S.tol} (tick “Auto tolerance” to go back)`));
  updateSpan();
  if (S.exmode === 'grid') scheduleGm(); else debouncedAnalyze();   // 格网判法下只重算格网
}
$('#rngTol').oninput = e => setTol(e.target.value, true);
$('#numTol').onchange = e => setTol(e.target.value, false);
$('#chkAutoTol').onchange = async e => {
  S.autoTol = e.target.checked;
  if (S.autoTol) {
    await refreshSpan();
    if (S.span != null) {
      setStatus(T(`自动容差 = ${S.tol}（样本间距 ${Math.round(S.span)} × 45%）`,
        `Auto tolerance = ${S.tol} (colour distance ${Math.round(S.span)} × 45%)`));
      analyze();
    }
  } else setStatus(T('已取消自动容差，容差不再跟着样本间距走', 'Auto tolerance off — the tolerance no longer follows the colour distance'));
};

/* 纯度 */
function updatePurityInfo() {
  const el = $('#purityInfo');
  const p = S.purity;
  if (p <= 0.505) {
    el.textContent = T('0.50 = 不剔除过渡带：所有像素只按最近颜色归类，刀缝全靠「刀缝阈值」排掉',
      '0.50 = keep the blend band: every pixel is assigned to its nearest colour and cuts are removed only by the cut threshold');
  } else {
    el.textContent = T(`剔除混色比 t∈(${(1 - p).toFixed(2)}, ${p.toFixed(2)}) 的过渡像素（既不算漆膜也不算露底）`,
      `Blend pixels with mixing ratio t in (${(1 - p).toFixed(2)}, ${p.toFixed(2)}) are excluded (neither coating nor bare)`);
  }
}

function setPurity(v) {
  S.purity = Math.round(clamp(+v, 0.5, 0.99) * 100) / 100;
  $('#rngPurity').value = S.purity; $('#numPurity').value = S.purity.toFixed(2);
  updatePurityInfo();
  if (S.exmode === 'grid') scheduleGm(); else debouncedAnalyze();
}
$('#rngPurity').oninput = e => setPurity(e.target.value);
$('#numPurity').onchange = e => setPurity(e.target.value);

/* 刀缝阈值 */
function setLine(v, keepSuggest) {
  S.line = clamp(Math.round(+v), 1, 60);
  if (!keepSuggest) markLineSrc('manual');
  if (!keepSuggest && (S.exmode === 'auto' || S.exmode === 'auto_blk')) {   // 自动判法下用户自己改线宽：以他的为准
    S.lineManual = true;
    if (S.knee && S.knee.k) setStatus(T('已按你设的刀缝 ' + S.line + ' 计算（自动值 ' + S.knee.k + '，重选「自动定刀缝」可恢复）',
      `Using your cut width ${S.line} (auto was ${S.knee.k}; re-select “automatic” to restore it)`));
  }
  $('#rngLine').value = Math.min(25, S.line); $('#numLine').value = S.line;
  try { scheduleGm(400); } catch (_) {}       // 改阈值＝改刀宽，格网立刻重算，不用再点按钮
  try { localStorage.setItem('baige.line', String(S.line)); } catch (_) {}
  if (!keepSuggest) { S.suggestHold = null; S.widthTouched = true; }   // 用户自己改的：本图内不再用建议值覆盖
  // 判法=格网时刀宽就是格网的刀带宽：只重算格网，别再跑一遍像素分析（会顺带扫 k、跑“按块数回落”）
  if (S.exmode !== 'grid') debouncedAnalyze();
}
$('#rngLine').oninput = e => setLine(e.target.value);
$('#numLine').onchange = e => setLine(e.target.value);

/* 样本角色 */
$('#btnClearRoi').onclick = clearAllRois;
$('#sCoat').onclick = () => setSampleRole('coat');
$('#sSub').onclick = () => setSampleRole('sub');
/* 删除靠样本行右侧的 × 或 Delete 键，不再单独放一个「删除」按钮 */

/* 量尺：设完必须在这里给出反馈（数值框在「吸色」页，用户看不见） */
function applyLineFromRuler(px, wFull, wHalf) {
  const v = Math.round(px) + 1;
  setLine(v);
  const chain = (wHalf && wHalf < px - 0.6)
    ? T(`量到的半高宽 ${fmt(wHalf, 1)} px 只是缝心，加上两侧发虚的边缘，基线全宽 ${fmt(wFull, 1)} px`,
        `the half-max width ${fmt(wHalf, 1)} px is only the centre; with the soft edges the full width is ${fmt(wFull, 1)} px`)
    : T(`基线全宽 ${fmt(wFull, 1)} px`, `full width ${fmt(wFull, 1)} px`);
  let msg = T(`已把「刀缝阈值」设为 ${S.line} px＝${chain}，再 +1 取整。\n` +
            `（k 的单位是分析图像素，值越大删得越狠；结果已重算，在「结果」栏看。）`,
            `Cut threshold set to ${S.line} px = ${chain}, +1 rounded.\n` +
            `(k is in analysis-image pixels; larger removes more. The result was recomputed — see the result box.)`);
  if (v > 14) {
    msg = T(`⚠️ ${chain} 比常见刀缝（2~10 px）宽得多，多半量到了格子内部或成片脱落。\n` +
          `已按 ${S.line} px 设置（可能把真实脱落一起删掉），建议换一条细缝重测。`,
          `⚠️ ${chain} is much wider than a normal cut (2–10 px) — you probably measured inside a square or across a flake.\n` +
          `Set to ${S.line} px anyway (it may erase real removal); measure a thin cut instead.`);
  }
  $('#rulerMsg').textContent = msg;
}
$('#rUse').onclick = () => {
  if (!S.rulerLast) {
    $('#rulerMsg').textContent = T('还没量：先在图上垂直于刀缝横着拖一条（长度 30 px 以上）',
      'Nothing measured yet: drag a line perpendicular to a cut (30 px or longer)');
    return;
  }
  const full = S.rulerLast.full_analysis || S.rulerLast.width_analysis;
  markLineSrc('ruler');
  applyLineFromRuler(full, full, S.rulerLast.width_analysis);
};
$('#rMean').onclick = () => {
  if (!S.rulerHist.length && !S.rulerFull.length) {
    $('#rulerMsg').textContent = T('还没有可靠的测量记录：先量 1~2 条刀缝', 'No reliable measurements yet: measure one or two cuts first');
    return;
  }
  const arr = S.rulerFull.length ? S.rulerFull : S.rulerHist.map(v => v * 1.8);
  const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
  markLineSrc('ruler');
  applyLineFromRuler(avg, avg);
};
$('#rClear').onclick = () => {
  S.rulerHist = []; S.rulerLast = null; S.rulerDone = null; S.rulerSeg = null;
  $('#rulerMsg').textContent = T('切到这一页后，在图上垂直于刀缝横着拖一小段（30px 以上）。',
    'On this tab, drag a short segment across a cut, perpendicular to it (30 px or more).');
  draw();
};

/* 分辨率 */
$('#selRes').onchange = e => changeRes(e.target.value);

/* 选区旋转 */
function setRoiAngle(v) {
  if (S.sel < 0 || !S.rois[S.sel]) return;
  S.rois[S.sel].ang = Math.round(+v * 10) / 10;
  syncRoiAngle(); refreshRoiList(); draw();
  if (S.exmode === 'grid') scheduleGm(); else debouncedAnalyze();
}
/* 拖拽上传 */
['dragenter', 'dragover'].forEach(ev => wrap.addEventListener(ev, e => {
  e.preventDefault(); $('#drop').classList.add('hover');
}));
['dragleave', 'drop'].forEach(ev => wrap.addEventListener(ev, e => {
  e.preventDefault();
  if (ev === 'drop' || e.target === wrap) $('#drop').classList.remove('hover');
}));
wrap.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) upload(f);
});

/* 启动 */
S.softOn = false; try { localStorage.removeItem('baige.soft'); } catch (_) {}   // 软权重已撤下，旧开关作废
try { S.blocksOn = localStorage.getItem('baige.showBlocks') !== '0'; } catch (_) { S.blocksOn = true; }
$('#chkBlocks').checked = S.blocksOn;
if ($('#chkSoftR')) $('#chkSoftR').checked = false;
S.roiNative = prefGet('roiNative', true);        // 默认开「选区原生」，但记住用户改过的
$('#chkRoiNative').checked = S.roiNative;
S.semi = prefGet('semi', true);                  // 半自动：记住"取消勾"
$('#chkSemi').checked = S.semi;
S.detail = prefGet('detail', false);
$('#chkDetail').checked = S.detail;
S.showOverlay = prefGet('overlay', true);
$('#chkOverlay').checked = S.showOverlay;
try {
  const keep = localStorage.getItem('baige.exmode');
  // 旧的 auto 被 auto_blk 取代；shape/local 已撤下，统一回落到新的第一顺位
  setExmode((keep === 'width' || keep === 'auto_blk') ? keep : 'auto_blk');
} catch (_) { setExmode('auto_blk'); }
try { updateKneeHint(); } catch (_) {}
resizeCanvas();
syncParams();
setStatus(T('滚轮缩放 · 中键拖动平移 · 双击适应窗口', 'Wheel to zoom · middle-drag to pan · double-click to fit'));
