/* Copyright (C) 2026 iamsad3508
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Part of Baige (百格附着力分析). Commercial licence: see COMMERCIAL.md */
/* 百格分析 · 界面多语言（zh / en）
 *
 * 用法：
 *   HTML：<button data-i18n="btn.open">打开图片</button>   ← 英文时按 key 换成译文
 *         <label data-i18n-title="chk.semi.tip" title="…">…</label>
 *   JS  ：t('btn.calc')            → 当前语言的文案（缺译文时回退中文）
 *         tf('res.row', 3, 12.5)   → 带 %s / %d 占位（用 {0} {1} 也行，见 tf）
 *
 * 语言优先级：?lang=zh|en > localStorage['baige.lang'] > navigator.language（zh 开头=中文）> 英文
 * 切换语言：左上角下拉，写 localStorage 后整页刷新（比逐处重渲染稳）。
 *
 * 注意：服务端拼出来的诊断长文本、CSV 表头、PNG 标注条仍是中文（第二阶段）。
 */
(function () {
  'use strict';

  var ZH = {
    'brand.name': '百格分析',
    'devcredit': '用 DeepSeek API 开发',
    'devcredit.tip': '本工具的代码与判定逻辑由 DeepSeek API 辅助开发',
    'btn.open': '打开图片',
    'btn.calc': '▶ 计算',
    'btn.report': '导出报告',
    'btn.png': '导出标注图',
    'btn.rec': '记录',
    'chk.semi': '半自动',
    'chk.semi.tip': '框完就自动吸色+定阈值+出结果（不可盲信，只当起点）',
    'chk.roinative': '选区原生',
    'chk.roinative.tip': '只把选区那一块按原图像素分析（不降到 6MP）：小选区更细、更快、更省内存；刀缝阈值仍按画布口径',
    'chk.detail': '诊断明细',
    'chk.detail.tip': '勾上才显示判定余量、露底厚度分布、色差/容差敏感性、每选区像素构成等明细；关着只给总判级 + 每个选区的脱落率与判级 + 合计',
    'pill.knee': '⚠ 判法没定出 k',
    'rec.title': '记录',
    'rec.close': '关闭',
    'rec.loading': '加载中…',
    'rec.foot': '记录只有你点 × 才会从服务器删除。存档图是原始上传文件（不重新编码），重开时数字逐位一致。',
    'file.none': '未打开图片',
    'chk.overlay': '覆盖层',
    'chk.overlay.tip': '把分类结果显示在图上：红=露底（刀缝+脱落，原样显示），绿=完好漆膜；想区分刀缝和脱落，看「刀缝判法」和结果文字',
    'btn.fit': '适应窗口',
    'btn.fit.tip': 'Ctrl+0 / 双击画布',
    'btn.one': '1:1',
    'btn.one.tip': 'Ctrl+1：1 图像像素 = 1 屏幕像素',
    'up.cancel': '取消上传',
    'up.title': '上传中…',
    'drop.text': '拖一张测试照片到这里，或点左上「打开图片」',
    'drop.hint': '建议垂直拍摄、光照均匀；刀缝越清晰，量化越准',
    'hint.needImage': '先打开一张测试照片',
    'side.title': '参数与结果',
    'tab.roi': '框选',
    'tab.pick': '吸色',
    'tab.ruler': '量尺',
    'tab.annot': '块确认',
    'roi.h': '选区',
    'roi.empty': '还没有选区：在图上拖出划格区',
    'roi.clear': '清空全部',
    'roi.hint': '拖动框内平移 · 拖四角/四边改大小 · 右键删除',
    'roi.keys.h': '快捷键',
    'roi.keys.tips': '方向键微移 1px，Shift+方向键 10px；Delete 删除选中框；Ctrl+0 适应窗口；Ctrl+1 切 1:1；滚轮以鼠标为中心缩放；按住中键拖动平移。',
    'pick.h': '颜色样本（点颜色可选中）',
    'pick.empty': '还没有样本：图上点一下完好漆膜，再点一下露底基材',
    'pick.coat': '设为漆膜',
    'pick.sub': '设为基材',
    'tol.h': '颜色容差',
    'tol.auto': '自动容差',
    'tol.auto.factor': '按样本间距 × 45%',
    'tol.span': '样本间距：—（先吸漆膜和基材）',
    'pur.h': '露底纯度',
    'pur.info': '0.50 = 不剔除过渡带',
    'pur.tips': '这是手动判据，不是算出来的：每个像素先算「混色比 t」——离漆膜色的 Lab 距离 ÷ (离漆膜 + 离基材)，t≈0 纯漆膜、t≈0.5 一半一半（刀缝/掉漆边缘）、t≈1 纯露底。落在 (1−纯度, 纯度) 之间的像素当过渡带整个剔除，两边都不算。所以 0.50 = 完全不剔除，0.75 = 把 t 在 0.25~0.75 的混色像素都扔掉。',
    'tips.h': '参数怎么调（按顺序，别乱试）',
    'tips.body': '① 容差：先留「自动」。看结果区的未分类%——>10% 说明容差太小或颜色没吸全（亮处/暗处各吸一个漆膜）；<3% 且判定面积明显偏大，再往小调。② 露底纯度：照片有反光、刀缝发白发灰、边缘一片糊的时候才动它，常用 0.75；纯色平面图（标准示意图、扫描件）用 0.75 最稳，真实照片噪点多时 0.6~0.8。③ 刀缝阈值：最靠得住的办法是量一条刀缝 → 用测量值；没有量尺就看「露底厚度」分布——如果 1-3px 占一半以上，说明刀缝没删干净，往外调 2~3。验证：容差 ±25% 结果不跨档、换一组吸色点结果不变，这个数才算能用。',
    'guide.h': '怎么用（3 步）',
    'guide.close.tip': '不再显示这条说明',
    'ruler.h': '刀缝量尺',
    'g.ruler.1': '在图上垂直于刀缝横着拖一小段（≥30 屏幕像素），松手后会显示这条缝的半高宽和全宽。',
    'g.ruler.2': '点下面「用测量值」，阈值 k 会按全宽 + 1 自动填好；也可以直接点 k 曲线表里的某一档手改。',
    'g.ruler.3': '不想量也行：判法选「按块数回落」会自动定 k，量尺只是最快、最靠谱的一条路。',
    'g.ruler.tip': '手机上先把图放大（双指）再量，否则 6 px 的缝在屏幕上只有 1 px，量出来偏小。',
    'ruler.msg': '切到这一页后，在图上垂直于刀缝横着拖一小段（30px 以上）。',
    'ruler.use': '用测量值',
    'ruler.mean': '用平均值',
    'ruler.clear': '清空',
    'line.h': '忽略细线宽度（刀缝阈值）',
    'line.mode': '刀缝判法',
    'mode.auto': '按块数回落',
    'mode.width': '按露底分布',
    'mode.tip': '按块数回落：自动扫 k，按块数峰谷定刀缝（默认）。按露底分布：按露底厚度分布给一个阈值，你可以随时手改，比阈值粗的算脱落。',
    'annot.h': '点块确认（不用画）',
    'g.annot.1': '上面的开关勾着时，框完选区一算，程序就把检出的露底拆成块，按块号画在图上。',
    'g.annot.2': '点某一块（编号或它的轮廓）= 把这块当刀缝排除，块会变灰打叉；再点一下恢复。',
    'g.annot.3': '只排除其实是刀缝或脏点的块，别排除真实的掉漆。排除后数字立刻更新，保存记录会一起存。',
    'g.annot.tip': '只能减不能加：自动没检出的块点不上去，那种情况要改 k 或换吸色点重算。',
    'annot.chk': '图上显示块号',
    'annot.chk.tip': '勾上：算完就自动列块并把块号、轮廓画在图上（调参数会实时更新）；取消：不显示、也不去算，省一次运算',
    'annot.reset': '全部恢复',
    'annot.tips': '点某块 = 标成刀缝排除（变灰打叉），再点一下恢复；不用描边、不用选笔宽。',
    'annot.ants': '蚂蚁线只是画出来的：只描块的外圈、并按块大小做了简化，所以它围住的面积和块的像素数不完全相等（一般差 1~3%，带孔的大块偏大、细碎小块偏小）。判级、百分比、排除都按像素数算，与轮廓无关，不影响结果。',
    'block.h': '块列表',
    'block.empty': '还没列出：勾着上面的开关、框好选区并计算后会自动列块。',
    'block.info.h': '确认口径',
    'res.h': '结果',
    'res.save': '保存到记录',
    'res.save.tip': '把这张图和当前参数存进「记录」，服务器不会自动删除',
    'res.copy': '复制',
    'res.copy.tip': '复制成表格（制表符分隔），可直接粘进 Excel 自动分列',
    'res.placeholder': '打开图片 → 吸两个颜色 → 框出划格区 → 计算',
    'res.total': '合计',
    'res.worst': '最差',
    'res.dropPct': '脱落',
    'res.refOnly': '仅参考',
    'res.noRoi': '还没有选区：先在图上框出划格区，再计算。',
    'res.needColors': '请先吸两个颜色（完好漆膜 + 露底基材）',
    'res.calculating': '计算中…',
    'res.uploading': '上传中…',
    'res.uploaded': '已上传，正在解析…',
    'res.restore': '已恢复',
    'res.copied': '已复制',
    'file.info': '存档 {0}×{1} → 分析 {2}×{3}（缩小 {4}×）',
    'file.roinative': '选区原生 {0}×{1}（{2}× 于画布）',
    'file.rotated': '旋转 {0}°',
    'status.autoBlocks': '已自动更新：{0} 块（点块可排除）',
    'src.link': '源码 · AGPL-3.0',
    'lang.label': '语言'
  };

  var EN = {
    'brand.name': 'Baige',
    'devcredit': 'Developed with DeepSeek API',
    'devcredit.tip': 'Code and grading logic of this tool were developed with the DeepSeek API',
    'btn.open': 'Open image',
    'btn.calc': '▶ Calculate',
    'btn.report': 'Report',
    'btn.png': 'Export PNG',
    'btn.rec': 'Records',
    'chk.semi': 'Semi-auto',
    'chk.semi.tip': 'After framing: auto-pick colours, pick the threshold and calculate (a starting point, not authoritative)',
    'chk.roinative': 'Native ROI',
    'chk.roinative.tip': 'Analyse the ROI at native pixels instead of downscaling to 6 MP: finer, faster, less memory; k stays in canvas units',
    'chk.detail': 'Diagnostics',
    'chk.detail.tip': 'Show grading margin, exposure-thickness histogram, tolerance sensitivity, per-ROI pixel composition; off = grade + per-ROI percentages + total only',
    'pill.knee': '⚠ threshold not determined',
    'rec.title': 'Records',
    'rec.close': 'Close',
    'rec.loading': 'Loading…',
    'rec.foot': 'Records are deleted only when you click ×. The stored image is the original upload (never re-encoded), so reopening gives identical numbers.',
    'file.none': 'No image open',
    'chk.overlay': 'Overlay',
    'chk.overlay.tip': 'Show the classification on the image: red = exposed substrate (cuts + loss, as-is), green = intact coating',
    'btn.fit': 'Fit',
    'btn.fit.tip': 'Ctrl+0 / double-click canvas',
    'btn.one': '1:1',
    'btn.one.tip': 'Ctrl+1: one image pixel = one screen pixel',
    'up.cancel': 'Cancel upload',
    'up.title': 'Uploading…',
    'drop.text': 'Drop a test photo here, or click “Open image”',
    'drop.hint': 'Shoot straight down with even lighting; sharper cuts give better numbers',
    'hint.needImage': 'Open a test photo first',
    'side.title': 'Parameters & result',
    'tab.roi': 'ROI',
    'tab.pick': 'Colours',
    'tab.ruler': 'Ruler',
    'tab.annot': 'Blocks',
    'roi.h': 'Regions',
    'roi.empty': 'No region yet: drag a box over the cross-hatch area',
    'roi.clear': 'Clear all',
    'roi.hint': 'Drag inside to move · drag corners/edges to resize · right-click to delete',
    'roi.keys.h': 'Shortcuts',
    'roi.keys.tips': 'Arrows nudge 1 px, Shift+arrows 10 px; Delete removes the selected box; Ctrl+0 fit; Ctrl+1 toggle 1:1; wheel zooms at the cursor; middle-drag pans.',
    'pick.h': 'Colour samples (click a swatch to select)',
    'pick.empty': 'No sample yet: click intact coating, then click exposed substrate',
    'pick.coat': 'Set as coating',
    'pick.sub': 'Set as substrate',
    'tol.h': 'Colour tolerance',
    'tol.auto': 'Auto tolerance',
    'tol.auto.factor': 'sample gap × 45%',
    'tol.span': 'Sample gap: — (pick coating and substrate first)',
    'pur.h': 'Substrate purity',
    'pur.info': '0.50 = no transition band removed',
    'pur.tips': 'This is a manual criterion, not a computed one: each pixel gets a mixing ratio t = distance to coating ÷ (distance to coating + distance to substrate). t≈0 pure coating, t≈0.5 half-and-half (cuts, flaking edges), t≈1 pure substrate. Pixels inside (1−purity, purity) are dropped as a transition band and counted on neither side. So 0.50 removes nothing, 0.75 drops everything with t between 0.25 and 0.75.',
    'tips.h': 'How to tune (in this order)',
    'tips.body': '① Tolerance: keep “auto”. Check unclassified% — above 10% means the tolerance is too small or a colour is missing (pick coating in both bright and dark areas); below 3% with an obviously inflated area, lower it. ② Purity: only touch it with glare, washed-out cuts or blurred edges; 0.75 is the usual value, and is safest for flat artwork/scans (0.6–0.8 for noisy photos). ③ Cut threshold: the most reliable way is to measure one cut → use the measured value; without a ruler look at the exposure-thickness histogram — if 1-3 px dominates, cuts are not fully removed, raise by 2–3. Verify: result must not cross a grade at tolerance ±25%, and must not change with a second set of samples.',
    'guide.h': 'How to use (3 steps)',
    'guide.close.tip': "Don't show this again",
    'ruler.h': 'Cut ruler',
    'g.ruler.1': 'Drag a short segment across a cut, perpendicular to it (≥30 screen px). On release you get the half-max width and the full width of that cut.',
    'g.ruler.2': 'Click “Use measurement” below: the threshold k is filled in as full width + 1. You can also click any row of the k table to set it by hand.',
    'g.ruler.3': 'Measuring is optional: with the “by block count” method k is found automatically. The ruler is just the fastest and most reliable route.',
    'g.ruler.tip': 'On a phone, zoom in (two fingers) before measuring — otherwise a 6 px cut is only 1 px on screen and reads too small.',
    'ruler.msg': 'On this tab, drag a short stroke (30 px or more) perpendicular to a cut.',
    'ruler.use': 'Use measured',
    'ruler.mean': 'Use average',
    'ruler.clear': 'Clear',
    'line.h': 'Cut threshold (structure width ignored)',
    'line.mode': 'Threshold rule',
    'mode.auto': 'By block count',
    'mode.width': 'By exposure distribution',
    'mode.tip': 'By block count: scans k automatically and finds the cut width at the block-count peak/fall (default). By exposure distribution: suggests a threshold from the exposed-thickness histogram; you can override it any time.',
    'annot.h': 'Block review (no drawing)',
    'g.annot.1': 'While the switch above is on, every analysis splits the detected bare areas into blocks and draws them with numbers on the image.',
    'g.annot.2': 'Click a block (its number or outline) = treat it as a cut and exclude it; it turns grey with a cross. Click again to restore.',
    'g.annot.3': 'Only exclude blocks that really are cuts or dirt, not real coating removal. Numbers update immediately and are stored with the record.',
    'g.annot.tip': 'Subtract only — a block the detector missed cannot be added; in that case change k or pick another colour sample and recalculate.',
    'annot.chk': 'Show block numbers',
    'annot.chk.tip': 'On: after each calculation blocks are listed and their numbers/outlines drawn (updates live); off: nothing is computed or drawn',
    'annot.reset': 'Restore all',
    'annot.tips': 'Click a block = exclude it as a cut (turns grey, crossed out); click again to restore. No drawing, no brush width.',
    'annot.ants': 'The marching ants are decorative: they trace only the outer ring and are simplified by block size, so the enclosed area does not exactly equal the pixel count (typically within 1–3%). Grades, percentages and exclusions all use pixels, not outlines.',
    'block.h': 'Block list',
    'block.empty': 'Nothing listed yet — with the switch above on, blocks appear automatically after you frame a region and calculate.',
    'block.info.h': 'Counting basis',
    'res.h': 'Result',
    'res.save': 'Save to records',
    'res.save.tip': 'Store this image and the current parameters; the server never deletes records automatically',
    'res.copy': 'Copy',
    'res.copy.tip': 'Copy as a tab-separated table, ready to paste into Excel',
    'res.placeholder': 'Open image → pick two colours → frame the cross-hatch area → calculate',
    'res.total': 'total',
    'res.worst': 'worst',
    'res.dropPct': 'removed',
    'res.refOnly': 'reference only',
    'res.noRoi': 'No region yet: frame the cross-hatch area first, then calculate.',
    'res.needColors': 'Pick two colours first (intact coating + exposed substrate)',
    'res.calculating': 'Calculating…',
    'res.uploading': 'Uploading…',
    'res.uploaded': 'Uploaded, decoding…',
    'res.restore': 'restored',
    'res.copied': 'copied',
    'file.info': 'stored {0}×{1} → analysed {2}×{3} (downscaled {4}×)',
    'file.roinative': 'native ROI {0}×{1} ({2}× of canvas)',
    'file.rotated': 'rotated {0}°',
    'status.autoBlocks': 'updated: {0} blocks (click to exclude)',
    'src.link': 'Source · AGPL-3.0',
    'lang.label': 'Language'
  };

  var DICT = { zh: ZH, en: EN };
  // 优先级：?lang=zh|en（自测/截图用）> localStorage > navigator.language
  var saved = null;
  try { saved = localStorage.getItem('baige.lang'); } catch (e) { saved = null; }
  var qs = '';
  try { qs = (new URLSearchParams(location.search).get('lang') || '').toLowerCase(); } catch (e) { qs = ''; }
  if (qs === 'zh' || qs === 'en') {
    saved = qs;
    try { localStorage.setItem('baige.lang', qs); } catch (e) {}
  }
  var nav = (navigator.language || navigator.userLanguage || 'zh').toLowerCase();
  var LANG = (saved === 'zh' || saved === 'en') ? saved : (nav.indexOf('zh') === 0 ? 'zh' : 'en');

  function t(key) {
    var table = DICT[LANG] || ZH;
    if (table[key] != null) return table[key];
    if (ZH[key] != null) return LANG === 'en' ? ZH[key] : ZH[key];   // 缺译文：退回中文
    return key;
  }

  var tOriginals = [];          // 记住被翻译过的节点原文，切回中文时用

  function applyI18n(root) {
    root = root || document;
    if (LANG === 'zh') return;
    root.querySelectorAll('[data-i18n]').forEach(function (el) {
      var k = el.getAttribute('data-i18n');
      if (!el.__zh) { el.__zh = el.textContent; tOriginals.push(el); }
      var v = DICT.en[k];
      if (v != null) el.textContent = v;
    });
    root.querySelectorAll('[data-i18n-tip]').forEach(function (el) {
      var k = el.getAttribute('data-i18n-tip');
      if (!el.__zhTip) { el.__zhTip = el.getAttribute('title') || ''; }
      var v = DICT.en[k];
      if (v != null) el.setAttribute('title', v);
    });
    root.querySelectorAll('[data-i18n-ph]').forEach(function (el) {
      var k = el.getAttribute('data-i18n-ph');
      var v = DICT.en[k];
      if (v != null) el.setAttribute('placeholder', v);
    });
  }

  function setLang(lang) {
    try { localStorage.setItem('baige.lang', lang); } catch (e) {}
    location.reload();
  }

  function tf(key) {
    var args = Array.prototype.slice.call(arguments, 1);
    return String(t(key)).replace(/\{(\d+)\}/g, function (m, i) {
      return args[+i] != null ? args[+i] : m;
    });
  }

  /* T(zh, en)：给 app.js 里那些一次性/带变量的文案用——不用为每句都在字典里加 key。
     en 缺省时退回中文，缺译文不会变空白。 */
  function T(zh, en) {
    return (LANG === 'en' && en != null) ? en : zh;
  }

  window.I18N = { lang: LANG, t: t, tf: tf, setLang: setLang, apply: applyI18n, dict: DICT };
  window.t = t;
  window.tf = tf;
  window.T = T;
  window.setLang = setLang;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { applyI18n(); });
  } else {
    applyI18n();
  }
})();
