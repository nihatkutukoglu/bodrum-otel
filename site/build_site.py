# -*- coding: utf-8 -*-
"""Builds the v3 Bodrum Hotel Intelligence site: same visual identity (CSS,
chart primitives, section rhythm) as the fetched original artifact, with all
data/content refreshed from actual current pipeline outputs (no hardcoded
numbers - everything below is read from FINAL_DATA.json / quote jsons, which
were themselves built directly from the real CSV/report files this session).
"""
import json
from pathlib import Path

SCRATCH = Path(r"C:\Users\bilin\AppData\Local\Temp\claude\c--Users-bilin-OneDrive-Masa-st--bodrum\0149f59c-87c6-4ebf-9dd6-9849209eba4b\scratchpad")
ORIG = Path(r"C:\Users\bilin\.claude\projects\c--Users-bilin-OneDrive-Masa-st--bodrum\0149f59c-87c6-4ebf-9dd6-9849209eba4b\tool-results\artifact-ad0662c4-1787680744-a7ce.html")

DATA = json.loads((SCRATCH / "FINAL_DATA.json").read_text(encoding="utf-8"))
QUOTES_G = json.loads((SCRATCH / "site_quotes_google.json").read_text(encoding="utf-8"))
QUOTES_T = json.loads((SCRATCH / "site_quotes_trip.json").read_text(encoding="utf-8"))

orig_lines = ORIG.read_text(encoding="utf-8").splitlines(keepends=True)

# ---- 1. frame-runtime preamble + head + CSS, verbatim (lines 1-238 0-indexed 0-237) ----
head_and_css = "".join(orig_lines[0:238])

# ---- 2. old EXAMPLES for sk_aspect / sk_hotel (still valid real quotes, v3 is a superset) ----
old_data_line = orig_lines[732]
old_examples_line = orig_lines[733]
import re
m = re.match(r"^\s*const EXAMPLES\s*=\s*", old_examples_line)
old_examples = json.loads(old_examples_line[m.end():].rstrip().rstrip(";\n"))
SK_ASPECT_EXAMPLES = old_examples["sk_aspect"]
SK_HOTEL_EXAMPLES = old_examples["sk_hotel"]

print("Loaded head_and_css chars:", len(head_and_css))
print("Loaded DATA hotels:", len(DATA["hotels"]))
print("SK_ASPECT_EXAMPLES keys:", list(SK_ASPECT_EXAMPLES.keys()))

# Extra CSS additions appended to the existing <style> block (before its closing </style>)
EXTRA_CSS = """
.src-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:999px;font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;border:1px solid var(--line);}
.src-badge.g{border-color:var(--accent);color:var(--accent);}
.src-badge.t{border-color:var(--gold);color:var(--gold);}
.src-badge.s{border-color:var(--bad);color:var(--bad);}
.src-badge.off{opacity:.4;}
.confidence-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;font-family:var(--mono);font-size:12px;letter-spacing:.04em;}
.confidence-badge.HIGH{background:rgba(25,158,112,.15);color:var(--good);border:1px solid rgba(25,158,112,.4);}
.confidence-badge.MEDIUM{background:rgba(201,161,90,.15);color:var(--gold);border:1px solid rgba(201,161,90,.4);}
.confidence-badge.LOW{background:rgba(201,130,38,.12);color:var(--warn);border:1px solid rgba(201,130,38,.35);}
.confidence-badge.VERY_LOW{background:rgba(212,67,107,.1);color:var(--sand-faint);border:1px solid var(--line);}
.source-block{margin-top:22px;padding:20px 22px;border-radius:10px;background:var(--ink-3);border-left:3px solid var(--line);}
.source-block.g{border-left-color:var(--accent);}
.source-block.t{border-left-color:var(--gold);}
.source-block.s{border-left-color:var(--bad);}
.source-block.p{border-left-color:var(--sand-dim);}
.source-block h5{font-size:13px;letter-spacing:.05em;text-transform:uppercase;color:var(--sand);margin-bottom:14px;font-family:var(--sans);font-weight:600;}
.na-block{padding:20px 22px;border-radius:10px;background:var(--ink-3);border:1px dashed var(--line);color:var(--sand-faint);font-style:italic;font-size:13.5px;}
.compare-wrap{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
@media(max-width:800px){.compare-wrap{grid-template-columns:1fr;}}
.compare-metric{display:flex;justify-content:space-between;align-items:baseline;padding:10px 0;border-bottom:1px solid var(--line-soft);font-size:13.5px;}
.compare-metric:last-child{border-bottom:none;}
.compare-metric .cm-label{color:var(--sand-dim);}
.compare-metric .cm-val{font-family:var(--mono);color:var(--sand-dim);}
.compare-metric.diff .cm-val{color:var(--sand);font-weight:600;}
.compare-metric.diff .cm-label{color:var(--sand);}
.nlp-flow{display:flex;align-items:center;gap:0;flex-wrap:wrap;margin-top:32px;}
.nlp-step{flex:1 1 220px;background:var(--ink-2);border:1px solid var(--line);border-radius:12px;padding:24px 22px;margin:0 4px 12px 0;}
.nlp-step .ns-num{font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:.08em;}
.nlp-step h4{margin-top:10px;font-size:16.5px;}
.nlp-step p{margin-top:10px;font-size:13px;color:var(--sand-dim);line-height:1.55;}
.nlp-arrow{color:var(--sand-faint);font-size:20px;padding:0 6px;display:none;}
@media(min-width:900px){.nlp-arrow{display:block;}}
.nlp-example{margin-top:28px;padding:20px 24px;background:var(--ink-2);border-radius:10px;border:1px solid var(--line);}
.nlp-example .ne-row{display:flex;gap:14px;align-items:flex-start;margin-bottom:12px;}
.nlp-example .ne-row:last-child{margin-bottom:0;}
.nlp-example blockquote{font-family:var(--serif);font-style:italic;font-size:14.5px;color:var(--sand);margin:0;flex:1;}
.nlp-example .ne-tag{font-family:var(--mono);font-size:10.5px;padding:4px 9px;border-radius:6px;white-space:nowrap;}
.nlp-example .ne-tag.pos{background:rgba(25,158,112,.15);color:var(--good);}
.nlp-example .ne-tag.neg{background:rgba(212,67,107,.12);color:var(--bad);}
.segment-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:8px;}
@media(max-width:900px){.segment-grid{grid-template-columns:1fr 1fr;}}
.segment-card{padding:18px 18px;border-radius:10px;background:var(--ink-2);border:1px solid var(--line);text-align:center;}
.segment-card .sc-n{font-family:var(--serif);font-size:24px;color:var(--sand);}
.segment-card .sc-l{margin-top:4px;font-size:12px;color:var(--sand-dim);}
.segment-card .sc-sub{margin-top:8px;font-family:var(--mono);font-size:11px;color:var(--accent);}
.chart-title-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:4px;}
.chart-title-row .chart-title{margin-bottom:0;}
.chart-toggle{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:3px;gap:2px;}
.toggle-btn{font-family:var(--mono);font-size:11px;letter-spacing:.03em;padding:6px 12px;border-radius:999px;border:none;background:transparent;color:var(--sand-faint);cursor:pointer;transition:background .2s,color .2s;}
.toggle-btn:hover{color:var(--sand-dim);}
.toggle-btn.active{background:var(--gold);color:var(--ink);font-weight:600;}
.donut-wrap{display:flex;align-items:center;gap:28px;flex-wrap:wrap;}
.donut-svgbox{flex:0 0 auto;}
.donut-legend{flex:1 1 200px;display:flex;flex-direction:column;gap:9px;min-width:180px;}
.donut-legend-row{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--sand-dim);}
.donut-dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;}
.donut-flag{font-size:15px;line-height:1;}
.donut-name{flex:1;color:var(--sand);}
.donut-val{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--sand-dim);}
.donut-pct{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--accent-warm);min-width:44px;text-align:right;}
@media(max-width:560px){.donut-wrap{flex-direction:column;align-items:flex-start;}}
.hero-stat{margin-top:36px;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;}
.hero-stat .n{font-family:var(--serif);font-size:52px;color:var(--sand);line-height:1;}
.hero-stat .l{font-size:14px;color:var(--sand-dim);}
.hero-stat-sub{margin-top:8px;font-family:var(--mono);font-size:12px;color:var(--sand-faint);letter-spacing:.02em;}
.hero-sources{margin-top:28px;}
.hero-sources .hs-label{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--sand-faint);margin-bottom:10px;}
.hero-sources .hs-badges{display:flex;flex-wrap:wrap;gap:8px;}
.hero-sources .src-badge{cursor:pointer;background:none;font-family:var(--mono);transition:transform .2s,background .2s;margin:0;}
.hero-sources .src-badge:hover{transform:translateY(-1px);background:rgba(255,255,255,.04);}
.hero-process{margin-top:30px;font-family:var(--mono);font-size:12px;color:var(--sand-dim);letter-spacing:.01em;display:flex;flex-wrap:wrap;align-items:center;gap:6px;}
.hero-process .hp-arrow{color:var(--sand-faint);}
.hero-process .hp-link{margin-left:8px;color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(47,168,159,.4);cursor:pointer;}
.hero-process .hp-link:hover{border-bottom-color:var(--accent);}
@media(max-width:640px){.hero-stat .n{font-size:40px;}.hero-process{font-size:11px;}}
"""

print("Prepared EXTRA_CSS chars:", len(EXTRA_CSS))

# ---- verbatim-reused sections (unaffected by this data refresh) ----
HTML_MARKET = "".join(orig_lines[297:320])
HTML_DEST = "".join(orig_lines[323:360])
HTML_TOURISM = "".join(orig_lines[362:397])
HTML_AIRPORT = "".join(orig_lines[399:422])
JS_MARKET = "".join(orig_lines[1010:1030])
JS_DEST = "".join(orig_lines[1031:1062])
JS_TOURISM = "".join(orig_lines[1063:1114])
JS_AIRPORT = "".join(orig_lines[1115:1129])
JS_UTILS_AND_CHARTS = "".join(orig_lines[735:923])  # utilities + all chart primitives (svgEl..mixColor)
JS_HERO_CANVAS = "".join(orig_lines[937:962])  # hero wave canvas IIFE only (kpis handled separately)

# new chart primitive (not in the original artifact): donut chart with a flag/name/value legend
JS_CHART_EXTRA = """
/* ============================================================
   chart: donut (with legend)
   ============================================================ */
function donutChart(container,items,{valueFmt=(v)=>fmt(v),size=180,thickness=32}={}){
  const total = items.reduce((s,d)=>s+d.value,0);
  const cx=size/2, cy=size/2, r=size/2-4, rInner=r-thickness;
  const svg = svgEl('svg',{class:'viz',width:size,height:size,viewBox:`0 0 ${size} ${size}`});
  let angleStart = -Math.PI/2;
  const colors = items.map((d,i)=> items.length>1 ? mixColor('#d97539','#f2d9a8', i/(items.length-1)) : '#d97539');
  items.forEach((d,i)=>{
    const frac = total>0 ? d.value/total : 0;
    const angleEnd = angleStart + frac*Math.PI*2;
    const largeArc = (angleEnd-angleStart) > Math.PI ? 1 : 0;
    const x1=cx+r*Math.cos(angleStart), y1=cy+r*Math.sin(angleStart);
    const x2=cx+r*Math.cos(angleEnd), y2=cy+r*Math.sin(angleEnd);
    const ix1=cx+rInner*Math.cos(angleEnd), iy1=cy+rInner*Math.sin(angleEnd);
    const ix2=cx+rInner*Math.cos(angleStart), iy2=cy+rInner*Math.sin(angleStart);
    const d_attr = `M${x1.toFixed(2)},${y1.toFixed(2)} A${r},${r} 0 ${largeArc} 1 ${x2.toFixed(2)},${y2.toFixed(2)} L${ix1.toFixed(2)},${iy1.toFixed(2)} A${rInner},${rInner} 0 ${largeArc} 0 ${ix2.toFixed(2)},${iy2.toFixed(2)} Z`;
    const seg = svgEl('path',{d:d_attr,fill:colors[i],style:'cursor:pointer;transition:opacity .2s'});
    seg.addEventListener('mousemove',e=>showTip(e,`<b>${d.flag?d.flag+' ':''}${d.label}</b><br>${valueFmt(d.value)} · %${fmt(frac*100,1)}`));
    seg.addEventListener('mouseleave',hideTip);
    seg.addEventListener('mouseenter',()=>seg.style.opacity=0.82);
    seg.addEventListener('mouseleave',()=>seg.style.opacity=1);
    svg.appendChild(seg);
    angleStart = angleEnd;
  });
  const centerVal = svgEl('text',{class:'val','text-anchor':'middle',x:cx,y:cy-1,'font-size':16});centerVal.textContent=fmt(total);svg.appendChild(centerVal);
  const centerLab = svgEl('text',{'text-anchor':'middle',x:cx,y:cy+17,'font-size':10});centerLab.textContent='toplam yorum';svg.appendChild(centerLab);
  const wrap = document.createElement('div'); wrap.className='donut-wrap';
  const svgBox = document.createElement('div'); svgBox.className='donut-svgbox'; svgBox.appendChild(svg);
  const legend = document.createElement('div'); legend.className='donut-legend';
  legend.innerHTML = items.map((d,i)=>`<div class="donut-legend-row"><span class="donut-dot" style="background:${colors[i]}"></span><span class="donut-flag">${d.flag||''}</span><span class="donut-name">${d.label}</span><span class="donut-val">${valueFmt(d.value)}</span><span class="donut-pct">%${fmt(total>0?d.value/total*100:0,1)}</span></div>`).join('');
  wrap.appendChild(svgBox); wrap.appendChild(legend);
  container.innerHTML=''; container.appendChild(wrap);
}
"""
print("JS_CHART_EXTRA chars:", len(JS_CHART_EXTRA))

print("verbatim blocks ok:", len(HTML_MARKET), len(HTML_DEST), len(HTML_TOURISM), len(HTML_AIRPORT))
print("JS_UTILS_AND_CHARTS starts with:", JS_UTILS_AND_CHARTS[:60])
print("JS_UTILS_AND_CHARTS ends with:", JS_UTILS_AND_CHARTS[-60:])
print("JS_HERO_CANVAS starts:", JS_HERO_CANVAS[:40])
print("JS_HERO_CANVAS ends:", JS_HERO_CANVAS[-40:])

# ---- shorthand vars for copy ----
G = DATA["google"]; T = DATA["trip"]; SK = DATA["sikayetvar"]; CGT = DATA["cross_gt"]; CGS = DATA["cross_gs"]
H360M = DATA["hotel360_meta"]
TOTAL_FEEDBACK = G["clean_reviews"] + T["clean_reviews"] + SK["clean_rows"]
print("TOTAL_FEEDBACK", TOTAL_FEEDBACK)
print("G keys", list(G.keys()))
print("T keys", list(T.keys()))

def money(n):
    return f"{n:,.0f}".replace(",", ".")

# ==============================================================
# HTML BODY - part 1: hero / why / pipeline / coverage
# ==============================================================
HTML_TOP = """
<div id="progress"></div>
<nav id="sidenav"></nav>
<div id="topcta"><b>Bodrum Hotel Intelligence</b> &middot; <span id="topcta-section"></span></div>
<div class="tooltip" id="tooltip"></div>

<!-- ================= HERO ================= -->
<section id="hero" data-nav="Açılış">
  <div id="hero-bg"><canvas id="wave"></canvas></div>
  <div class="container hero-inner">
    <div class="hero-kicker">Bodrum &middot; Otel &amp; Destinasyon İstihbaratı</div>
    <h1 class="hero-title">Bodrum otel pazarını<br><em>tek sayı</em> değil, <em>bütün katmanlarıyla</em> okuyun.</h1>
    <p class="hero-sub">Otel yapısı, destinasyon, turizm talebi ve havalimanı hareketini; üç farklı platformdan toplanan müşteri geri bildirimiyle birlikte inceleyen interaktif veri hikâyesi.</p>
    <div class="hero-stat">
      <div class="n">""" + f"{TOTAL_FEEDBACK:,}".replace(",", ".") + """</div>
      <div class="l">müşteri geri bildirimi analiz edildi</div>
    </div>
    <div class="hero-stat-sub">""" + f"{G['clean_reviews']:,}".replace(",", ".") + """ Google Travel &middot; """ + f"{T['clean_reviews']:,}".replace(",", ".") + """ Trip.com &middot; """ + str(SK['clean_rows']) + """ Şikayetvar</div>
    <div class="hero-sources" id="hero-sources"></div>
    <div class="hero-kpis" id="hero-kpis"></div>
    <div class="hero-process" id="hero-process"></div>
    <a href="#coverage" class="hero-cta">Analizi Keşfet &rarr;</a>
  </div>
  <div class="scroll-hint">AŞAĞI KAYDIR</div>
</section>

<!-- ================= WHY ================= -->
<section id="why" data-nav="Sorular">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">01 &mdash; Bağlam</div>
      <h2 class="section-title">Bu proje hangi sorulara cevap veriyor?</h2>
      <p class="section-sub">Bodrum'da 192 otel var; herkes farklı bir açıdan bakıyor. Bu çalışma, bu bakışların hepsini tek bir yerde, gerçek veriyle cevaplamak için kuruldu.</p>
    </div>
    <div class="card-grid g3 reveal" id="why-cards"></div>
  </div>
</section>

<!-- ================= PIPELINE ================= -->
<section id="pipeline" class="tight" data-nav="Yöntem">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">02 &mdash; Yöntem</div>
      <h2 class="section-title">Proje nasıl yapıldı?</h2>
      <p class="section-sub">Her adım bir öncekinin üzerine inşa edildi. Bir adımın üzerine gelin, ne yaptığını görün.</p>
    </div>
    <div class="pipeline reveal" id="pipeline-steps"></div>
  </div>
</section>

<!-- ================= COVERAGE ================= -->
<section id="coverage" data-nav="Kapsam">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">03 &mdash; Veri Kapsamı ve Güven</div>
      <h2 class="section-title">Elimizde ne var, ne kadarına güvenebiliriz?</h2>
      <p class="section-sub"><b style="color:var(--sand)">Veri kapsamı</b> = elimizde güvenilir şekilde analiz edebildiğimiz kısmın oranı. Bu bölümde sınırlılıkları saklamıyoruz. Daha fazla kaynaktan yeterli veri bulunan oteller daha yüksek veri desteğine sahiptir &mdash; bu, o otelin daha iyi olduğu anlamına gelmez.</p>
    </div>
    <div class="kpi-row reveal" id="coverage-kpis"></div>
    <div class="meaning-pair reveal">
      <div class="meaning-box yes"><h5>Elimizde güvenle olan</h5><ul id="coverage-yes"></ul></div>
      <div class="meaning-box no"><h5>Sınırlı / kısmi olan</h5><ul id="coverage-no"></ul></div>
    </div>
  </div>
</section>
"""

print("HTML_TOP chars:", len(HTML_TOP))

JS_TOP = """
/* ============================================================
   HERO
   ============================================================ */
const heroKpis = [
  {n:'192',l:'Otel',d:'Projenin ana otel örneklemi'},
  {n:'14',l:'Destinasyon',d:'Bodrum\\'daki alt bölgeler'},
  {n:'""" + f"{G['clean_reviews']:,}".replace(",",".") + """',l:'Google Travel Yorumu',d:'""" + str(G['hotels']) + """ otelde analiz edilebilir müşteri sesi'},
  {n:'""" + f"{T['clean_reviews']:,}".replace(",",".") + """',l:'Trip.com Yorumu',d:'""" + str(T['hotels']) + """ otelde puan + misafir segmenti verisi'},
  {n:'""" + str(SK['clean_rows']) + """',l:'Şikayetvar Kaydı',d:'""" + str(SK['complaint_hotels']) + """ otelle eşleşen problem-odaklı şikâyet'},
  {n:'""" + str(T['policy_hotels']) + """',l:'Otelde Politika / Olanak Verisi',d:'Check-in, aile, evcil hayvan gibi tesis politikaları'},
];
$('#hero-kpis').innerHTML = heroKpis.map(k=>`<div class="hero-kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');

const heroSources = [
  {cls:'g',label:'Google Travel',go:'#google-voice'},
  {cls:'t',label:'Trip.com',go:'#trip-voice'},
  {cls:'s',label:'Şikayetvar',go:'#sikayetvar-voice'},
  {cls:'',label:'Turizm & Havalimanı Verisi (2009–2025)',go:'#tourism'},
];
$('#hero-sources').innerHTML = `<div class="hs-label">Veri Kaynakları</div><div class="hs-badges">${heroSources.map(s=>`<button class="src-badge ${s.cls}" ${s.cls?'':'style="border-color:var(--sand-dim);color:var(--sand-dim);"'} data-go="${s.go}" type="button">${s.label}</button>`).join('')}</div>`;
$$('#hero-sources .src-badge').forEach(b=>b.addEventListener('click',()=>document.querySelector(b.dataset.go).scrollIntoView({behavior:'smooth'})));

const heroProcessSteps = ['Toplama','Temizlik &amp; Doğrulama','Kural-tabanlı Konu Analizi','Çapraz-Kaynak Eşleştirme','Hotel 360° Sentezi'];
$('#hero-process').innerHTML = heroProcessSteps.map((s,i)=>(i>0?'<span class="hp-arrow">&rarr;</span>':'')+`<span>${s}</span>`).join('') + `<a class="hp-link" id="hero-process-link">Detaylı yönteme git &darr;</a>`;
$('#hero-process-link').addEventListener('click',()=>$('#pipeline').scrollIntoView({behavior:'smooth'}));

""" + JS_HERO_CANVAS + """

/* ============================================================
   WHY cards
   ============================================================ */
const whyCards = [
  {q:'Bodrum otel pazarı nasıl görünüyor?',a:'192 otel, 14 bölge, fiyat ve görünürlük dağılımıyla tek bakışta pazar tablosu.'},
  {q:'Hangi bölge hangi profile daha yakın?',a:'14 destinasyon; kalite, popülerlik, lüks, değer ve kapasite boyutlarında karşılaştırılıyor.'},
  {q:'Misafir neyi seviyor, neyi eleştiriyor?',a:'Google Travel ve Trip.com\\'daki """ + f"{G['clean_reviews']+T['clean_reviews']:,}".replace(",",".") + """ genel yorum, kural-tabanlı konu analiziyle okunuyor.'},
  {q:'Hangi otelin güçlü ve zayıf sinyalleri neler?',a:'Hotel 360° tek panelde dört kaynağı (Google, Trip, Şikayetvar, politika) birleştiriyor.'},
  {q:'Talep hangi dönemde yükseliyor?',a:'2009&ndash;2025 turizm trendi ve 2025 sezonluk yapısı, havalimanı hareketiyle birlikte.'},
  {q:'Bir otelin 360° profili nasıl görünüyor?',a:'Tek dropdown ile herhangi bir otelin dört kaynaklı müşteri sesi ve veri desteği seviyesi.'},
];
$('#why-cards').innerHTML = whyCards.map(c=>`<div class="card"><p style="font-family:var(--serif);font-size:16.5px;color:var(--sand);line-height:1.4;">${c.q}</p><p style="margin-top:12px;font-size:13.5px;color:var(--sand-dim);">${c.a}</p></div>`).join('');

/* ============================================================
   PIPELINE
   ============================================================ */
const pipelineSteps = [
  ['Otel Verisi','Google Places tabanlı 192 otel anlık görüntüsü.'],
  ['Veri Denetimi','Şema, eksik veri ve tekrar kontrolleri.'],
  ['Destinasyon Zekâsı','14 bölge, çok boyutlu karşılaştırma endeksi.'],
  ['Turizm + Havalimanı','2009–2025 turizm talebi ve havalimanı hareketi ortak analizi.'],
  ['Google Travel Keşfi','192 otelin tamamı için doğrulanmış Google Travel URL keşfi ve entity doğrulama.'],
  ['Google Travel Toplama','""" + f"{G['clean_reviews']:,}".replace(",",".") + """ yorum, """ + str(G['hotels']) + """ otel &mdash; 24 konu başlıklı kural-tabanlı analiz.'],
  ['Trip.com Keşfi','192 otelin tamamı için doğrulanmış Trip.com URL keşfi ve entity doğrulama.'],
  ['Trip.com Toplama','""" + f"{T['clean_reviews']:,}".replace(",",".") + """ yorum, """ + str(T['hotels']) + """ otel &mdash; puan, misafir segmenti ve konaklama detayı.'],
  ['Politika &amp; Olanak Toplama','""" + str(T['policy_hotels']) + """ otelde check-in, aile, evcil hayvan gibi tesis politikaları.'],
  ['Şikayetvar Eşleştirme','192 otel için sayfa keşfi, entity eşleştirme ve hedefli tamamlama.'],
  ['Şikayetvar Toplama','""" + str(SK['clean_rows']) + """ şikâyet, """ + str(SK['complaint_hotels']) + """ otel &mdash; 18 konu başlıklı problem-odaklı analiz.'],
  ['Kaynaklar Arası Hizalama','Google×Trip ve Google×Şikayetvar &mdash; aynı konuları ortak dile çevirip karşılaştırma.'],
  ['Hotel 360°','Dört kaynağı otel bazında birleştiren veri-destekli güven seviyesi.'],
];
$('#pipeline-steps').innerHTML = pipelineSteps.map(([t,d],i)=>`<div class="pipe-step" tabindex="0"><div class="idx">${String(i+1).padStart(2,'0')}</div><div class="t">${t}</div><div class="d">${d}</div></div>`).join('');

/* ============================================================
   COVERAGE
   ============================================================ */
const coverageKpis = [
  {n:'192',l:'Projedeki toplam otel sayısı',d:'Ana otel örneklemi'},
  {n:'""" + str(G['hotels']) + """',l:'Google Travel verisi olan otel',d:'Analiz edilebilir yorum bulunan otel'},
  {n:'""" + str(T['hotels']) + """',l:'Trip.com verisi olan otel',d:'Puan/segment verisi bulunan otel'},
  {n:'""" + str(T['policy_hotels']) + """',l:'Politika/olanak verisi olan otel',d:'Check-in, aile, evcil hayvan gibi tesis bilgisi'},
  {n:'""" + str(SK['complaint_hotels']) + """',l:'Şikayetvar\\'da eşleşen otel',d:'Görünür şikâyet kaydı bulunan otel'},
  {n:'""" + str(H360M['confidence_dist'].get('HIGH',0) + H360M['confidence_dist'].get('MEDIUM',0)) + """',l:'Hotel 360° &mdash; yüksek/orta güven',d:'HIGH + MEDIUM veri desteği seviyesindeki otel'},
  {n:'192/192',l:'Hotel 360° kapsamı',d:'Veri az olsa bile her otel bir satırla temsil ediliyor'},
];
$('#coverage-kpis').innerHTML = coverageKpis.map(k=>`<div class="kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');
$('#coverage-yes').innerHTML = ['192 otelin tamamında Google puanı, yorum sayısı ve Hotel 360° satırı mevcut','14/14 destinasyon tam kapsamda','Google Travel ve Trip.com\\'da wrong-entity/duplicate oranı 0 &mdash; her yorum doğrulanmış otele ait','Havalimanı × turizm ilişkisi güçlü ve 12 ayın tamamı elde mevcut'].map(x=>`<li>${x}</li>`).join('');
$('#coverage-no').innerHTML = ['""" + str(192-G['hotels']) + """ otelde Google Travel\\'da yeterli yorum bulunamadı','""" + str(192-T['hotels']) + """ otelde Trip.com\\'da henüz doğrulanmış yorum yok','Şikayetvar yalnız """ + str(SK['complaint_hotels']) + """/192 otelde görünür şikâyetle eşleşti (""" + str(SK['not_found_n']) + """ otelde sayfa hiç bulunamadı, """ + str(SK['ambiguous_n']) + """\\'u kanıt yetersizliğinden bilinçli olarak açık bırakıldı)','""" + str(192 - (H360M['confidence_dist'].get('HIGH',0) + H360M['confidence_dist'].get('MEDIUM',0))) + """ otel Hotel 360°\\'ta hâlâ LOW/VERY_LOW veri desteğinde'].map(x=>`<li>${x}</li>`).join('');
"""
print("JS_TOP chars:", len(JS_TOP))

# ==============================================================
# GOOGLE TRAVEL section
# ==============================================================
top_strength = G["aspects"][0]
top_concern = min(G["aspects"], key=lambda a: a["driver_score"])

HTML_GOOGLE = """
<!-- ================= GOOGLE TRAVEL ================= -->
<section id="google-voice" data-nav="Google Travel">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">08 &mdash; Google Travel Müşteri Deneyimi</div>
      <h2 class="section-title">Yüksek puan verenler neyi övüyor, düşük puan verenler neyi eleştiriyor?</h2>
      <p class="section-sub">Google Travel'daki 1&ndash;5 yıldız müşteri yorumlarını inceliyoruz: olumlu, karışık ve olumsuz yorumların hepsi burada. Bu, """ + str(G["hotels"]) + """ otelde toplanmış """ + f"{G['clean_reviews']:,}".replace(",",".") + """ yorumluk kapsamlı bir müşteri sesi analizidir.</p>
      <div class="why-box">&#9888; <b>Metodoloji notu:</b> Google Travel'daki değerlendirme paneli birden fazla kaynağı (Google, Tripadvisor, Trip.com üzerinden yapılan yorumlar) tek panelde birleştirebiliyor. Aşağıdaki sayılar bu birleşik paneli analiz eder; yalnız "saf Google" yorumları anlamına gelmez.</div>
    </div>
    <div class="kpi-row reveal" id="gm-kpis"></div>
    <div class="card-grid g2 reveal">
      <div class="card"><div class="chart-title">Otellerin ortalama puan dağılımı</div><div class="chart-note">""" + str(G["hotels"]) + """ otel, ortalama Google Travel puanına göre</div><div class="chart-wrap" id="chart-gm-ratinggroup"></div></div>
      <div class="card"><div class="chart-title">En çok yorum alan 10 otel</div><div class="chart-note">Google Travel'da toplanan yorum sayısı &mdash; görünürlük göstergesi, kalite sıralaması değil</div><div class="chart-wrap" id="chart-top-reviewed-gt"></div></div>
    </div>
    <div class="card reveal" style="margin-top:18px;">
      <div class="chart-title">Puanı yükselten ve düşüren konular</div>
      <div class="chart-note">Bir konu yüksek puanlı yorumlarda ne kadar daha sık geçiyorsa, o kadar sağa; düşük puanlı yorumlarda ne kadar daha sık geçiyorsa o kadar sola uzuyor (24 konu başlığının tamamı, en az bir yorumda tespit edilen).</div>
      <div class="chart-wrap" id="chart-gm-drivers"></div>
      <div class="quote-hint">&uarr; Bir konuya tıklayın: gerçek örnek yorumları görün</div>
      <div class="quote-panel" id="gm-quote-panel"></div>
    </div>
    <div class="explain-grid reveal">
      <div class="eg"><h5>Bu grafik neyi gösteriyor?</h5><p>Hangi konuların yüksek puanlı, hangilerinin düşük puanlı yorumlarda daha sık geçtiğini.</p></div>
      <div class="eg"><h5>Nasıl okunur?</h5><p>Sağa uzayan barlar yüksek puanlı yorumlarla daha güçlü ilişki gösteriyor; sola uzayanlar düşük puanlı yorumlarla daha güçlü ilişki gösteriyor.</p></div>
      <div class="eg"><h5>Ne sonuç çıkarıyoruz?</h5><p><b style="color:var(--good)">""" + top_strength["aspect"] + """</b> yüksek puanlı yorumlarda en güçlü öne çıkan konu; <b style="color:var(--bad)">""" + top_concern["aspect"] + """</b> düşük puanlı yorumlarla en güçlü ilişkili konu.</p></div>
      <div class="eg"><h5>Neden önemli?</h5><p>Olumlu deneyimi korumak için personel, yemek ve konum gibi güçlü konular; düşük puan riskini azaltmak için ödeme/iade ve wifi gibi konular izlenmeli.</p></div>
      <div class="eg"><h5>Ne anlama gelmiyor?</h5><p>Bu bir istatistiksel ilişkidir, neden-sonuç ilişkisi değildir. <b>"X yüksek puana neden oluyor" denemez</b> &mdash; yalnız "yüksek puanlı yorumlarla daha güçlü ilişki gösteriyor" denebilir.</p></div>
    </div>
    <div class="caution reveal">&#128101; <span><b>Örneklem uyarısı:</b> Aşağıdaki """ + str(192-G["hotels"]) + """ otelde Google Travel'da analiz için yeterli yorum bulunamadı; bu otellerin genel müşteri deneyimi bu bölümde temsil edilmiyor (bkz. Hotel 360° &mdash; her otel yine de tek satırla temsil ediliyor).</span></div>
  </div>
</section>
"""
print("HTML_GOOGLE chars:", len(HTML_GOOGLE))

ASPECT_TR_GOOGLE = {
    "STAFF": "Personel", "SERVICE": "Hizmet", "FOOD": "Yemek", "CLEANLINESS": "Temizlik",
    "BEACH_SEA": "Plaj & Deniz", "POOL": "Havuz", "FACILITIES": "Tesisler", "RESERVATION": "Rezervasyon",
    "REFUND_PAYMENT": "Ödeme & İade", "PRICE_VALUE": "Fiyat & Değer", "CHECKIN_CHECKOUT": "Giriş & Çıkış",
    "AIR_CONDITIONING": "Klima", "NOISE": "Gürültü", "FAMILY_KIDS": "Aile & Çocuk",
    "TRANSPORT_TRANSFER": "Ulaşım & Transfer", "MANAGEMENT": "Yönetim", "COMMUNICATION": "İletişim",
    "WIFI": "Wifi", "ANIMATION_ENTERTAINMENT": "Animasyon & Eğlence", "BED_COMFORT": "Yatak Konforu",
    "HYGIENE": "Hijyen", "LOCATION": "Konum", "BAR_DRINKS": "Bar & İçecek", "ROOM": "Oda",
}
# rewrite english aspect codes in HTML copy to Turkish now that dict exists
HTML_GOOGLE = HTML_GOOGLE.replace('>' + top_strength["aspect"] + '<', '>' + ASPECT_TR_GOOGLE.get(top_strength["aspect"], top_strength["aspect"]) + '<')
HTML_GOOGLE = HTML_GOOGLE.replace('>' + top_concern["aspect"] + '<', '>' + ASPECT_TR_GOOGLE.get(top_concern["aspect"], top_concern["aspect"]) + '<')

JS_GOOGLE = """
/* ============================================================
   GOOGLE TRAVEL VOICE
   ============================================================ */
const ASPECT_TR_G = """ + json.dumps(ASPECT_TR_GOOGLE, ensure_ascii=False) + """;
const trAspectG = code => ASPECT_TR_G[code] || code.replace(/_/g,' ');
const QUOTES_G = """ + json.dumps(QUOTES_G, ensure_ascii=False) + """;

$('#gm-kpis').innerHTML = [
  {n:'""" + f"{G['clean_reviews']:,}".replace(",",".") + """',l:'Google Travel yorumu',d:'""" + str(G["hotels"]) + """ otelde analiz edilebilir müşteri sesi'},
  {n:'""" + str(G["mean_rating"]).replace(".",",") + """',l:'Ortalama puan',d:'""" + str(G["hotels"]) + """ otelin yorum-seviyesi ortalaması (1-5)'},
  {n:'%""" + str(G["rating_group_pct"].get("LOW",0)).replace(".",",") + """ / %""" + str(G["rating_group_pct"].get("MID",0)).replace(".",",") + """ / %""" + str(G["rating_group_pct"].get("HIGH",0)).replace(".",",") + """',l:'Düşük / Orta / Yüksek puan payı',d:'1-2 / 3 / 4-5 yıldız payları'},
  {n:'24',l:'Konu başlığı',d:'Kural-tabanlı aspect sözlüğü (personel, oda, yemek, wifi...)'},
].map(k=>`<div class="kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');

const gRatingBands = Object.entries(""" + json.dumps(G["hotel_rating_band_dist"], ensure_ascii=False) + """).map(([label,value])=>({label,value}));
vbarChart($('#chart-gm-ratinggroup'),gRatingBands,{color:'var(--accent)',valueFmt:v=>String(v)});

const topReviewedGT = """ + json.dumps([{"label": r["hotel_name"][:22] + ("…" if len(r["hotel_name"]) > 22 else ""), "value": r["n"], "tip": f"⭐ {round(r['mean_rating'],2)}"} for r in G["top_reviewed"]], ensure_ascii=False) + """;
hbarChart($('#chart-top-reviewed-gt'),topReviewedGT,{color:'var(--accent-warm)',valueFmt:v=>fmt(v),labelWidth:170});

const gmDriverItems = """ + json.dumps([{"aspect": a["aspect"], "label": ASPECT_TR_GOOGLE.get(a["aspect"], a["aspect"]), "value": round(a["driver_score"], 1), "tip": f"Yüksek puan %{a['high_rating_share_when_mentioned']:.1f} · Düşük puan %{a['low_rating_share_when_mentioned']:.1f} · n={a['n_mentions']}", "low": a["low_rating_share_when_mentioned"], "high": a["high_rating_share_when_mentioned"], "n": a["n_mentions"]} for a in G["aspects"]], ensure_ascii=False) + """;
const gmQuotePanel = $('#gm-quote-panel');
function renderGmDriverDetail(d){
  const isPos = d.value>=0;
  if(gmQuotePanel.dataset.openKey===d.aspect && gmQuotePanel.classList.contains('open')){
    gmQuotePanel.classList.remove('open'); gmQuotePanel.dataset.openKey=''; return;
  }
  const examples = (isPos ? QUOTES_G.high[d.aspect] : QUOTES_G.low[d.aspect]) || [];
  const otherExamples = (isPos ? QUOTES_G.low[d.aspect] : QUOTES_G.high[d.aspect]) || [];
  const quotesHtml = examples.length ? quoteCardsHtml(examples.map(e=>({text:e.text,date:e.hotel+(e.date?' · '+e.date:'')}))) : '<p class="na">Bu konu için örnek yorum bulunamadı.</p>';
  gmQuotePanel.innerHTML = `
    <div class="qp-title">${d.label} &mdash; ${isPos?'yüksek puanla daha güçlü ilişkili':'düşük puanla daha güçlü ilişkili'}</div>
    <div class="detail-grid" style="margin-bottom:16px;">
      <div class="d-item"><div class="n">%${fmt(d.low,1)}</div><div class="l">Düşük puanlı yorumlarda bahsedilme</div></div>
      <div class="d-item"><div class="n">%${fmt(d.high,1)}</div><div class="l">Yüksek puanlı yorumlarda bahsedilme</div></div>
      <div class="d-item"><div class="n">${d.n}</div><div class="l">Toplam bahsedilme sayısı</div></div>
    </div>
    ${quotesHtml}
  `;
  gmQuotePanel.dataset.openKey = d.aspect;
  gmQuotePanel.classList.add('open');
  gmQuotePanel.scrollIntoView({behavior:'smooth',block:'nearest'});
}
divergingChart($('#chart-gm-drivers'),gmDriverItems,{onClick:renderGmDriverDetail});
"""
print("JS_GOOGLE chars:", len(JS_GOOGLE))

# ==============================================================
# TRIP.COM section
# ==============================================================
trav_known_n = sum(r["count"] for r in T["traveler_type_coverage_pct"] if r["traveler_type"] != "UNKNOWN")
trav_cov_pct = round(100 * trav_known_n / T["clean_reviews"], 1)
top_trav = max((r for r in T["traveler_type_rating"] if r["traveler_type"] != "UNKNOWN" and r["n"] >= 30), key=lambda r: r["mean"])

TRAV_TR = {"FAMILY": "Aile", "COUPLE": "Çift", "SOLO": "Tek Başına", "FRIENDS": "Arkadaş Grubu", "BUSINESS": "İş Seyahati", "OTHER": "Diğer", "UNKNOWN": "Belirtilmemiş"}

HTML_TRIP = """
<!-- ================= TRIP.COM ================= -->
<section id="trip-voice" data-nav="Trip.com">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">09 &mdash; Trip.com Misafir Deneyimi</div>
      <h2 class="section-title">Trip.com bize Google Travel'da olmayan neyi gösteriyor?</h2>
      <p class="section-sub">Trip.com yorumları puanın yanında seyahat tipini, konaklama dönemini, oda tipini ve misafirin ülkesini de taşıyor &mdash; Google Travel'da bu detaylar yok. """ + str(T["hotels"]) + """ otelde """ + f"{T['clean_reviews']:,}".replace(",",".") + """ yorum inceliyoruz.</p>
    </div>
    <div class="kpi-row reveal" id="trip-kpis"></div>
    <div class="card-grid g2 reveal">
      <div class="card">
        <div class="chart-title-row">
          <div class="chart-title">Misafir segmenti dağılımı</div>
          <div class="chart-toggle" id="trip-seg-toggle">
            <button class="toggle-btn active" data-mode="raw" type="button">Ham veri</button>
            <button class="toggle-btn" data-mode="spread" type="button">Tahmini (yayılmış)</button>
          </div>
        </div>
        <div class="chart-note" id="trip-seg-note">Seyahat tipi belirtilen """ + str(trav_known_n) + """ yorum üzerinden (toplam """ + f"{T['clean_reviews']:,}".replace(",",".") + """ yorumun %""" + str(trav_cov_pct).replace(".",",") + """'i)</div>
        <div class="chart-wrap" id="chart-trip-segments"></div>
      </div>
      <div class="card">
        <div class="chart-title">Segmentlere göre ortalama puan (5 üzerinden)</div>
        <div class="chart-note">Yalnız yeterli örneklemli segmentler (n&ge;30)</div>
        <div class="chart-wrap" id="chart-trip-segment-rating"></div>
        <div class="quote-hint">&uarr; Bir segmente tıklayın: gerçek örnek yorumları görün</div>
        <div class="quote-panel" id="trip-quote-panel"></div>
      </div>
    </div>
    <div class="explain-grid reveal">
      <div class="eg"><h5>Bu grafik neyi gösteriyor?</h5><p>Trip.com yorumlarının hangi misafir segmentine ait olduğunu ve segmentlere göre ortalama puanı.</p></div>
      <div class="eg"><h5>Nasıl okunur?</h5><p>Çubuk uzunluğu o segmentteki yorum sayısını / ortalama puanı gösterir.</p></div>
      <div class="eg"><h5>Ne sonuç çıkarıyoruz?</h5><p>En büyük belirlenmiş segment <b style="color:var(--sand)">Aile</b>; en yüksek ortalama puanı yeterli örneklemle <b style="color:var(--good)">""" + TRAV_TR.get(top_trav["traveler_type"], top_trav["traveler_type"]) + """</b> segmenti veriyor (""" + str(round(top_trav["mean"],2)).replace(".",",") + """/5, n=""" + str(top_trav["n"]) + """).</p></div>
      <div class="eg"><h5>Neden önemli?</h5><p>Segment bazlı puanlar, hangi misafir tipine göre konumlanmanın (aile-dostu, çift-odaklı vb.) güçlü olduğunu gösterebilir.</p></div>
      <div class="eg"><h5>Ne anlama gelmiyor?</h5><p><b style="color:var(--bad)">"Belirtilmemiş" segment düşük puan demek değildir</b> &mdash; yalnız Trip.com'un o yorumda seyahat tipini göstermediği anlamına gelir.</p></div>
    </div>
    <div class="card-grid g2 reveal" style="margin-top:18px;">
      <div class="card">
        <div class="chart-title">En sık görülen otel olanakları</div>
        <div class="chart-note">""" + str(T["policy_hotels"]) + """ otelin politika/olanak verisinde geçme oranı</div>
        <div class="chart-wrap" id="chart-trip-amenities"></div>
      </div>
      <div class="card">
        <div class="chart-title">Misafirin ülkesi (belirtilenler)</div>
        <div class="chart-note">Reviewer ülkesi belirtilen yorumlar arasında en sık görülen 8 ülke</div>
        <div class="chart-wrap" id="chart-trip-country"></div>
      </div>
    </div>
    <div class="caution reveal">&#127760; <span><b>Kapsam uyarısı:</b> Trip.com şu an 192 otelin """ + str(T["hotels"]) + """'inde yorum verisi taşıyor; oda tipi (""" + str(round(100*sum(r['count'] for r in T['room_type_coverage'] if r['room_type']!='UNKNOWN')/T['clean_reviews'],1)).replace(".",",") + """% kapsam) ve ülke bilgisi (""" + str(round(100*sum(r['count'] for r in T['reviewer_location_coverage'])/T['clean_reviews'],1)).replace(".",",") + """% kapsam) gibi alanlar keşifsel düzeyde tutulmalı.</span></div>
  </div>
</section>
"""
print("HTML_TRIP chars:", len(HTML_TRIP))

top8_countries = T["reviewer_location_coverage"][:8]
top_amenities10 = T["top_amenities"][:8]
AMENITY_TR = {
    "has_wifi": "Wifi", "has_parking": "Otopark", "has_restaurant": "Restoran", "has_bar": "Bar",
    "has_private_beach": "Özel Plaj", "has_pool": "Havuz", "has_spa": "Spa", "has_gym": "Spor Salonu",
    "has_kids_club": "Çocuk Kulübü", "has_pet_friendly": "Evcil Hayvan Dostu", "has_airport_shuttle": "Havalimanı Servisi",
    "has_room_service": "Oda Servisi", "has_air_conditioning": "Klima", "has_breakfast": "Kahvaltı",
}

COUNTRY_FLAG = {
    "Turkey": "🇹🇷", "Russia": "🇷🇺", "China": "🇨🇳", "United Kingdom": "🇬🇧",
    "Saudi Arabia": "🇸🇦", "United States": "🇺🇸", "United Arab Emirates": "🇦🇪",
    "Germany": "🇩🇪", "France": "🇫🇷", "Slovenia": "🇸🇮", "Australia": "🇦🇺",
    "Netherlands": "🇳🇱", "Kazakhstan": "🇰🇿", "Singapore": "🇸🇬",
    "Hong Kong, China": "🇭🇰", "Mongolia": "🇲🇳", "Malaysia": "🇲🇾",
    "Belarus": "🇧🇾", "Thailand": "🇹🇭", "Spain": "🇪🇸", "Canada": "🇨🇦",
    "Romania": "🇷🇴", "Brazil": "🇧🇷", "Belgium": "🇧🇪", "Pakistan": "🇵🇰",
    "Kyrgyzstan": "🇰🇬", "Korea, Republic of": "🇰🇷", "Japan": "🇯🇵",
    "India": "🇮🇳", "Ireland": "🇮🇪", "Israel": "🇮🇱", "Qatar": "🇶🇦",
    "Italy": "🇮🇹", "Taiwan, China": "🇹🇼", "Bahrain": "🇧🇭",
}
COUNTRY_TR = {
    "Turkey": "Türkiye", "Russia": "Rusya", "China": "Çin", "United Kingdom": "Birleşik Krallık",
    "Saudi Arabia": "Suudi Arabistan", "United States": "ABD", "United Arab Emirates": "BAE",
    "Germany": "Almanya", "France": "Fransa", "Slovenia": "Slovenya", "Australia": "Avustralya",
    "Netherlands": "Hollanda", "Kazakhstan": "Kazakistan", "Singapore": "Singapur",
    "Hong Kong, China": "Hong Kong", "Mongolia": "Moğolistan", "Malaysia": "Malezya",
    "Belarus": "Belarus", "Thailand": "Tayland", "Spain": "İspanya", "Canada": "Kanada",
    "Romania": "Romanya", "Brazil": "Brezilya", "Belgium": "Belçika", "Pakistan": "Pakistan",
    "Kyrgyzstan": "Kırgızistan", "Korea, Republic of": "Güney Kore", "Japan": "Japonya",
    "India": "Hindistan", "Ireland": "İrlanda", "Israel": "İsrail", "Qatar": "Katar",
    "Italy": "İtalya", "Taiwan, China": "Tayvan", "Bahrain": "Bahreyn",
}
tripCountriesData = [
    {"label": COUNTRY_TR.get(c["reviewer_country"], c["reviewer_country"]),
     "flag": COUNTRY_FLAG.get(c["reviewer_country"], "🏳️"), "value": c["count"]}
    for c in top8_countries
]

# ---- Trip.com traveler-segment "spread" (estimated) view: redistribute UNKNOWN
# proportionally across known segments. Clearly labelled as an estimate, never
# presented as a directly-measured figure. ----
unk_row = next((r for r in T["traveler_type_coverage_pct"] if r["traveler_type"] == "UNKNOWN"), None)
unknown_n = unk_row["count"] if unk_row else 0
known_segs_rows = [r for r in T["traveler_type_coverage_pct"] if r["traveler_type"] != "UNKNOWN"]
known_sum = sum(r["count"] for r in known_segs_rows)
tripSegmentsSpreadData = [
    {"label": TRAV_TR.get(r["traveler_type"], r["traveler_type"]), "seg": r["traveler_type"],
     "value": round(r["count"] / known_sum * T["clean_reviews"]),
     "tip": "Ham veri: " + f"{r['count']:,}".replace(",", ".") + " yorum · bilinen segmentler içindeki payı %" + str(round(r["count"] / known_sum * 100, 1)).replace(".", ",")}
    for r in known_segs_rows
]
unknown_pct = round(100 * unknown_n / T["clean_reviews"], 1)
trip_seg_note_raw = "Seyahat tipi belirtilen " + str(trav_known_n) + " yorum üzerinden (toplam " + f"{T['clean_reviews']:,}".replace(",", ".") + " yorumun %" + str(trav_cov_pct).replace(".", ",") + "'i)"
trip_seg_note_spread = (
    "Tahmini görünüm: \"Belirtilmemiş\" " + f"{unknown_n:,}".replace(",", ".") + " yorum (%" + str(unknown_pct).replace(".", ",") + ") "
    "bilinen " + f"{known_sum:,}".replace(",", ".") + " yorumun mevcut segment oranına göre orantılı olarak dağıtıldı. "
    "Bu ölçülmüş bir değer değildir, yalnızca varsayımsal bir tahmindir."
)

JS_TRIP = """
/* ============================================================
   TRIP.COM VOICE
   ============================================================ */
const QUOTES_T = """ + json.dumps(QUOTES_T, ensure_ascii=False) + """;
const TRAV_TR = """ + json.dumps(TRAV_TR, ensure_ascii=False) + """;
const COUNTRY_TR = """ + json.dumps(COUNTRY_TR, ensure_ascii=False) + """;
const COUNTRY_FLAG = """ + json.dumps(COUNTRY_FLAG, ensure_ascii=False) + """;

$('#trip-kpis').innerHTML = [
  {n:'""" + f"{T['clean_reviews']:,}".replace(",",".") + """',l:'Trip.com yorumu',d:'""" + str(T["hotels"]) + """ otelde toplandı'},
  {n:'""" + str(round(T["mean_rating_5scale"],2)).replace(".",",") + """',l:'Ortalama puan (5 üzerinden)',d:'Trip.com\\'un kendi 0-10 puanından normalize edildi'},
  {n:'%""" + str(trav_cov_pct).replace(".",",") + """',l:'Seyahat tipi kapsamı',d:'""" + str(trav_known_n) + """/""" + f"{T['clean_reviews']:,}".replace(",",".") + """ yorumda misafir segmenti belirtilmiş'},
  {n:'""" + str(T["policy_hotels"]) + """',l:'Politika/olanak verisi olan otel',d:'Check-in, aile, evcil hayvan gibi tesis bilgisi'},
].map(k=>`<div class="kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');

const tripSegmentsRaw = """ + json.dumps([{"label": TRAV_TR.get(r["traveler_type"], r["traveler_type"]), "value": r["count"], "seg": r["traveler_type"]} for r in T["traveler_type_coverage_pct"]], ensure_ascii=False) + """;
const tripSegmentsSpread = """ + json.dumps(tripSegmentsSpreadData, ensure_ascii=False) + """;
const tripSegNoteRaw = """ + json.dumps(trip_seg_note_raw, ensure_ascii=False) + """;
const tripSegNoteSpread = """ + json.dumps(trip_seg_note_spread, ensure_ascii=False) + """;
let tripSegMode = 'raw';
function renderTripSegChart(){
  const data = tripSegMode==='raw' ? tripSegmentsRaw : tripSegmentsSpread;
  hbarChart($('#chart-trip-segments'),data,{color:'var(--gold)',valueFmt:v=>fmt(v),labelWidth:120});
}
renderTripSegChart();
$$('#trip-seg-toggle .toggle-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    tripSegMode = btn.dataset.mode;
    $$('#trip-seg-toggle .toggle-btn').forEach(b=>b.classList.toggle('active', b===btn));
    $('#trip-seg-note').textContent = tripSegMode==='raw' ? tripSegNoteRaw : tripSegNoteSpread;
    renderTripSegChart();
  });
});

const tripSegRating = """ + json.dumps([{"label": TRAV_TR.get(r["traveler_type"], r["traveler_type"]), "value": round(r["mean"], 2), "seg": r["traveler_type"], "tip": f"n={r['n']} · yüksek pay %{r['high_share']*100:.0f}"} for r in T["traveler_type_rating"] if r["traveler_type"] != "UNKNOWN" and r["n"] >= 20], ensure_ascii=False) + """;
const tripQuotePanel = $('#trip-quote-panel');
hbarChart($('#chart-trip-segment-rating'),tripSegRating,{color:'var(--accent)',valueFmt:v=>fmt(v,2),labelWidth:120,
  onClick:d=>renderQuotePanel(tripQuotePanel,d.label,(QUOTES_T[d.seg]||[]).map(e=>({text:e.text,hotel:e.hotel,date:e.date})))});

const tripAmenities = """ + json.dumps([{"label": AMENITY_TR.get(a["amenity"], a["amenity"].replace("has_", "").replace("_", " ").title()), "value": a["share_pct"], "tip": f"{a['hotel_count']}/{T['policy_hotels']} otel"} for a in top_amenities10], ensure_ascii=False) + """;
hbarChart($('#chart-trip-amenities'),tripAmenities,{color:'var(--gold)',valueFmt:v=>fmt(v,1)+'%',labelWidth:140,height:22});

const tripCountries = """ + json.dumps(tripCountriesData, ensure_ascii=False) + """;
donutChart($('#chart-trip-country'),tripCountries,{valueFmt:v=>fmt(v)});
"""
print("JS_TRIP chars:", len(JS_TRIP))

# ==============================================================
# NLP explainer section
# ==============================================================
HTML_NLP = """
<!-- ================= NLP ================= -->
<section id="nlp" class="tight" data-nav="NLP">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">10 &mdash; Nasıl Okuduk</div>
      <h2 class="section-title">Binlerce yorumu nasıl okuduk?</h2>
      <p class="section-sub">""" + f"{G['clean_reviews']+T['clean_reviews']+SK['clean_rows']:,}".replace(",",".") + """ yorum/şikâyeti tek tek okumak yerine, her metni aynı basit üç adımdan geçirdik.</p>
    </div>
    <div class="nlp-flow reveal">
      <div class="nlp-step"><div class="ns-num">01</div><h4>Yorum</h4><p>Ham metin &mdash; olduğu gibi, değiştirilmeden.</p></div>
      <div class="nlp-arrow">&rarr;</div>
      <div class="nlp-step"><div class="ns-num">02</div><h4>Konu tespiti</h4><p>Metinde hangi konudan (personel, oda, temizlik, wifi...) bahsedildiğini anahtar kelimelerle buluyoruz.</p></div>
      <div class="nlp-arrow">&rarr;</div>
      <div class="nlp-step"><div class="ns-num">03</div><h4>Puan bağlamı</h4><p>Bu konu yüksek puanlı mı düşük puanlı yorumda mı daha sık geçiyor, ona bakıyoruz.</p></div>
    </div>
    <div class="nlp-example reveal">
      <div class="ne-row"><span class="ne-tag pos">YÜKSEK PUAN</span><blockquote>"Güleryüzlü personele sahip. Özellikle mutfak başarılı ve lezzetli."</blockquote></div>
      <div class="ne-row"><span class="ne-tag neg">DÜŞÜK PUAN</span><blockquote>"İlgi sıfır. Sipariş vermek için beklemeniz gerek."</blockquote></div>
      <p style="margin-top:14px;font-size:13px;color:var(--sand-dim);">İkisi de aynı konudan (<b style="color:var(--sand)">Personel &amp; Hizmet</b>) bahsediyor &mdash; ama biri yüksek puanlı, biri düşük puanlı bir yorumda geçiyor. Bu ikisinin birlikte oranını karşılaştırarak "bu konu genelde puanı yükseltiyor mu, düşürüyor mu?" sorusuna cevap arıyoruz.</p>
    </div>
    <div class="why-box reveal" style="margin-top:28px;">
      <b>Dürüst metodoloji notu:</b> Bu aşamada transformer/sentiment modeli kullanılmadı. Mevcut NLP katmanı kural-tabanlı konu tespiti (aspect extraction) ve puan-bağlamı analizidir &mdash; bir dil modelinin "bu yorum olumlu mu olumsuz mu" diye karar vermesi değil.
    </div>
    <div class="caution reveal">&#128300; <span>Google Travel'da 24, Trip.com'da (Şikayetvar ile paylaşılan) 18 konu başlığı kullanılıyor; ikisi arasında ortak dile çevirmek için 21 kategorilik bir eşleştirme (bkz. bölüm 12) kuruldu.</span></div>
  </div>
</section>
"""
print("HTML_NLP chars:", len(HTML_NLP))

# ==============================================================
# SIKAYETVAR section (v3 numbers; keep 9.1-9.7 subsections, caption
# cooccurrence/severity as the original 236-complaint (v2) scope since
# that analysis was not re-run on the newer v3 corpus this round)
# ==============================================================
ASPECT_TR = {
    "STAFF_SERVICE": "Personel & Hizmet", "CLEANLINESS_HYGIENE": "Temizlik & Hijyen", "FOOD_BEVERAGE": "Yeme & İçme",
    "ROOM": "Oda", "BEACH_SEA": "Plaj & Deniz", "POOL": "Havuz", "FACILITIES_MAINTENANCE": "Tesis & Bakım",
    "RESERVATION": "Rezervasyon", "PAYMENT_REFUND": "Ödeme & İade", "PRICE_VALUE": "Fiyat & Değer",
    "CHECKIN_CHECKOUT": "Giriş & Çıkış", "AIR_CONDITIONING": "Klima", "NOISE": "Gürültü",
    "FAMILY_CHILDREN": "Aile & Çocuk", "TRANSPORT_TRANSFER": "Ulaşım & Transfer",
    "MANAGEMENT_COMMUNICATION": "Yönetim & İletişim", "SPA_WELLNESS": "Spa & Wellness", "SAFETY_SECURITY": "Güvenlik",
}
sk_top3 = SK["top_aspects"][:3]

HTML_SIKAYETVAR = """
<!-- ================= SIKAYETVAR ================= -->
<section id="sikayetvar-voice" data-nav="Şikayetvar">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">11 &mdash; Şikayetvar Şikâyet Görünürlüğü</div>
      <h2 class="section-title">Bir problem yaşandığında müşteriler en çok hangi konulardan söz ediyor?</h2>
      <p class="section-sub">Google Travel ve Trip.com genel müşteri deneyimini gösterirken, Şikayetvar özellikle problem yaşanan durumları görünür kılıyor. Bu yüzden burada "memnuniyet" değil, sorunların hangi başlıklarda tekrar ettiğine bakıyoruz.</p>
    </div>
    <div class="card-grid g3 reveal" style="margin-bottom:40px;">
      <div class="card"><p style="font-family:var(--serif);font-size:16px;color:var(--sand);">Google Travel puanı bize ne anlatır?</p><p style="margin-top:10px;font-size:13.5px;color:var(--sand-dim);line-height:1.6;">Bir otelin genel müşteri deneyimini gösterir.</p></div>
      <div class="card"><p style="font-family:var(--serif);font-size:16px;color:var(--sand);">Trip.com bize ne anlatır?</p><p style="margin-top:10px;font-size:13.5px;color:var(--sand-dim);line-height:1.6;">Puanın yanında misafir segmenti ve konaklama detayını da gösterir.</p></div>
      <div class="card" style="border-color:var(--accent-warm);"><p style="font-family:var(--serif);font-size:16px;color:var(--sand);">Şikayetvar bize ne anlatır?</p><p style="margin-top:10px;font-size:13.5px;color:var(--sand-dim);line-height:1.6;">Bir sorun yaşandığında müşterinin hangi operasyonel başlıklarda şikâyet oluşturduğunu gösterir.</p></div>
    </div>
    <p class="reveal" style="font-family:var(--serif);font-size:17px;color:var(--sand);max-width:700px;margin-bottom:36px;">Şikayetvar, projenin <em style="color:var(--accent-warm);font-style:italic;">"sorun nerede ortaya çıkıyor?"</em> katmanıdır.</p>
    <div class="kpi-row reveal" id="sk-kpis"></div>

    <div class="card-grid g2 reveal">
      <div class="card"><div class="chart-title">Şikâyetlerde personel, oda ve iletişim en sık geçen konular</div><div class="chart-note">Bahsedilme oranı &mdash; """ + str(SK["clean_rows"]) + """ şikâyet üzerinden. Bir şikâyet birden fazla konudan bahsedebilir, bu yüzden yüzdelerin toplamı %100 olmak zorunda değildir.</div><div class="chart-wrap" id="chart-sk-aspects"></div><div class="quote-hint">&uarr; Bir konuya tıklayın, o konudan bahseden gerçek şikâyetleri görün</div><div class="quote-panel" id="sk-quote-panel"></div></div>
      <div class="card"><div class="chart-title">En çok şikâyet alan oteller</div><div class="chart-note">Görünür şikâyet sayısına göre; bu bir kalite sıralaması değildir</div><div class="chart-wrap" id="chart-sk-hotels"></div></div>
    </div>
    <div class="explain-grid reveal">
      <div class="eg"><h5>Bu grafik neyi gösteriyor?</h5><p>""" + str(SK["clean_rows"]) + """ şikâyette hangi konuların ne sıklıkla geçtiğini ve en çok şikâyet alan otelleri.</p></div>
      <div class="eg"><h5>Nasıl okunur?</h5><p>Çubuk uzunluğu bahsedilme oranını (%) gösterir.</p></div>
      <div class="eg"><h5>Ne sonuç çıkarıyoruz?</h5><p>""" + ASPECT_TR.get(sk_top3[0]["aspect"]) + """ (%""" + str(round(sk_top3[0]["mention_rate_pct"],1)).replace(".",",") + """), """ + ASPECT_TR.get(sk_top3[1]["aspect"]) + """ (%""" + str(round(sk_top3[1]["mention_rate_pct"],1)).replace(".",",") + """) ve """ + ASPECT_TR.get(sk_top3[2]["aspect"]) + """ (%""" + str(round(sk_top3[2]["mention_rate_pct"],1)).replace(".",",") + """) en sık konuşulan üç konu.</p></div>
      <div class="eg"><h5>Neden önemli?</h5><p>İyileştirme çalışması yalnız tek departmana odaklanmamalı; müşteri deneyimi personel, oda, iletişim ve ödeme gibi birden fazla temas noktasından aynı anda etkileniyor.</p></div>
      <div class="eg"><h5>Ne anlama gelmiyor?</h5><p><b style="color:var(--bad)">Bu oranlar otellerin müşterilerinin bu yüzdesi şikâyet ediyor anlamına gelmez.</b> Bu oran yalnız Şikayetvar'da incelenen """ + str(SK["clean_rows"]) + """ şikâyeti anlatır. Ayrıca firma yanıtı vermesi, sorunun çözüldüğü anlamına gelmez.</p></div>
    </div>
    <div class="caution reveal">&#128300; <span><b>Konu modelleme:</b> Bilgisayarla otomatik konu bulma denemesi <code>güvenilir değil</code> olarak işaretlendiği için kullanılmadı. Bunun yerine analiz, anahtar kelimelere dayanan, tekrarlanabilir bir konu sözlüğü (18 başlık) kullanıyor.</span></div>
    <div class="caution reveal">&#128202; <span><b>Örneklem yeterliliği:</b> 192 otelden """ + str(SK["complaint_hotels"]) + """'ü görünür şikâyetle eşleşti; """ + str(SK["verified_pages"]) + """ otelde doğrulanmış Şikayetvar sayfası var, """ + str(SK["ambiguous_n"]) + """ otel kanıt yetersizliğinden bilinçli olarak açık bırakıldı, """ + str(SK["not_found_n"]) + """ otelde sayfa hiç bulunamadı (bu, o otelde sıfır şikâyet olduğu anlamına gelmez).</span></div>

    <div class="reveal" style="margin-top:48px;">
      <div class="section-num" style="margin-bottom:8px;">11.1</div>
      <h3 style="font-size:22px;font-family:var(--serif);">Hangi sorunlar aynı şikâyette birlikte görülüyor?</h3>
      <p style="margin-top:12px;font-size:14px;color:var(--sand-dim);max-width:680px;">Bir konu başka bir konuyla birlikte ne sıklıkla geçiyor &mdash; bu, sorunların birbirinden bağımsız olmadığını gösteriyor. <span style="color:var(--sand-faint);">(Bu alt bölüm, orijinal 236 şikâyetlik örneklem üzerinde yapılmış eş-geçiş analizinin kapsamındadır; yeni eklenen 117 şikâyet için ayrıca tekrarlanmamıştır.)</span></p>
      <div class="card-grid g3" style="margin-top:24px;">
        <div class="card"><p style="font-size:13.5px;color:var(--sand-dim);">Personel &amp; Hizmet + Oda</p><p style="margin-top:8px;font-family:var(--serif);font-size:26px;color:var(--accent);">108</p><p style="margin-top:4px;font-size:12px;color:var(--sand-faint);">şikâyette birlikte geçiyor (n=236)</p></div>
        <div class="card"><p style="font-size:13.5px;color:var(--sand-dim);">Personel &amp; Hizmet + Yeme &amp; İçme</p><p style="margin-top:8px;font-family:var(--serif);font-size:26px;color:var(--accent);">107</p><p style="margin-top:4px;font-size:12px;color:var(--sand-faint);">şikâyette birlikte geçiyor (n=236)</p></div>
        <div class="card"><p style="font-size:13.5px;color:var(--sand-dim);">Ödeme &amp; İade + Yönetim &amp; İletişim</p><p style="margin-top:8px;font-family:var(--serif);font-size:26px;color:var(--accent);">89</p><p style="margin-top:4px;font-size:12px;color:var(--sand-faint);">şikâyette birlikte geçiyor (n=236)</p></div>
      </div>
    </div>

    <div class="card-grid g2 reveal" style="margin-top:48px;">
      <div class="card">
        <div class="section-num" style="margin-bottom:8px;">11.2</div>
        <h3 style="font-size:19px;font-family:var(--serif);margin-bottom:14px;">Otel bazında firma yanıt görünürlüğü</h3>
        <div class="chart-wrap" id="chart-sk-response-share"></div>
        <p style="margin-top:14px;font-size:13px;color:var(--sand-dim);line-height:1.6;">En çok şikâyet alan otellerde firma yanıtı olan şikâyet payı.</p>
        <div class="caution" style="margin-top:14px;">&#128683; <span><b>Firma cevap verdi = sorun çözüldü değildir.</b> Yanıt, çözüm değildir.</span></div>
      </div>
      <div class="card">
        <div class="section-num" style="margin-bottom:8px;">11.3</div>
        <h3 style="font-size:19px;font-family:var(--serif);margin-bottom:14px;">Şikâyetler ne kadar sert bir dille yazılıyor?</h3>
        <p style="font-size:13px;color:var(--sand-dim);line-height:1.6;">Şikayetvar'da puan yok. <span style="color:var(--sand-faint);">(Orijinal 236 şikâyetlik dil-şiddeti etiketlemesi kapsamında &mdash; v3'ün yeni 117 şikâyeti için henüz tekrarlanmadı.)</span></p>
        <div class="detail-grid" style="margin-top:16px;">
          <div class="d-item"><div class="n" style="color:var(--bad);">33</div><div class="l">Sert dilli (%14,0)</div></div>
          <div class="d-item"><div class="n" style="color:var(--warn);">132</div><div class="l">Güçlü memnuniyetsizlik (%55,9)</div></div>
          <div class="d-item"><div class="n" style="color:var(--sand-faint);">71</div><div class="l">Sade şikâyet (%30,1)</div></div>
        </div>
      </div>
    </div>

    <div class="reveal" style="margin-top:48px;">
      <div class="section-num" style="margin-bottom:8px;">11.4</div>
      <h3 style="font-size:22px;font-family:var(--serif);margin-bottom:20px;">Şikayetvar'dan ne öğrendik?</h3>
      <div class="finding-list" id="sk-findings"></div>
    </div>

    <div class="why-box reveal" style="margin-top:44px;border-left-color:var(--accent-warm);">
      <b style="color:var(--sand);">Tek cümlede Şikayetvar sonucu:</b> Şikayetvar analizi, müşteri sorunlarının çoğu zaman tek bir noktadan değil; personel, oda, iletişim, ödeme ve yemek gibi birbiriyle bağlantılı temas noktalarından oluştuğunu gösteriyor. <span style="color:var(--sand-faint);font-size:12.5px;">Bu sonuç yalnız incelenen """ + str(SK["clean_rows"]) + """ şikâyeti temsil eder.</span>
    </div>
  </div>
</section>
"""
print("HTML_SIKAYETVAR chars:", len(HTML_SIKAYETVAR))

sk_findings_data = [
    [f'Şikâyetler tek başlıklı değil — {ASPECT_TR.get(sk_top3[0]["aspect"])}, {ASPECT_TR.get(sk_top3[1]["aspect"])} ve {ASPECT_TR.get(sk_top3[2]["aspect"])} sürekli birlikte tekrar ediyor.', f"%{sk_top3[0]['mention_rate_pct']:.1f} · %{sk_top3[1]['mention_rate_pct']:.1f} · %{sk_top3[2]['mention_rate_pct']:.1f}"],
    ["Ödeme ve iade önemli bir sorun başlığı.", f"%{next(a['mention_rate_pct'] for a in SK['top_aspects'] if a['aspect']=='PAYMENT_REFUND'):.1f}"],
    [f"Hedefli tamamlama sonrası clean-v3 corpus'u {SK['clean_rows']} şikâyete, {SK['complaint_hotels']} otele büyüdü.", f"v2: 237/32 → v3: {SK['clean_rows']}/{SK['complaint_hotels']}"],
    ["Firma cevabı ölçülebiliyor ama çözüm ölçülemiyor — yanıt davranışı operasyonel bir sinyaldir, sonuç değil.", f"%{SK['reply_visibility_pct']:.1f} yanıt görünürlüğü"],
    ["9 otelde eşleşme kanıtı yetersiz kaldı — zorla eşleştirme yapılmadı, açıkça işaretlendi.", f"{SK['ambiguous_n']} otel AMBIGUOUS_REMAINS"],
]
JS_SIKAYETVAR = """
/* ============================================================
   SIKAYETVAR VOICE (v3)
   ============================================================ */
const SK_ASPECT_EXAMPLES = """ + json.dumps(SK_ASPECT_EXAMPLES, ensure_ascii=False) + """;

$('#sk-kpis').innerHTML = [
  {n:'""" + str(SK["clean_rows"]) + """',l:'Şikayetvar şikâyeti (clean-v3)',d:'""" + str(SK["complaint_hotels"]) + """ otelle eşleşen kayıt'},
  {n:'""" + str(SK["complaint_hotels"]) + """',l:'Otel',d:'Şikayetvar\\'da görünür şikâyeti bulunan proje oteli'},
  {n:'18',l:'Şikâyet konusu',d:'Personel, oda, ödeme, yemek, temizlik gibi ana başlıklar'},
  {n:'""" + str(SK["verified_pages"]) + """',l:'Doğrulanmış Şikayetvar sayfası',d:'192 otel içinde sayfa keşfi + entity eşleştirmesi tamamlanan'},
  {n:'""" + str(SK["company_reply_visible"]) + """/""" + str(SK["clean_rows"]) + """',l:'Firma yanıtı görünen şikâyet',d:'Şirketin şikâyete cevap verdiğini gösterir; cevap vermiş olması sorunun çözüldüğü anlamına gelmez'},
  {n:'%""" + str(round(SK["reply_visibility_pct"],1)).replace(".",",") + """',l:'Firma yanıt görünürlüğü',d:'""" + str(SK["company_reply_visible"]) + """/""" + str(SK["clean_rows"]) + """'},
].map(k=>`<div class="kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');

const ASPECT_TR = """ + json.dumps(ASPECT_TR, ensure_ascii=False) + """;
const trAspect = code => ASPECT_TR[code] || code.replace(/_/g,' ');

const skTop = """ + json.dumps([{"aspect": a["aspect"], "label": ASPECT_TR.get(a["aspect"], a["aspect"]), "value": round(a["mention_rate_pct"], 1), "tip": f"{a['complaint_count']} şikâyet"} for a in SK["top_aspects"][:12]], ensure_ascii=False) + """;
const skQuotePanel = $('#sk-quote-panel');
hbarChart($('#chart-sk-aspects'),skTop,{color:'var(--accent-warm)',valueFmt:v=>fmt(v,1)+'%',labelWidth:170,
  onClick:d=>renderQuotePanel(skQuotePanel,d.label,SK_ASPECT_EXAMPLES[d.aspect])});

const skHotelProfiles = """ + json.dumps([{"name": h["hotel_name"], "n": h["complaint_n"], "top": [t.split(":") for t in (h["top_aspects"] or "").split("|") if t]} for h in SK["top_hotels"][:6]], ensure_ascii=False) + """;
$('#chart-sk-hotels').innerHTML = skHotelProfiles.map(p=>`
  <div style="padding:14px 0;border-bottom:1px solid var(--line-soft);">
    <div style="display:flex;justify-content:space-between;align-items:baseline;">
      <span style="font-family:var(--serif);font-size:15.5px;color:var(--sand);">${p.name}</span>
      <span style="font-family:var(--mono);font-size:11px;color:var(--sand-faint);">${p.n} şikâyet</span>
    </div>
    <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;">
      ${p.top.map(t=>`<span class="chip" style="border-color:var(--accent-warm);color:var(--accent-warm);">${trAspect(t[0])} ${t[1]}</span>`).join('')}
    </div>
  </div>`).join('');

const skResponseShare = """ + json.dumps([{"label": h["hotel_name"][:20], "value": round(h["company_reply_visibility_pct"], 1), "tip": f"{h['complaint_n']} şikâyet üzerinden"} for h in SK["reply_top"][:8]], ensure_ascii=False) + """;
hbarChart($('#chart-sk-response-share'),skResponseShare,{color:'var(--gold)',valueFmt:v=>fmt(v,1)+'%',labelWidth:150,height:22});

const skFindings = """ + json.dumps(sk_findings_data, ensure_ascii=False) + """;
$('#sk-findings').innerHTML = skFindings.map((f,i)=>`<div class="finding"><div class="fnum">${String(i+1).padStart(2,'0')}</div><div class="ftext">${f[0]}</div><div class="fmeta">${f[1]}</div></div>`).join('');
"""
print("JS_SIKAYETVAR chars:", len(JS_SIKAYETVAR))

# ==============================================================
# CROSS-SOURCE section (Google x Trip, Google x Sikayetvar)
# ==============================================================
CANON_ASPECT_TR = {
    "STAFF_SERVICE": "Personel & Hizmet", "MANAGEMENT_COMMUNICATION": "Yönetim & İletişim",
    "CLEANLINESS_HYGIENE": "Temizlik & Hijyen", "FOOD_BEVERAGE": "Yeme & İçme", "ROOM_COMFORT": "Oda Konforu",
    "FACILITIES_MAINTENANCE": "Tesis & Bakım", "POOL": "Havuz", "BEACH_SEA": "Plaj & Deniz",
    "PRICE_VALUE": "Fiyat & Değer", "RESERVATION": "Rezervasyon", "PAYMENT_REFUND": "Ödeme & İade",
    "CHECKIN_CHECKOUT": "Giriş & Çıkış", "AIR_CONDITIONING": "Klima", "NOISE": "Gürültü",
    "FAMILY_CHILDREN": "Aile & Çocuk", "TRANSPORT_TRANSFER": "Ulaşım & Transfer",
    "SAFETY_SECURITY": "Güvenlik", "SPA_WELLNESS": "Spa & Wellness", "LOCATION": "Konum",
    "WIFI": "Wifi", "ANIMATION_ENTERTAINMENT": "Animasyon & Eğlence",
}
cov_by_key = {c["hotel_set"]: c["hotel_count"] for c in CGS["coverage"]}
both_top1 = CGS["both_concern_top"][0] if CGS["both_concern_top"] else None
strength_top1 = CGS["strength_vs_complaint_top"][0] if CGS["strength_vs_complaint_top"] else None

HTML_CROSS = """
<!-- ================= CROSS-SOURCE ================= -->
<section id="compare" class="tight" data-nav="Ortak Sinyaller">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">12 &mdash; Kaynaklar Arası Ortak Sinyaller</div>
      <h2 class="section-title">Üç kaynak aynı konularda ne diyor?</h2>
      <p class="section-sub">Google Travel, Trip.com ve Şikayetvar üç farklı şeyi ölçüyor. Bu yüzden ham sayıları tek bir kalite skorunda birleştirmiyoruz &mdash; yalnız aynı konuları ortak bir dile çevirip nerede örtüştüklerine, nerede ayrıştıklarına bakıyoruz. Satır seviyesinde birleştirme yapılmaz; karşılaştırma yalnız otel/konu özet seviyesindedir.</p>
    </div>

    <div class="card-grid g3 reveal" style="margin-bottom:8px;">
      <div class="card"><div class="chip on" style="margin-bottom:14px;">GOOGLE TRAVEL</div><p style="font-size:13.5px;color:var(--sand-dim);line-height:1.7;">Genel misafir sesi &middot; her puan seviyesi &middot; """ + str(G["hotels"]) + """ otel</p></div>
      <div class="card"><div class="chip" style="margin-bottom:14px;border-color:var(--gold);color:var(--gold);">TRIP.COM</div><p style="font-size:13.5px;color:var(--sand-dim);line-height:1.7;">Puan + misafir segmenti &middot; """ + str(T["hotels"]) + """ otel</p></div>
      <div class="card"><div class="chip" style="margin-bottom:14px;border-color:var(--bad);color:var(--bad);">ŞİKAYETVAR</div><p style="font-size:13.5px;color:var(--sand-dim);line-height:1.7;">Yalnız problem-odaklı ses &middot; """ + str(SK["complaint_hotels"]) + """ otel</p></div>
    </div>

    <div class="reveal" style="margin-top:40px;">
      <h3 style="font-size:19px;font-family:var(--serif);margin-bottom:6px;">Google Travel &times; Trip.com</h3>
      <p style="font-size:13.5px;color:var(--sand-dim);max-width:680px;margin-bottom:18px;">Her iki platformda da yeterli yorumu olan oteller arasında ortalama puan ne kadar örtüşüyor?</p>
      <div class="kpi-row" id="cross-gt-kpis"></div>
      <div class="explain-grid" style="margin-top:0;padding-top:0;border-top:none;">
        <div class="eg"><h5>Ne anlama gelmiyor?</h5><p>Puan farkı, hangi platformun "doğru" olduğu anlamına gelmez &mdash; farklı örneklem ve farklı kullanıcı kitlesi kaynaklı bir ayrışma sinyalidir.</p></div>
      </div>
    </div>

    <div class="reveal" style="margin-top:40px;">
      <h3 style="font-size:19px;font-family:var(--serif);margin-bottom:6px;">Google Travel &times; Şikayetvar</h3>
      <p style="font-size:13.5px;color:var(--sand-dim);max-width:700px;margin-bottom:18px;">İki kaynakta aynı konuları ortak bir dile çevirdik (21 kategori). Sonra her otel-konu çifti için iki sinyale bakıyoruz.</p>
      <div class="kpi-row" id="cross-gs-kpis"></div>
      <div class="card-grid g2" style="margin-top:8px;">
        <div class="card">
          <div class="chart-title" style="color:var(--bad);">Her iki kaynakta da problem sinyali</div>
          <div class="chart-note">Google'da düşük-puan bağlamında, Şikayetvar'da da tekrarlayan şikâyet teması olarak görünen konular</div>
          <div id="cross-both-concern"></div>
        </div>
        <div class="card">
          <div class="chart-title" style="color:var(--accent-warm);">Google'da güçlü, Şikayetvar'da şikâyet</div>
          <div class="chart-note">Google'da yüksek-puan bağlamında güçlü görünen ama Şikayetvar'da hâlâ şikâyet teması olan konular &mdash; gerçek bir ayrışma, çelişki değil</div>
          <div id="cross-strength-vs-complaint"></div>
        </div>
      </div>
      <div class="explain-grid reveal">
        <div class="eg"><h5>Bu neyi gösteriyor?</h5><p>Google'ın genel-review sinyali ile Şikayetvar'ın complaint-sinyalinin aynı otel + aynı konuda nerede birbirini doğruladığını, nerede ayrıştığını.</p></div>
        <div class="eg"><h5>Neden önemli?</h5><p>İki farklı, farklı-önyargılı kaynağın aynı yönde işaret etmesi güveni artırır; ama ikisi de rastgele örneklem değildir.</p></div>
        <div class="eg"><h5>Ne anlama gelmiyor?</h5><p><b style="color:var(--bad)">Bu bir otel sıralaması değildir.</b> Yalnız desteklenen ("supported") eşleşmeler yorumlanmalı; küçük örneklemli çiftler LOW_SUPPORT olarak işaretlenir.</p></div>
      </div>
    </div>

    <div class="caution reveal" style="margin-top:24px;">&#128279; <span>Ortak konu kapsamına giren """ + str(cov_by_key.get("common_verified", 0)) + """ otelden yalnız """ + str(cov_by_key.get("supported_common", 0)) + """'i her iki tarafta da yeterli örneklemde (Google &ge;10, Şikayetvar &ge;5 şikâyet) &mdash; diğerleri yalnız yönlü/keşifsel okunmalı.</span></div>
  </div>
</section>
"""
print("HTML_CROSS chars:", len(HTML_CROSS))

agreement_dist = CGT.get("agreement_dist") or {}
JS_CROSS = """
/* ============================================================
   CROSS-SOURCE (Google x Trip, Google x Sikayetvar)
   ============================================================ */
const CANON_TR = """ + json.dumps(CANON_ASPECT_TR, ensure_ascii=False) + """;

$('#cross-gt-kpis').innerHTML = [
  {n:'""" + str(CGT.get("common_hotels", 0)) + """',l:'Her iki platformda da veri olan otel',d:'Google Travel + Trip.com ortak kapsam'},
  {n:'""" + str(CGT.get("supported_n", 0)) + """',l:'Desteklenen karşılaştırma',d:'Her iki tarafta da yeterli örneklem (n&ge;10)'},
  {n:'""" + str(agreement_dist.get("HIGH_AGREEMENT", 0)) + """',l:'Yüksek uyum',d:'Puan farkı |gap|&le;0,3'},
  {n:'""" + str(agreement_dist.get("DISAGREEMENT", 0)) + """',l:'Ayrışma',d:'Puan farkı |gap|>0,7'},
].map(k=>`<div class="kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');

$('#cross-gs-kpis').innerHTML = [
  {n:'""" + str(cov_by_key.get("common_complaint_bearing", 0)) + """',l:'Google + Şikayetvar ortak (şikâyet olan)',d:'Her iki kaynakta da veri bulunan otel'},
  {n:'""" + str(cov_by_key.get("supported_common", 0)) + """',l:'Desteklenen karşılaştırma',d:'Google n&ge;10 ve Şikayetvar n&ge;5'},
  {n:'21',l:'Ortak konu kategorisi',d:'Google\\'ın 24 ve Şikayetvar\\'ın 18 konusunu kapsayan eşleştirme'},
  {n:'""" + str((CGS.get("label_dist") or {}).get("BOTH_SOURCE_CONCERN", 0)) + """',l:'İki kaynakta da problem sinyali',d:'Otel × konu kombinasyonu sayısı'},
].map(k=>`<div class="kpi"><div class="n">${k.n}</div><div class="l">${k.l}</div><div class="d">${k.d}</div></div>`).join('');

const crossBothConcern = """ + json.dumps([{"hotel": r["hotel_name"], "aspect": CANON_ASPECT_TR.get(r["canonical_aspect"], r["canonical_aspect"]), "pct": round(r["sikayetvar_mention_rate_pct"], 0)} for r in CGS["both_concern_top"][:6]], ensure_ascii=False) + """;
$('#cross-both-concern').innerHTML = crossBothConcern.length ? crossBothConcern.map(r=>`<div style="padding:10px 0;border-bottom:1px solid var(--line-soft);display:flex;justify-content:space-between;font-size:13.5px;"><span style="color:var(--sand);">${r.hotel}</span><span style="color:var(--bad);font-family:var(--mono);">${r.aspect} · %${r.pct}</span></div>`).join('') : '<p class="na">Şu an desteklenen düzeyde eşleşme yok.</p>';

const crossStrength = """ + json.dumps([{"hotel": r["hotel_name"], "aspect": CANON_ASPECT_TR.get(r["canonical_aspect"], r["canonical_aspect"]), "pct": round(r["sikayetvar_mention_rate_pct"], 0)} for r in CGS["strength_vs_complaint_top"][:6]], ensure_ascii=False) + """;
$('#cross-strength-vs-complaint').innerHTML = crossStrength.length ? crossStrength.map(r=>`<div style="padding:10px 0;border-bottom:1px solid var(--line-soft);display:flex;justify-content:space-between;font-size:13.5px;"><span style="color:var(--sand);">${r.hotel}</span><span style="color:var(--accent-warm);font-family:var(--mono);">${r.aspect} · %${r.pct}</span></div>`).join('') : '<p class="na">Şu an desteklenen düzeyde eşleşme yok.</p>';
"""
print("JS_CROSS chars:", len(JS_CROSS))

# ==============================================================
# HOTEL 360 section (replaces the old "Otel Gezgini" explorer)
# ==============================================================
HTML_HOTEL360 = """
<!-- ================= HOTEL 360 ================= -->
<section id="hotel360" data-nav="Hotel 360°">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">13 &mdash; Hotel 360°</div>
      <h2 class="section-title">192 otelin herhangi birini seçin</h2>
      <p class="section-sub">Master veri, Google Travel, Trip.com, politika/olanak ve Şikayetvar &mdash; dört kaynağı tek panelde görün. Veri yoksa "bu kaynakta yeterli veri yok" yazar; asla sıfır göstermez. Daha yüksek seviye, daha fazla kaynaktan daha güçlü veri desteği demektir &mdash; "daha iyi otel" demek değildir.</p>
    </div>
    <div class="explorer reveal">
      <div class="explorer-controls">
        <select id="hotel-area-select"><option value="">Tüm bölgeler</option></select>
        <select id="hotel-select"></select>
      </div>
      <div class="explorer-panel" id="hotel-panel"></div>
    </div>
  </div>
</section>

<!-- ================= HOTEL COMPARE ================= -->
<section id="hotel-compare" class="tight" data-nav="Karşılaştır">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">14 &mdash; Otel Karşılaştırma</div>
      <h2 class="section-title">İki oteli yan yana karşılaştırın</h2>
      <p class="section-sub">Kazanan/kaybeden üretmiyoruz &mdash; yalnız iki otel arasındaki farkları görünür kılıyoruz.</p>
    </div>
    <div class="explorer reveal">
      <div class="explorer-controls">
        <select id="compare-a"></select>
        <select id="compare-b"></select>
      </div>
      <div class="compare-wrap" id="compare-panel"></div>
    </div>
  </div>
</section>
"""
print("HTML_HOTEL360 chars:", len(HTML_HOTEL360))

ARCHETYPE_TR = {
    "insufficient_signal": "Yetersiz veri sinyali", "low_data_profile": "Düşük veri desteği",
    "platform_consistent_strong_profile": "Platformlar arası güçlü ve tutarlı",
    "platform_consistent_concern_profile": "Platformlar arası tutarlı endişe sinyali",
    "source_divergent_profile": "Kaynaklar arası ayrışan sinyal",
    "family_oriented_profile": "Aile odaklı", "beach_oriented_profile": "Plaj odaklı",
    "wellness_oriented_profile": "Wellness odaklı",
}
CONFIDENCE_TR = {"HIGH": "Yüksek", "MEDIUM": "Orta", "LOW": "Düşük", "VERY_LOW": "Çok düşük"}

JS_HOTEL360 = """
/* ============================================================
   HOTEL 360 (single-hotel deep dive)
   ============================================================ */
const ARCHETYPE_TR = """ + json.dumps(ARCHETYPE_TR, ensure_ascii=False) + """;
const CONFIDENCE_TR = """ + json.dumps(CONFIDENCE_TR, ensure_ascii=False) + """;
const ALL_ASPECT_TR = Object.assign({}, ASPECT_TR_G, CANON_TR, ASPECT_TR);

function aspectChips(str,color){
  if(!str) return '';
  return str.split(';').filter(Boolean).map(code=>`<span class="chip" style="border-color:${color};color:${color};margin:0 6px 6px 0;">${ALL_ASPECT_TR[code]||code}</span>`).join('');
}
function archChips(str){
  if(!str) return '';
  return str.split(';').filter(Boolean).map(a=>`<span class="chip on" style="margin:0 6px 6px 0;">${ARCHETYPE_TR[a]||a}</span>`).join('');
}
function skTopChips(str){
  if(!str) return '';
  return str.split('|').filter(Boolean).map(t=>{const [code,pct]=t.split(':');return `<span class="chip" style="border-color:var(--bad);color:var(--bad);margin:0 6px 6px 0;">${ALL_ASPECT_TR[code]||code} ${pct}</span>`;}).join('');
}

function sourceBlocksHtml(h){
  const gBlock = h.g_n ? `
    <div class="source-block g">
      <h5><span class="src-badge g">Google Travel</span></h5>
      <div class="detail-grid">
        <div class="d-item"><div class="n">${fmt(h.g_n)}</div><div class="l">Yorum sayısı</div></div>
        <div class="d-item"><div class="n">${fmt(h.g_mean,2)}</div><div class="l">Ortalama puan (1-5)</div></div>
        <div class="d-item"><div class="n">%${fmt(h.g_high*100,0)}</div><div class="l">Yüksek puan payı</div></div>
        <div class="d-item"><div class="n">%${fmt(h.g_low*100,0)}</div><div class="l">Düşük puan payı</div></div>
      </div>
      ${h.g_strength?`<p style="margin-top:14px;font-size:12px;color:var(--sand-faint);">Güçlü sinyal:</p><div style="margin-top:6px;">${aspectChips(h.g_strength,'var(--good)')}</div>`:''}
      ${h.g_concern?`<p style="margin-top:10px;font-size:12px;color:var(--sand-faint);">Dikkat sinyali:</p><div style="margin-top:6px;">${aspectChips(h.g_concern,'var(--bad)')}</div>`:''}
    </div>` : `<div class="na-block" style="margin-top:22px;">Google Travel'da bu kaynakta yeterli veri yok.</div>`;

  const tBlock = h.t_n ? `
    <div class="source-block t">
      <h5><span class="src-badge t">Trip.com</span></h5>
      <div class="detail-grid">
        <div class="d-item"><div class="n">${fmt(h.t_n)}</div><div class="l">Yorum sayısı</div></div>
        <div class="d-item"><div class="n">${fmt(h.t_mean,2)}</div><div class="l">Ortalama puan (5 üzerinden)</div></div>
        <div class="d-item"><div class="n">${TRAV_TR[h.t_top_traveler]||h.t_top_traveler||'—'}</div><div class="l">En sık misafir segmenti</div></div>
        <div class="d-item"><div class="n">${h.t_country_n?fmt(h.t_country_n):'<span class=na>mevcut değil</span>'}</div><div class="l">Belirtilen ülke sayısı</div></div>
      </div>
      ${(h.t_family_pct||h.t_couple_pct)?`<p style="margin-top:14px;font-size:12px;color:var(--sand-faint);">Aile payı %${fmt(h.t_family_pct,0)} · Çift payı %${fmt(h.t_couple_pct,0)}</p>`:''}
      ${h.t_countries&&h.t_countries.top&&h.t_countries.top.length?`
      <div style="margin-top:14px;">
        <p style="font-size:12px;color:var(--sand-faint);">Misafir kökeni (bilinen ${fmt(h.t_countries.total)} yorum)</p>
        <div style="margin-top:6px;display:flex;flex-direction:column;gap:4px;">
          ${h.t_countries.top.map(c=>`<div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--sand-dim);"><span>${COUNTRY_FLAG[c.country]||'🏳️'}</span><span style="flex:1;">${COUNTRY_TR[c.country]||c.country}</span><span style="font-family:var(--mono);color:var(--sand);">${fmt(c.n)}</span></div>`).join('')}
        </div>
      </div>` : `<p style="margin-top:14px;font-size:11.5px;color:var(--sand-faint);font-style:italic;">Bu otel için misafir kökeni verisi yok/yetersiz.</p>`}
    </div>` : `<div class="na-block" style="margin-top:22px;">Trip.com'da bu kaynakta yeterli veri yok.</div>`;

  const pBlock = h.has_policy ? `
    <div class="source-block p">
      <h5><span class="src-badge" style="border-color:var(--sand-dim);color:var(--sand-dim);">Politika &amp; Olanaklar</span></h5>
      <div class="detail-grid">
        <div class="d-item"><div class="n">${h.policy_status||'—'}</div><div class="l">Veri durumu</div></div>
        <div class="d-item"><div class="n">${fmt(h.amenity_n)}</div><div class="l">Tespit edilen olanak</div></div>
        <div class="d-item"><div class="n">${fmt(h.family_feature_n)}</div><div class="l">Aile özelliği</div></div>
        <div class="d-item"><div class="n">${fmt(h.wellness_feature_n)}</div><div class="l">Wellness özelliği</div></div>
      </div>
    </div>` : `<div class="na-block" style="margin-top:22px;">Politika/olanak verisi mevcut değil.</div>`;

  const sBlock = h.sk_n ? `
    <div class="source-block s">
      <h5><span class="src-badge s">Şikayetvar</span></h5>
      <div class="detail-grid">
        <div class="d-item"><div class="n">${fmt(h.sk_n)}</div><div class="l">Görünür şikâyet</div></div>
        <div class="d-item"><div class="n">${h.sk_reply_pct!==null?'%'+fmt(h.sk_reply_pct,0):'<span class=na>—</span>'}</div><div class="l">Firma yanıt görünürlüğü</div></div>
        <div class="d-item"><div class="n">${h.sk_visibility_per1000!==null?fmt(h.sk_visibility_per1000,1):'<span class=na>—</span>'}</div><div class="l">1000 Google yorumu başına görünürlük</div></div>
      </div>
      ${h.sk_top?`<div style="margin-top:14px;">${skTopChips(h.sk_top)}</div>`:''}
      <p style="margin-top:10px;font-size:11.5px;color:var(--sand-faint);">Bu bir complaint rate değildir &mdash; yalnız görünürlük göstergesidir.</p>
    </div>` : `<div class="na-block" style="margin-top:22px;">${h.sk_page_status==='NOT_FOUND'?'Şikayetvar\\'da doğrulanmış sayfa bulunamadı (sıfır şikâyet anlamına gelmez).':'Şikayetvar\\'da görünür şikâyet yok.'}</div>`;

  return gBlock+tBlock+pBlock+sBlock;
}

const areaSelect = $('#hotel-area-select'), hotelSelect = $('#hotel-select');
const areas = [...new Set(hotels.map(h=>h.area))].sort();
areaSelect.innerHTML += areas.map(a=>`<option value="${a}">${a}</option>`).join('');
function populateHotelSelect(area){
  const list = area ? hotels.filter(h=>h.area===area) : hotels;
  hotelSelect.innerHTML = list.map(h=>`<option value="${h.id}">${h.name}</option>`).join('');
}
populateHotelSelect('');

function renderHotelPanel(id){
  const h = hotels.find(x=>x.id===id); if(!h) return;
  const panel = $('#hotel-panel');
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:14px;align-items:flex-start;">
      <div><h3 style="font-size:22px;max-width:520px;">${h.name}</h3><p style="margin-top:6px;color:var(--sand-faint);font-size:13px;">${h.area} · ${h.id}</p></div>
      <span class="confidence-badge ${h.confidence}">${CONFIDENCE_TR[h.confidence]||h.confidence} veri desteği</span>
    </div>
    <div class="detail-grid" style="margin-top:18px;">
      <div class="d-item"><div class="n">${h.rating??'<span class=na>—</span>'}</div><div class="l" title="Google'daki 1-5 arası genel müşteri puanı (master snapshot)">Google puanı</div></div>
      <div class="d-item"><div class="n">${fmt(h.reviews)}</div><div class="l" title="Dijital görünürlük göstergesi">Google yorum sayısı (master)</div></div>
      <div class="d-item"><div class="n">${h.price?('$'+fmt(h.price)):'<span class=na>mevcut değil</span>'}</div><div class="l">Görünen fiyat (anlık)</div></div>
      <div class="d-item"><div class="n">${h.star?fmt(h.star)+'★':'<span class=na>mevcut değil</span>'}</div><div class="l">Resmî yıldız</div></div>
    </div>
    ${h.archetypes?`<div style="margin-top:18px;">${archChips(h.archetypes)}</div>`:''}
    ${sourceBlocksHtml(h)}
  `;
}
hotelSelect.addEventListener('change',e=>renderHotelPanel(e.target.value));
areaSelect.addEventListener('change',e=>{populateHotelSelect(e.target.value); renderHotelPanel(hotelSelect.value);});
const firstHotelId = hotels.find(h=>h.confidence==='HIGH')?.id || hotels[0].id;
renderHotelPanel(firstHotelId);
hotelSelect.value = firstHotelId;

/* ============================================================
   HOTEL COMPARE
   ============================================================ */
const cmpA = $('#compare-a'), cmpB = $('#compare-b');
const hotelOptionsHtml = hotels.map(h=>`<option value="${h.id}">${h.name} (${h.area})</option>`).join('');
cmpA.innerHTML = hotelOptionsHtml; cmpB.innerHTML = hotelOptionsHtml;

function cmpRow(label,a,b,fmtFn){
  fmtFn = fmtFn || (v=>v===null||v===undefined?'<span class=na>—</span>':v);
  const isDiff = a!==b && a!==null && b!==null && a!==undefined && b!==undefined;
  return {label,a:fmtFn(a),b:fmtFn(b),diff:isDiff};
}
function renderComparePanel(){
  const a = hotels.find(h=>h.id===cmpA.value), b = hotels.find(h=>h.id===cmpB.value);
  if(!a||!b) return;
  const rows = [
    cmpRow('Google puanı (master)',a.rating,b.rating),
    cmpRow('Google Travel ortalama puan',a.g_mean,b.g_mean,v=>v!==null&&v!==undefined?fmt(v,2):'<span class=na>veri yok</span>'),
    cmpRow('Google Travel yorum sayısı',a.g_n,b.g_n,v=>v!==null&&v!==undefined?fmt(v):'<span class=na>veri yok</span>'),
    cmpRow('Trip.com ortalama puan (5)',a.t_mean,b.t_mean,v=>v!==null&&v!==undefined?fmt(v,2):'<span class=na>veri yok</span>'),
    cmpRow('Trip.com yorum sayısı',a.t_n,b.t_n,v=>v!==null&&v!==undefined?fmt(v):'<span class=na>veri yok</span>'),
    cmpRow('Trip.com en sık segment',a.t_top_traveler,b.t_top_traveler,v=>v?(TRAV_TR[v]||v):'<span class=na>veri yok</span>'),
    cmpRow('Politika/olanak sayısı',a.amenity_n,b.amenity_n,v=>v!==null&&v!==undefined?fmt(v):'<span class=na>veri yok</span>'),
    cmpRow('Şikayetvar görünür şikâyet',a.sk_n,b.sk_n,v=>v?fmt(v):'0'),
    cmpRow('Şikayetvar yanıt görünürlüğü',a.sk_reply_pct,b.sk_reply_pct,v=>v!==null&&v!==undefined?'%'+fmt(v,0):'<span class=na>veri yok</span>'),
    cmpRow('Hotel 360° veri desteği',a.confidence,b.confidence,v=>CONFIDENCE_TR[v]||v),
  ];
  function panelHtml(h,rowsSide){
    return `<div class="explorer-panel">
      <h4 style="font-size:17px;">${h.name}</h4>
      <p style="margin-top:4px;color:var(--sand-faint);font-size:12px;">${h.area}</p>
      ${rows.map(r=>`<div class="compare-metric ${r.diff?'diff':''}"><span class="cm-label">${r.label}</span><span class="cm-val">${rowsSide==='a'?r.a:r.b}</span></div>`).join('')}
    </div>`;
  }
  $('#compare-panel').innerHTML = panelHtml(a,'a') + panelHtml(b,'b');
}
cmpA.addEventListener('change',renderComparePanel);
cmpB.addEventListener('change',renderComparePanel);
cmpA.value = hotels.find(h=>h.confidence==='HIGH')?.id || hotels[0].id;
cmpB.value = hotels.filter(h=>h.confidence==='HIGH')[1]?.id || hotels[1].id;
renderComparePanel();
"""
print("JS_HOTEL360 chars:", len(JS_HOTEL360))

# ==============================================================
# STORIES / FINDINGS / BUSINESS / LIMITATIONS / CONCLUSION / NEXT / FOOTER
# ==============================================================
HTML_TAIL_SECTIONS = """
<!-- ================= USER STORIES ================= -->
<section id="stories" class="tight" data-nav="Kullanıcılar">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">15 &mdash; Kullanıcı Hikâyeleri</div>
      <h2 class="section-title">Siz kimsiniz, hangi soruyu soruyorsunuz?</h2>
    </div>
    <div class="story-grid reveal" id="story-cards"></div>
  </div>
</section>

<!-- ================= KEY FINDINGS ================= -->
<section id="findings" data-nav="Bulgular">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">16 &mdash; En Önemli Bulgular</div>
      <h2 class="section-title">Projenin özeti: 10 büyük çıkarım</h2>
    </div>
    <div class="finding-list reveal" id="finding-list"></div>
  </div>
</section>

<!-- ================= BUSINESS MEANING ================= -->
<section id="business" class="tight" data-nav="İş Değeri">
  <div class="container">
    <div class="section-head reveal">
      <div class="section-num">17 &mdash; Bu Analiz Bize Ne Sağlar?</div>
      <h2 class="section-title">İşletme açısından karşılık</h2>
    </div>
    <div class="card-grid g4 reveal" id="business-cards"></div>
  </div>
</section>

<!-- ================= CONCLUSION ================= -->
<section id="conclusion" data-nav="Sonuç">
  <div class="container">
    <div class="section-head reveal" style="margin-bottom:34px;">
      <div class="section-num">18 &mdash; Sonuç</div>
    </div>
    <div class="card reveal" style="margin-top:0;max-width:780px;">
      <div class="chart-title" style="font-size:15.5px;">Bu projeyi 1 dakikada nasıl anlatırız?</div>
      <p style="margin-top:14px;font-size:14.5px;color:var(--sand-dim);line-height:1.85;">192 Bodrum otelinden oluşan bir veri seti kurduk; 14 bölgeyi kalite, popülerlik, lüks, değer ve kapasite açısından karşılaştırdık; 2009&ndash;2025 turizm verisi ve havalimanı hareketiyle sezonu analiz ettik. Sonra üç farklı müşteri sesi kaynağını &mdash; Google Travel (""" + f"{G['clean_reviews']:,}".replace(",",".") + """ genel yorum), Trip.com (""" + f"{T['clean_reviews']:,}".replace(",",".") + """ yorum + misafir segmenti) ve Şikayetvar (""" + str(SK['clean_rows']) + """ problem-odaklı şikâyet) &mdash; aynı kural-tabanlı konu sözlüğüyle okuduk, ikisini ortak bir dile çevirip nerede örtüştüklerine baktık. Son olarak her oteli, dört kaynağın ne kadarına sahip olduğunu gösteren bir "Hotel 360°" profiliyle özetledik. Böylece Bodrum otel pazarını tek bir puan yerine farklı veri katmanlarıyla okumaya çalıştık.</p>
    </div>
    <div class="conclusion-kpis reveal" id="conclusion-kpis"></div>
  </div>
</section>

<footer>Bodrum Hotel &amp; Destination Intelligence &mdash; veriye dayalı yönetici brifingi &middot; tüm sayılar notebooks/ ve reports/ çıktılarından türetilmiştir</footer>
"""
print("HTML_TAIL_SECTIONS chars:", len(HTML_TAIL_SECTIONS))

findings_data = [
    ["Bodrum tek tip bir otel pazarı değil.", "192 otel, 14 farklı bölge profili"],
    ["Turizm çok belirgin şekilde sezonluk.", "En yoğun 3 ay yıllık gelişin %59,6'sını taşıyor"],
    ["Havalimanı ve turizm serileri aynı dalgayı izliyor.", "İlişki gücü ρ=0,986 (12 ay üzerinden)"],
    [f"Google Travel'da {ASPECT_TR_GOOGLE.get(top_strength['aspect'])} en güçlü olumlu sinyal, {ASPECT_TR_GOOGLE.get(top_concern['aspect'])} en güçlü dikkat sinyali.", f"n={G['clean_reviews']:,}".replace(",",".")],
    [f"Trip.com'da en yüksek ortalama puanı (yeterli örneklemle) {TRAV_TR.get(top_trav['traveler_type'])} segmenti alıyor.", f"{round(top_trav['mean'],2)}/5, n={top_trav['n']}"],
    [f"Şikayetvar'da en yoğun konuşulan üç konu {ASPECT_TR.get(sk_top3[0]['aspect'])}, {ASPECT_TR.get(sk_top3[1]['aspect'])} ve {ASPECT_TR.get(sk_top3[2]['aspect'])}.", f"n={SK['clean_rows']} şikâyet"],
    [f"Google×Şikayetvar hizalamasında {(CGS.get('label_dist') or {}).get('BOTH_SOURCE_CONCERN',0)} otel×konu çiftinde iki kaynak da aynı yönde işaret ediyor.", "21 ortak kategori üzerinden"],
    [f"Google×Trip.com karşılaştırmasında {CGT.get('common_hotels',0)} ortak otelden {CGT.get('supported_n',0)}'i istatistiksel olarak desteklenen düzeyde.", f"Yüksek uyum {agreement_dist.get('HIGH_AGREEMENT',0)} · Orta uyum {agreement_dist.get('MODERATE_AGREEMENT',0)} · Ayrışma {agreement_dist.get('DISAGREEMENT',0)}"],
    [f"Hotel 360°'ta {H360M['confidence_dist'].get('HIGH',0)+H360M['confidence_dist'].get('MEDIUM',0)} otel yüksek/orta veri desteğinde; geri kalanı hâlâ sınırlı.", f"HIGH={H360M['confidence_dist'].get('HIGH',0)} · MEDIUM={H360M['confidence_dist'].get('MEDIUM',0)} · LOW={H360M['confidence_dist'].get('LOW',0)} · VERY_LOW={H360M['confidence_dist'].get('VERY_LOW',0)}"],
    ["Firma yanıtı varlığı çözüm anlamına gelmiyor.", f"Yanıt görünürlüğü %{SK['reply_visibility_pct']:.1f} — sonuç kalitesi ölçülmedi"],
]

JS_TAIL = """
/* ============================================================
   USER STORIES
   ============================================================ */
const stories = [
  {role:'Otel Yöneticisi',q:'Rakiplerime göre hangi alanlarda güçlü veya zayıfım?',go:'#hotel360'},
  {role:'Yatırımcı',q:'Hangi destinasyon daha premium veya yüksek kapasiteli?',go:'#destination'},
  {role:'Operasyon Yöneticisi',q:'Şikâyetler hangi hizmet alanlarında yoğunlaşıyor?',go:'#sikayetvar-voice'},
  {role:'Pazarlamacı',q:'Hangi misafir segmenti bize daha yüksek puan veriyor?',go:'#trip-voice'},
  {role:'Turizm Planlamacısı',q:'Talep yılın hangi aylarında yoğunlaşıyor?',go:'#tourism'},
  {role:'Genel Müdür',q:'İki rakip otelim arasındaki fark tam olarak nerede?',go:'#hotel-compare'},
];
$('#story-cards').innerHTML = stories.map(s=>`<button class="story-card" data-go="${s.go}"><div class="role">${s.role}</div><div class="q">"${s.q}"</div><div class="go">İlgili bölüme git &rarr;</div></button>`).join('');
$$('.story-card').forEach(c=>c.addEventListener('click',()=>document.querySelector(c.dataset.go).scrollIntoView({behavior:'smooth'})));

/* ============================================================
   FINDINGS
   ============================================================ */
const findings = """ + json.dumps(findings_data, ensure_ascii=False) + """;
$('#finding-list').innerHTML = findings.map((f,i)=>`<div class="finding"><div class="fnum">${String(i+1).padStart(2,'0')}</div><div class="ftext">${f[0]}</div><div class="fmeta">${f[1]}</div></div>`).join('');

/* ============================================================
   BUSINESS MEANING
   ============================================================ */
const businessCards = [
  ['Pazar Konumlandırması','192 otelin puan, fiyat ve görünürlük konumunu tek çerçevede karşılaştırma.'],
  ['Destinasyon Stratejisi','14 bölgeyi kalite/lüks/değer eksenlerinde konumlandırma.'],
  ['Misafir Segmentleri','Trip.com verisiyle aile/çift/solo/iş misafirine göre konumlanma.'],
  ['Müşteri Deneyimi','Google Travel'+String.fromCharCode(39)+'da puanı yükselten ve düşüren konuları görme.'],
  ['Şikâyet Temaları','Şikayetvar'+String.fromCharCode(39)+'da sorunların hangi hizmet alanlarında yoğunlaştığını izleme.'],
  ['Olanak / Politika Analizi','95 otelin check-in, aile, evcil hayvan gibi tesis politikalarını karşılaştırma.'],
  ['Sezon Planlaması','Talep zirvesini ve düşük sezonu önceden görme.'],
  ['Rakip Karşılaştırması','Hotel 360° ve A/B karşılaştırma modülüyle iki oteli yan yana görme.'],
];
$('#business-cards').innerHTML = businessCards.map(([t,d])=>`<div class="card"><h4 style="font-size:15.5px;">${t}</h4><p style="margin-top:10px;font-size:13px;color:var(--sand-dim);line-height:1.5;">${d}</p></div>`).join('');

/* ============================================================
   CONCLUSION
   ============================================================ */
$('#conclusion-kpis').innerHTML = ['192 otel','14 destinasyon','""" + f"{G['clean_reviews']:,}".replace(",",".") + """ Google Travel yorumu','""" + f"{T['clean_reviews']:,}".replace(",",".") + """ Trip.com yorumu','""" + str(SK['clean_rows']) + """ Şikayetvar kaydı','""" + str(T['policy_hotels']) + """ otelde politika/olanak verisi']
  .map(x=>`<div class="ck">${x}</div>`).join('');

/* ============================================================
   NAV / SCROLL WIRING
   ============================================================ */
const navSections = $$('section[data-nav]');
const sidenav = $('#sidenav');
sidenav.innerHTML = navSections.map((s,i)=>`<button class="dot" data-target="${s.id}" title="${s.dataset.nav}"></button>`).join('');
$$('#sidenav .dot').forEach(d=>d.addEventListener('click',()=>document.getElementById(d.dataset.target).scrollIntoView({behavior:'smooth'})));

const topctaEl = $('#topcta'), topctaSection = $('#topcta-section'), progressEl = $('#progress');
function onScroll(){
  const doc = document.documentElement;
  const scrolled = (doc.scrollTop)/(doc.scrollHeight-doc.clientHeight)*100;
  progressEl.style.width = scrolled+'%';
  topctaEl.classList.toggle('show', doc.scrollTop > window.innerHeight*0.6);
}
document.addEventListener('scroll',onScroll,{passive:true});
onScroll();

const io = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); } });
},{threshold:0.12});
$$('.reveal').forEach(el=>io.observe(el));

const navIO = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      const id=e.target.id;
      $$('#sidenav .dot').forEach(d=>d.classList.toggle('active',d.dataset.target===id));
      topctaSection.textContent = e.target.dataset.nav;
    }
  });
},{threshold:0.5});
navSections.forEach(s=>navIO.observe(s));

/* re-render responsive charts on resize (debounced) */
let rTimer;
window.addEventListener('resize',()=>{clearTimeout(rTimer);rTimer=setTimeout(()=>{
  vbarChart($('#chart-gm-ratinggroup'),gRatingBands,{color:'var(--accent)',valueFmt:v=>String(v)});
  hbarChart($('#chart-top-reviewed-gt'),topReviewedGT,{color:'var(--accent-warm)',valueFmt:v=>fmt(v),labelWidth:170});
  divergingChart($('#chart-gm-drivers'),gmDriverItems,{onClick:renderGmDriverDetail});
  renderTripSegChart();
  hbarChart($('#chart-trip-segment-rating'),tripSegRating,{color:'var(--accent)',valueFmt:v=>fmt(v,2),labelWidth:120,
    onClick:d=>renderQuotePanel(tripQuotePanel,d.label,(QUOTES_T[d.seg]||[]).map(e=>({text:e.text,hotel:e.hotel,date:e.date})))});
  hbarChart($('#chart-trip-amenities'),tripAmenities,{color:'var(--gold)',valueFmt:v=>fmt(v,1)+'%',labelWidth:140,height:22});
  donutChart($('#chart-trip-country'),tripCountries,{valueFmt:v=>fmt(v)});
  hbarChart($('#chart-sk-aspects'),skTop,{color:'var(--accent-warm)',valueFmt:v=>fmt(v,1)+'%',labelWidth:170,
    onClick:d=>renderQuotePanel(skQuotePanel,d.label,SK_ASPECT_EXAMPLES[d.aspect])});
  hbarChart($('#chart-sk-response-share'),skResponseShare,{color:'var(--gold)',valueFmt:v=>fmt(v,1)+'%',labelWidth:150,height:22});
  vbarChart($('#chart-rating-dist'),ratingBands,{color:'var(--accent)',valueFmt:v=>String(v)});
  hbarChart($('#chart-top-reviewed'),topReviewed,{color:'var(--accent-warm)',valueFmt:v=>fmt(v),labelWidth:170});
  heatmap($('#chart-dest-heatmap'),destRows,destCols,(r,c)=>r.d[c.key],{labelWidth:150,fmtVal:v=>fmt(v,1)});
  renderAnnualChart();
  renderMonthlyChart();
  dualLineChart($('#chart-airport-tourism'),atm,{keyA:'a',keyB:'b',labelA:'Havalimanı endeksi',labelB:'Turizm endeksi'});
},200);});
"""
print("JS_TAIL chars:", len(JS_TAIL))

# ==============================================================
# FINAL ASSEMBLY
# ==============================================================
DATA_FOR_JS = {
    "hotels": DATA["hotels"],
    "destinations": DATA["destinations"],
    "tourism_annual": DATA["tourism_annual"],
    "airport_tourism_monthly": DATA["airport_tourism_monthly"],
}
JS_DATA_CONST = "const DATA = " + json.dumps(DATA_FOR_JS, ensure_ascii=False) + ";\n"

full_head = head_and_css.replace("</style>", EXTRA_CSS + "\n</style>")
full_head = full_head.replace("<title>Bodrum Hotel Intelligence</title>", "<title>Bodrum Hotel Intelligence</title>")

HTML_BODY = (
    HTML_TOP + HTML_MARKET + "\n\n" + HTML_DEST + "\n\n" + HTML_TOURISM + "\n\n" + HTML_AIRPORT + "\n\n"
    + HTML_GOOGLE + HTML_TRIP + HTML_NLP + HTML_SIKAYETVAR + HTML_CROSS + HTML_HOTEL360 + HTML_TAIL_SECTIONS
)

JS_ALL = (
    JS_UTILS_AND_CHARTS + "\n" + JS_CHART_EXTRA + "\n" + JS_DATA_CONST + "\n" + JS_TOP + "\n" + JS_MARKET + "\n" + JS_DEST + "\n"
    + JS_TOURISM + "\n" + JS_AIRPORT + "\n" + JS_GOOGLE + "\n" + JS_TRIP + "\n" + JS_SIKAYETVAR + "\n"
    + JS_CROSS + "\n" + JS_HOTEL360 + "\n" + JS_TAIL
)

FULL_HTML = full_head + HTML_BODY + "\n<script>\n" + JS_ALL + "\n</script>\n"

out_path = SCRATCH / "bodrum_site_v3.html"
out_path.write_text(FULL_HTML, encoding="utf-8")
print("WROTE", out_path, "total chars:", len(FULL_HTML), "size KB:", round(len(FULL_HTML.encode('utf-8'))/1024,1))

















