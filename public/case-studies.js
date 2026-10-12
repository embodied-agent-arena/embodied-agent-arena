(() => {
  'use strict';
  const dialog = document.getElementById('study-dialog');
  if (!dialog) return;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const num = (n, places = 2) => typeof n === 'number' && Number.isFinite(n) ? Number(n.toFixed(places)).toLocaleString('en-US', {maximumFractionDigits:places}) : '—';
  const compact = value => typeof value === 'object' ? JSON.stringify(value) : String(value ?? 'Not recorded');
  const shortNames = {'Claude Fable 5.1':'Fable 5.1','Claude Opus 5.5':'Opus 5.5','Qwen 3.5 397B-A17B':'Qwen 397B','Qwen 3.5 27B':'Qwen 27B'};
  const short = name => shortNames[name] || name;
  const colours = ['#ad5a60','#456f8c','#9a7042','#71669e','#527e71','#75974f','#727d87','#a84f81'];
  const domains = {geometry:'Geometry',spatial:'Spatial Reasoning',affordance:'Affordance',planning:'Task Planning',manipulation:'Manipulation'};
  const arrows = {right:'→',down_right:'↘',down:'↓',down_left:'↙',left:'←',up_left:'↖',up:'↑',up_right:'↗',none:'Stationary'};
  const budgetNames = {max_code_turns:'Code rounds',max_agent_iterations:'Model rounds',max_total_tokens:'Tokens',trial_timeout_seconds:'Time ceiling (s)',timeout_seconds:'Time ceiling (s)',max_env_steps:'Environment steps',max_primitive_calls:'Primitive calls',max_verifier_calls:'Verifier calls',code_timeout_seconds:'Per-cell time (s)',max_agent_attempts:'Agent attempts',llm_num_retries:'Request retries'};
  let records, metadata = {}, currentUrl, trigger, openId;
  let selectedModel = 0;
  let loadPromise;
  const load = () => loadPromise ||= Promise.all([
    fetch('data/case-studies.json').then(r => {if (!r.ok) throw new Error('Case data could not be loaded.'); return r.json();}),
    fetch('data/case-gallery.json').then(r => {if (!r.ok) throw new Error('Comparison metadata could not be loaded.'); return r.json();})
  ]).then(([cases, meta]) => {records = new Map(cases.map(c => [c.id,c])); metadata = meta; return cases;});
  const metricSpecs = [
    ['absolute_error_cm','Distance error','cm',1],['trace_3d_frechet','3D Fréchet distance','m',1],
    ['trace_2d_frechet','2D Fréchet (normalized)','',1],['rotation_error_deg','Rotation error','°',1],
    ['bbox_iou','Box IoU','',1],['focal_mape','Focal MAPE','%',100],['focal_mape_pct','Focal MAPE','%',1],
    ['focal_ape_pct','Focal error','%',1],['relative_l2','Relative L2 error','',1],
    ['abs_rel','Absolute relative error','',1],['mape_pct','MAPE','%',1]
  ];
  const specFor = c => metricSpecs.find(([key]) => c.predictions.some(p => typeof p.metrics?.[key] === 'number'));
  function summary(c, p) {
    if (p.valid === false) return {value:'Invalid', label:'No valid prediction', state:'uncertain'};
    const passed = metadata[c.id]?.judgements[p.model];
    if (c.domainKey === 'spatial') return {value:compact(p.answer),label:passed === true ? 'Correct' : passed === false ? 'Incorrect' : 'Recorded answer',state:passed === true ? 'correct' : passed === false ? 'incorrect' : ''};
    const spec = specFor(c);
    if (!spec) return {value:'View answer', label:'First-run prediction',state:''};
    const [key,label,unit,mult] = spec;
    const value = p.metrics?.[key];
    const precision = ['bbox_iou','relative_l2','abs_rel','trace_3d_frechet','trace_2d_frechet'].includes(key) ? 3 : 2;
    const numeric = typeof value === 'number' ? `${num(value * mult,precision)}${unit ? ` ${unit}` : ''}` : 'Not recorded';
    if (key === 'absolute_error_cm' && p.answer?.distance_cm != null) return {value:`${num(p.answer.distance_cm)} cm ${arrows[p.answer.direction] || ''}`, label:`Error ${numeric}`,state:''};
    return {value:numeric, label, state:''};
  }
  function referenceLabel(c) {
    const r = c.reference;
    if (r == null) return '';
    if (typeof r === 'string') {
      const options = c.options?.length ? c.options : [...c.question.matchAll(/\b([A-D])\.\s+([\s\S]*?)(?=\s+[A-D]\.\s+|$)/g)].map(m => `${m[1]}. ${m[2]}`);
      const option = options.find(v => typeof v === 'string' && v.startsWith(`${r}.`));
      return `Reference: ${option && option.length < 100 ? option : r}`;
    }
    if (r.distance_cm != null) return `Reference: ${num(r.distance_cm)} cm ${arrows[r.direction] || r.direction}`;
    return '';
  }
  function isOverlay(c) {return c.domainKey === 'affordance' || c.benchmark.startsWith('TraceSpatial');}
  // Reproject the recorded 3D reference into the source image; predictions
  // use the task's declared 0–1000 image-coordinate convention.
  function overlay(c, prediction, index) {
    const [w,h] = metadata[c.id]?.imageSize || [640,480];
    const box = (v, colour, dash='') => Array.isArray(v) && v.length === 4 && v.every(Number.isFinite) ? `<rect x="${v[0]}" y="${v[1]}" width="${Math.max(0,v[2]-v[0])}" height="${Math.max(0,v[3]-v[1])}" fill="none" stroke="${colour}" stroke-width="3" vector-effect="non-scaling-stroke" ${dash}/>` : '';
    const dot = (v, colour) => Array.isArray(v) && v.length >= 2 && v.slice(0,2).every(Number.isFinite) ? `<circle cx="${v[0]}" cy="${v[1]}" r="6" fill="${colour}" stroke="white" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` : '';
    const path = (points, colour, dash='') => points.length > 1 ? `<polyline points="${points.map(p => p.join(',')).join(' ')}" fill="none" stroke="${colour}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" ${dash}/>${dot(points[points.length-1],colour)}` : '';
    let shapes = '';
    if (c.domainKey === 'affordance') {
      shapes = box(c.reference?.bbox_2d,'#6be7b0','stroke-dasharray="6 4"') + box(prediction.answer?.bbox_2d,colours[index]) + dot(prediction.answer?.point_2d,colours[index]);
    } else {
      const k = c.reference?.gt_depth_intrinsics || c.reference?.intrinsics;
      const ref = k ? (c.reference?.trajectory || []).filter(p => p[2] > 0).map(([x,y,z]) => [k[0][0]*x/z+k[0][2],k[1][1]*y/z+k[1][2]]) : [];
      const pred = Array.isArray(prediction.answer) ? prediction.answer.filter(p => Array.isArray(p) && p.slice(0,2).every(Number.isFinite)).map(p => [p[0]*w/1000,p[1]*h/1000]) : [];
      shapes = path(ref,'#6be7b0','stroke-dasharray="6 4"') + path(pred,colours[index]);
    }
    return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(prediction.model)} prediction and reference over the original input"><image href="${esc(c.images[0])}" width="${w}" height="${h}"/>${shapes}</svg>`;
  }
  function inputMedia(c, detail=false) {
    if (c.video) return `<div class="gallery-film"><video ${detail ? 'controls' : 'muted loop'} playsinline preload="none" poster="${esc(c.poster)}" ${detail ? `src="${esc(c.video)}"` : `data-video="${esc(c.video)}"`}></video>${detail ? '' : '<span class="gallery-film-label">Opus 5.5 · Click to Watch ↗</span>'}</div>`;
    if (c.animation) return `<div class="gallery-film"><img ${detail ? `src="${esc(c.animation)}"` : `src="${esc(c.poster)}" data-animation="${esc(c.animation)}"`} alt="${esc(c.mediaKind)} recorded for Opus 5.5" loading="lazy"><span class="gallery-film-label">Opus 5.5 · ${esc(c.mediaKind)}</span></div>`;
    const images = detail ? c.images : c.images.slice(0,2);
    return `<div class="gallery-inputs ${images.length > 1 ? 'gallery-inputs-pair' : ''}">${images.map((src,i) => `<figure><img src="${esc(src)}" alt="${esc(c.benchmark)} task input ${i+1}" loading="lazy">${images.length > 1 ? `<figcaption>Input ${i+1}</figcaption>` : ''}</figure>`).join('')}</div>`;
  }
  function executionRecords(c) {
    const own = {model:c.model,outcome:c.stopReason,success:c.result === 'Reported success',protocol:c.domainKey === 'planning' ? 'Baseline' : 'Source-native primitives; per-round image delivery unverified',own:true};
    if (c.result.includes('Conflicting')) own.outcome='conflicting_records';
    return [...(metadata[c.id]?.recordedOutcomes || []),own];
  }
  const outcomeText = r => r.outcome === 'conflicting_records' ? 'Unresolved' : r.success ? 'Reported success' : ({budget_exhausted:'Budget exhausted',timeout:'Timed out',runtime_failure:'Runtime failure',task_failure:'Task not completed',max_iterations:'Round limit reached'}[r.outcome] || 'Not completed');
  function resultTile(c,p,i, visual=false) {
    const s = summary(c,p);
    return `<button class="gallery-model ${visual ? 'gallery-model-visual' : ''} ${s.state}" data-case="${esc(c.id)}" data-model="${i}" style="--model-colour:${colours[i]}" aria-label="Inspect ${esc(p.model)} on ${esc(c.title)}" title="${esc(p.model)}"><span class="gallery-model-name">${esc(short(p.model))}<span aria-hidden="true">↗</span></span>${visual ? overlay(c,p,i) : ''}<span class="gallery-model-value">${esc(s.value)}</span><span class="gallery-model-caption">${esc(s.label)}</span></button>`;
  }
  function caseSlide(c,index,total) {
    let body;
    if (isOverlay(c)) {
      body = `<div class="gallery-visual-grid">${c.predictions.map((p,i) => resultTile(c,p,i,true)).join('')}</div><p class="gallery-evidence-note"><span class="gallery-ref-line"></span> Reference <span class="gallery-pred-line"></span> Model prediction${c.benchmark.includes('3D') ? ' · Image-plane projection; depth is scored in 3D.' : ''}</p>`;
    } else if (c.predictions.length) {
      body = `<div class="gallery-comparison"><button class="gallery-scene" data-case="${esc(c.id)}" aria-label="Open ${esc(c.title)} task details">${inputMedia(c)}${referenceLabel(c) ? `<span class="gallery-reference">${esc(referenceLabel(c))}</span>` : ''}<span class="gallery-enlarge">Explore Task ↗</span></button><div class="gallery-score-panel"><p class="gallery-panel-label">First-Run Answers${c.domainKey === 'spatial' ? '' : ' · Individual Case'}</p><div class="gallery-model-grid">${c.predictions.map((p,i) => resultTile(c,p,i)).join('')}</div></div></div>`;
    } else {
      const rows = executionRecords(c);
      body = `<div class="gallery-comparison gallery-execution"><button class="gallery-scene" data-case="${esc(c.id)}" aria-label="Watch Opus 5.5 on ${esc(c.title)}">${inputMedia(c)}</button><div class="gallery-score-panel"><p class="gallery-panel-label">Recorded Outcomes${c.domainKey === 'manipulation' && rows.length>1 ? ' · Settings Differ' : ''}</p><div class="gallery-model-grid">${rows.map((r,i) => `<button class="gallery-model gallery-terminal ${r.outcome==='conflicting_records' ? 'uncertain' : r.success ? 'correct' : ''}" data-case="${esc(c.id)}" data-model="${i}" aria-label="Inspect ${esc(r.model)} execution record"><span class="gallery-model-name">${esc(short(r.model))}<span aria-hidden="true">↗</span></span><span class="gallery-model-value">${esc(outcomeText(r))}</span></button>`).join('')}</div>${c.domainKey === 'manipulation' && rows.length>1 ? '<p class="gallery-evidence-note">Observation settings vary across these recordings.</p>' : ''}${rows.length === 1 ? `<p class="gallery-run-facts">${c.modelRounds != null ? `${num(c.modelRounds)} model responses` : ''}<br>${c.elapsedSeconds != null ? `${num(c.elapsedSeconds,1)} s elapsed` : ''}</p>` : ''}</div></div>`;
    }
    return `<article class="gallery-slide" data-gallery-id="${esc(c.id)}" data-domain="${esc(c.domainKey)}" role="group" aria-roledescription="slide" aria-label="Case ${index+1} of ${total}: ${esc(c.benchmark)} — ${esc(c.title)}"><div class="gallery-slide-top"><div><p class="gallery-kicker">${esc(c.domain)} <span>/</span> ${esc(c.benchmark)}</p><h3>${esc(c.title)}</h3></div><button class="gallery-detail-button" data-case="${esc(c.id)}">Case Details <span aria-hidden="true">↗</span></button></div><p class="gallery-task">${esc(c.shortTask)}</p>${body}</article>`;
  }
  const content = document.getElementById('study-dialog-content');
  function renderModelDetail(c,index) {
    const staticCase = c.predictions.length > 0;
    const list = staticCase ? c.predictions : executionRecords(c);
    const p = list[index] || list[0];
    const selection = document.getElementById('gallery-model-detail');
    if (!p || !selection) return;
    dialog.querySelectorAll('[data-detail-model]').forEach(b => b.setAttribute('aria-pressed',String(Number(b.dataset.detailModel) === index)));
    if (staticCase) {
      const s = summary(c,p);
      selection.innerHTML = `<h3>${esc(p.model)}</h3><div class="gallery-detail-comparison"><div>${isOverlay(c) ? overlay(c,p,index) : inputMedia(c,true)}</div><div><p class="gallery-detail-metric ${s.state}"><strong>${esc(s.value)}</strong>${esc(s.label)}</p><h4>Recorded Answer</h4><pre>${esc(JSON.stringify(p.answer,null,2))}</pre>${c.reference != null ? `<h4>Reference</h4><pre>${esc(JSON.stringify(c.reference,null,2))}</pre>` : ''}</div></div>${Object.keys(p.metrics || {}).length ? `<h3>Case Metrics</h3><div class="study-table-wrap"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${Object.entries(p.metrics).map(([key,value]) => `<tr><td>${esc(key)}</td><td>${esc(compact(value))}</td></tr>`).join('')}</tbody></table></div>` : ''}`;
    } else {
      selection.innerHTML = `<h3>${esc(p.model)}</h3><p class="study-outcome">${esc(outcomeText(p))}</p><p class="study-meta">${esc(p.protocol)}</p>${p.own ? inputMedia(c,true) + `<p>${c.modelRounds != null ? `${num(c.modelRounds)} model responses · ` : ''}${c.elapsedSeconds != null ? `${num(c.elapsedSeconds,1)} s wall-clock time` : ''}</p>` : '<div class="gallery-record-only"><span aria-hidden="true">▤</span><p>This model has a terminal outcome record for this case. A matching video and per-run budget details are not included in the supplied materials.</p></div>'}`;
    }
  }
  async function open(id,button,modelIndex=null) {
    trigger = button; openId = id; selectedModel = modelIndex;
    content.innerHTML='<h2 id="study-dialog-title">Loading Case…</h2>';
    if (!dialog.open) dialog.showModal();
    document.body.classList.add('study-modal-open');
    document.querySelectorAll('.gallery-track video').forEach(v=>v.pause());
    try {
      await load();
      const c = records.get(id); if (!c) throw new Error('Case not found.');
      if (!dialog.open || openId !== id) return;
      const list = c.predictions.length ? c.predictions : executionRecords(c);
      selectedModel=modelIndex ?? (c.predictions.length ? 0 : list.length-1);
      const budget = Object.entries(c.budgets||{}).filter(([k,v]) => budgetNames[k] && v!=null).map(([k,v])=>`<li><span>${budgetNames[k]}</span><strong>${num(v)}</strong></li>`).join('');
      content.innerHTML=`<p class="eyebrow">${esc(c.domain)} · ${esc(c.benchmark)}</p><h2 id="study-dialog-title">${esc(c.title)}</h2><p class="study-question">${esc(c.shortTask)}</p><div class="gallery-model-switcher" role="group" aria-label="Select model">${list.map((p,i)=>`<button data-detail-model="${i}" aria-pressed="${i===selectedModel}">${esc(short(p.model))}</button>`).join('')}</div><div id="gallery-model-detail"></div>${isOverlay(c) ? '<p class="gallery-evidence-note">Dashed mint: reference. Solid colour: prediction. Overlays preserve the original image coordinates.</p>' : ''}${c.comparisonFigure ? `<a class="study-download" href="${esc(c.comparisonFigure)}" target="_blank" rel="noopener">Open Full Comparison Figure ↗</a>` : ''}${budget ? `<h3>${c.predictions.length ? '' : 'Opus 5.5 · '}Recorded Budget Ceilings</h3><ul class="study-budget-list">${budget}</ul>` : ''}<h3>${esc(c.instructionKind || 'Original Instruction')}</h3><p class="study-original-instruction">${esc(c.question)}</p>${c.options?.length ? `<ul>${c.options.map(v=>`<li>${esc(compact(v))}</li>`).join('')}</ul>` : ''}<h3>Reading This Case</h3><p class="study-protocol">${esc(c.protocolNote)}</p>${!c.predictions.length && list.length>1 ? '<p class="study-protocol">Terminal outcomes come from the supplied case records. They are not a matched-budget leaderboard; select each model to inspect its recorded protocol. The budget ceilings above apply to the Opus recording only.</p>' : ''}<a class="study-download" id="study-json-download">Download Case Record (JSON) ↓</a>`;
      renderModelDetail(c,selectedModel);
      dialog.scrollTop=0;
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl=URL.createObjectURL(new Blob([JSON.stringify({...c,galleryMetadata:metadata[c.id]},null,2)],{type:'application/json'}));
      const download=document.getElementById('study-json-download'); download.href=currentUrl; download.download=`${c.id}.json`;
    } catch(error) {content.innerHTML=`<h2 id="study-dialog-title">Case Details Unavailable</h2><p>${esc(error.message)} Please reload and try again.</p>`;}
  }
  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-case]');
    if (button) open(button.dataset.case,button,button.dataset.model === undefined ? null : Number(button.dataset.model));
    const model=event.target.closest('[data-detail-model]');
    if (model && records?.has(openId)) {selectedModel=Number(model.dataset.detailModel); renderModelDetail(records.get(openId),selectedModel);}
  });
  dialog.addEventListener('close',()=>{
    dialog.querySelectorAll('video').forEach(v=>v.pause()); content.replaceChildren();
    document.body.classList.remove('study-modal-open');
    if (currentUrl) {URL.revokeObjectURL(currentUrl);currentUrl=null;}
    trigger?.focus({preventScroll:true}); updateActiveMedia();
  });
  dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  const grid=document.querySelector('.study-grid');
  if (!grid) return;
  const filters=[...document.querySelectorAll('[data-study-filter]')];
  const search=document.getElementById('study-search');
  let active=filters.find(b=>b.getAttribute('aria-pressed')==='true')?.dataset.studyFilter || 'highlights';
  const gallery=document.createElement('div'); gallery.className='case-gallery';
  gallery.innerHTML=`<div class="gallery-track" tabindex="0" role="region" aria-roledescription="carousel" aria-label="Case comparisons; use left and right arrow keys to browse"></div><div class="gallery-navigation"><p class="gallery-help"><span aria-hidden="true">↔</span> Scroll to Explore · Click a Model for Details</p><div class="gallery-position"><output id="gallery-position" aria-live="polite"></output><button data-gallery-step="-1" aria-label="Previous case">←</button><button data-gallery-step="1" aria-label="Next case">→</button></div></div><input class="gallery-scrubber" type="range" min="1" max="15" value="1" aria-label="Choose a case"><p class="gallery-empty" hidden>No matching cases. Try another search.</p>`;
  grid.before(gallery);
  const track=gallery.querySelector('.gallery-track');
  const position=gallery.querySelector('#gallery-position');
  const scrubber=gallery.querySelector('.gallery-scrubber');
  let allCases=[], shown=[], slides=[], current=0, isVisible=false, ticking=false;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  let observer;
  // Cover every supplied benchmark/subset; retain paper cases and contrasting tasks.
  const showcaseIds = ["W1-w1_vlm_motion_real-01", "W2-w2_mindcube-01", "W5-w5_umd-03", "W3-alfworld_visual-01", "W4-calvin-01", "W1-w1_tracespatial_3d-01", "W2-w2_3dsrbench-01", "W5-w5_ragnet-02", "W3-discoveryworld-01", "W4-capx_libero_pro-03", "W1-w1_influx-01", "W2-w2_bop_ask-01", "W5-w5_reasonaff-03", "W3-alfred_official_visual-01", "W4-calvin-03", "W1-w1_mapfree-01", "W2-w2_mmsi_bench-01", "W5-w5_umd-01", "W3-alfworld_visual-03", "W4-capx_behavior1k-01", "W1-w1_multispa-01", "W2-w2_robospatial-01", "W3-humanclaw-02", "W4-capx_robosuite-01", "W1-w1_multispa-02", "W2-w2_vsi_bench-01", "W3-scienceworld_text-01", "W4-cliport-01", "W1-w1_tracespatial_2d-01", "W2-w2_vsi_bench-02", "W3-virtualhome_symbolic-01", "W4-maniskill-01", "W1-w1_vlm_depth-01", "W4-rlbench-01", "W1-w1_vlm_focal-01", "W4-robocasa-01", "W1-w1_vlm_intrinsics-01", "W4-robocasa-02", "W1-w1_vlm_motion_camera-01", "W4-robocasa365-01", "W1-w1_vlm_motion_camera_large-01", "W4-robotwin2-01", "W1-w1_vlm_motion_fixed-01", "W4-robowits-01", "W1-w1_vlm_pose-01", "W4-vimabench-01", "W1-w1_vlm_reference-01", "W4-vlabench-01", "W1-w1_vlm_reference-03", "W1-w1_vlm_size-01"];
  const showcase = new Set(showcaseIds);
  const priority=c=>showcase.has(c.id)?showcaseIds.indexOf(c.id):200;
  function hydrate(index) {
    for(let i=Math.max(0,index-1);i<=Math.min(slides.length-1,index+1);i++) {
      const slide=slides[i]; if(!slide || slide.dataset.ready) continue;
      slide.innerHTML=caseSlide(shown[i],i,shown.length);slide.dataset.ready='true';
    }
  }
  function updateActiveMedia() {
    slides.forEach((slide,i)=>{
      slide.querySelectorAll('video').forEach(v=>{
        if(i===current && isVisible && !document.hidden && !dialog.open && !reduced.matches){if(!v.src)v.src=v.dataset.video;v.play().catch(()=>{});}else v.pause();
      });
      slide.querySelectorAll('[data-animation]').forEach(img=>{
        const src=i===current&&isVisible&&!document.hidden&&!dialog.open&&!reduced.matches?img.dataset.animation:shown[i].poster;
        if(img.getAttribute('src')!==src)img.src=src;
      });
    });
  }
  function updatePosition() {
    if (!slides.length) return;
    current=slides.reduce((best,slide,i)=>Math.abs(slide.offsetLeft-track.offsetLeft-track.scrollLeft)<Math.abs(slides[best].offsetLeft-track.offsetLeft-track.scrollLeft)?i:best,0);
    hydrate(current);
    position.textContent=`${String(current+1).padStart(2,'0')} / ${shown.length}`;
    slides.forEach((slide,i)=>{slide.inert=i!==current;});
    scrubber.value=current+1;
    scrubber.setAttribute('aria-valuetext',`${current+1} of ${shown.length}: ${shown[current].benchmark}, ${shown[current].title}`);
    gallery.querySelector('[data-gallery-step="-1"]').disabled=current===0;
    gallery.querySelector('[data-gallery-step="1"]').disabled=current===slides.length-1;
    updateActiveMedia();
  }
  function go(index,smooth=true) {
    if(!slides.length) return;
    index=Math.max(0,Math.min(slides.length-1,index));hydrate(index);
    track.scrollTo({left:slides[index].offsetLeft-track.offsetLeft,behavior:smooth&&!reduced.matches?'smooth':'instant'});
    if(!smooth)updatePosition();
  }
  function applyFilter() {
    const query=(search?.value||'').trim().toLowerCase();
    shown=allCases.filter(c=>(active==='all'||active==='highlights'&&showcase.has(c.id)||c.domainKey===active)&&(!query||[c.benchmark,c.title,c.shortTask,c.question,...c.predictions.map(p=>p.model)].join(' ').toLowerCase().includes(query))).sort((a,b)=>priority(a)-priority(b));
    track.querySelectorAll('video').forEach(v=>v.pause());
    track.innerHTML=shown.map((c,i)=>`<div class="gallery-slot" aria-label="${esc(c.benchmark)}: ${esc(c.title)}"><p class="gallery-loading">${i+1} · ${esc(c.benchmark)}</p></div>`).join('');
    slides=[...track.children];current=0;hydrate(0);track.scrollLeft=0;
    scrubber.max=Math.max(1,shown.length);scrubber.value=1;
    gallery.querySelector('.gallery-empty').hidden=!!shown.length;
    track.hidden=!shown.length;gallery.querySelector('.gallery-navigation').hidden=!shown.length;scrubber.hidden=shown.length<2;
    const count=document.getElementById('study-count');if(count)count.textContent=`${shown.length} cases`;
    updatePosition();
  }
  filters.forEach(button=>button.addEventListener('click',()=>{active=button.dataset.studyFilter;filters.forEach(b=>b.setAttribute('aria-pressed',String(b===button)));applyFilter();}));
  search?.addEventListener('input',applyFilter);
  gallery.querySelectorAll('[data-gallery-step]').forEach(button=>button.addEventListener('click',()=>go(current+Number(button.dataset.galleryStep))));
  scrubber.addEventListener('input',()=>go(Number(scrubber.value)-1,false));
  track.addEventListener('keydown',event=>{if(event.target!==track)return;if(['ArrowRight','ArrowLeft','Home','End'].includes(event.key)){event.preventDefault();go(event.key==='Home'?0:event.key==='End'?slides.length-1:current+(event.key==='ArrowRight'?1:-1));}});
  track.addEventListener('scroll',()=>{if(!ticking){requestAnimationFrame(()=>{updatePosition();ticking=false;});ticking=true;}},{passive:true});
  // A vertical wheel gesture browses one case at a time. At the two ends it
  // scrolls the document normally; native horizontal trackpad gestures stay native.
  let wheelTotal=0,lastWheel=0,lockedUntil=0;
  track.addEventListener('wheel',event=>{
    if(event.ctrlKey||event.metaKey||Math.abs(event.deltaX)>Math.abs(event.deltaY)||!shown.length||window.innerWidth<760)return;
    const rect=track.getBoundingClientRect();
    const visibleHeight=Math.min(rect.bottom,window.innerHeight)-Math.max(rect.top,80);
    if(visibleHeight<Math.min(rect.height*.6,390))return;
    const direction=Math.sign(event.deltaY);const now=performance.now();
    const atEdge=direction>0&&current===slides.length-1||direction<0&&current===0;
    if(atEdge&&now>=lockedUntil)return;
    event.preventDefault();
    if(now<lockedUntil)return;
    if(now-lastWheel>200)wheelTotal=0;
    lastWheel=now;wheelTotal+=event.deltaY*(event.deltaMode===1?16:1);
    if(Math.abs(wheelTotal)>35){go(current+Math.sign(wheelTotal));wheelTotal=0;lockedUntil=now+650;}
  },{passive:false});
  observer=new IntersectionObserver(entries=>{isVisible=entries[0].isIntersecting;updateActiveMedia();},{threshold:.2});observer.observe(track);
  reduced.addEventListener('change',updateActiveMedia);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){document.querySelectorAll('.gallery-track video').forEach(v=>v.pause());}else updateActiveMedia();});
  window.addEventListener('resize',()=>go(current,false));
  load().then(cases=>{allCases=cases;grid.hidden=true;gallery.classList.add('is-ready');applyFilter();}).catch(error=>{
    gallery.innerHTML=`<p class="study-protocol">${esc(error.message)} <a href="case-atlas.html">Browse the Case Atlas ↗</a></p>`;
  });
})();
