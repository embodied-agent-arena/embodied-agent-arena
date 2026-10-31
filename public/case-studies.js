(() => {
  'use strict';
  const dialog = document.getElementById('study-dialog');
  if (!dialog) return;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const num = (n, places = 2) => typeof n === 'number' && Number.isFinite(n) ? Number(n.toFixed(places)).toLocaleString('en-US', {maximumFractionDigits:places}) : '—';
  const compact = value => typeof value === 'object' ? JSON.stringify(value) : String(value ?? 'Not recorded');
  const shortNames = {'Claude Fable 5.1':'Fable 5.1','Claude Opus 5.5':'Opus 5.5','Qwen 3.5 397B-A17B':'Qwen 397B','Qwen 3.5 27B':'Qwen 27B','DeepSeek 4.1 Flash-1017':'DeepSeek-1017'};
  const short = name => shortNames[name] || name;
  const colours = ['#ad5a60','#456f8c','#9a7042','#71669e','#527e71','#75974f','#727d87','#a84f81','#237e83','#c37b28'];
  const domains = {geometry:'Geometry',spatial:'Spatial Reasoning',affordance:'Affordance',planning:'Task Planning',manipulation:'Manipulation'};
  const arrows = {right:'→',down_right:'↘',down:'↓',down_left:'↙',left:'←',up_left:'↖',up:'↑',up_right:'↗',none:'Stationary'};
  const budgetNames = {max_code_turns:'Code rounds',max_agent_iterations:'Model rounds',max_total_tokens:'Tokens',trial_timeout_seconds:'Time ceiling (s)',timeout_seconds:'Time ceiling (s)',max_env_steps:'Environment steps',max_primitive_calls:'Primitive calls',max_verifier_calls:'Verifier calls',code_timeout_seconds:'Per-cell time (s)',max_agent_attempts:'Agent attempts',llm_num_retries:'Request retries'};
  let records, metadata = {}, mediaData = {}, currentUrl, trigger, openId;
  const pausedCases = new Set();
  const manualPlay = new Set();
  const boxCases = new Set();
  const representation = c => boxCases.has(c.id) && c.boxComparison ? c.boxComparison : c;
  let selectedModel = 0;
  let loadPromise;
  const load = () => loadPromise ||= Promise.all([
    fetch('data/case-studies.json?v=toolbox-20261031').then(r => {if (!r.ok) throw new Error('Case data could not be loaded.'); return r.json();}),
    fetch('data/case-gallery.json?v=toolbox-20261031').then(r => {if (!r.ok) throw new Error('Comparison metadata could not be loaded.'); return r.json();}),
    fetch('data/case-comparisons.json?v=toolbox-20261031').then(r => {if (!r.ok) throw new Error('Model recordings could not be loaded.'); return r.json();}),
    fetch('data/report/case-execution-health.json?v=toolbox-20261031').then(r => {if (!r.ok) throw new Error('Execution audit could not be loaded.'); return r.json();})
  ]).then(([cases, meta, media, health]) => {
    for(const [id,issue] of Object.entries(health.runs)) {
      const run=media.runs[id];
      if(!run || run.taskId!==issue.taskId || run.sourceReportSha256!==issue.reportSha256)throw new Error('Execution audit does not match the source record.');
      run.executionReview=issue;
    }
    mediaData=media;
    const all=[...cases.filter(c=>c.predictions.length),...media.groups];
    records=new Map(all.map(c=>[c.id,c]));
    for(const [alias,id] of Object.entries(media.aliases))if(records.has(id))records.set(alias,records.get(id));
    metadata=meta;return all;
  });
  const metricSpecs = [
    ['absolute_error_cm','Distance error','cm',1],['trace_3d_frechet','3D Fréchet distance','m',1],
    ['trace_2d_frechet','2D Fréchet (normalized)','',1],['rotation_error_deg','Rotation error','°',1],
    ['bbox_iou','Box–box IoU','',1],['focal_mape','Focal MAPE','%',100],['focal_mape_pct','Focal MAPE','%',1],
    ['focal_ape_pct','Focal error','%',1],['relative_l2','Relative L2 error','',1],
    ['abs_rel','Absolute relative error','',1],['mape_pct','MAPE','%',1]
  ];
  const specFor = c => metricSpecs.find(([key]) => c.predictions.some(p => typeof p.metrics?.[key] === 'number'));
  function summary(c, p) {
    if (c.representation === 'polygon') {
      if (p.available === false) return {value:'Not recorded',label:'Polygon absent from archive',state:'uncertain'};
      return {value:num(p.metrics?.polygon_mask_iou,3),label:p.valid ? 'Polygon IoU' : 'Polygon IoU · Invalid or absent output',state:p.valid ? '' : 'uncertain'};
    }
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
  function overlay(c, prediction, index, detail=false) {
    const [w,h] = metadata[c.id]?.imageSize || [640,480];
    const box = (v, colour, dash='') => Array.isArray(v) && v.length === 4 && v.every(Number.isFinite) ? `<rect x="${v[0]}" y="${v[1]}" width="${Math.max(0,v[2]-v[0])}" height="${Math.max(0,v[3]-v[1])}" fill="none" stroke="${colour}" stroke-width="3" vector-effect="non-scaling-stroke" ${dash}/>` : '';
    const dot = (v, colour) => Array.isArray(v) && v.length >= 2 && v.slice(0,2).every(Number.isFinite) ? `<circle cx="${v[0]}" cy="${v[1]}" r="6" fill="${colour}" stroke="white" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` : '';
    const path = (points, colour, dash='') => points.length > 1 ? `<polyline points="${points.map(p => p.join(',')).join(' ')}" fill="none" stroke="${colour}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" ${dash}/>${dot(points[points.length-1],colour)}` : '';
    let shapes = '';
    if (c.representation === 'polygon') {
      shapes = `<image href="${esc(c.reference.mask)}" width="${w}" height="${h}"/>`;
      if (prediction.valid) for (const polygon of prediction.answer.polygons || []) {
        const rings = [polygon.outer, ...(polygon.holes || [])];
        const d = rings.filter(r=>Array.isArray(r)&&r.length>=3&&r.every(p=>p.length===2&&p.every(Number.isFinite))).map(r=>`M ${r.map(p=>p.join(' ')).join(' L ')} Z`).join(' ');
        shapes += `<path d="${d}" fill="${colours[index]}" fill-opacity=".12" fill-rule="evenodd" stroke="${colours[index]}" stroke-width="2.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`;
      }
    } else if (c.domainKey === 'affordance') {
      shapes = box(c.reference?.bbox_2d,'#6be7b0','stroke-dasharray="6 4"') + box(prediction.answer?.bbox_2d,colours[index]) + dot(prediction.answer?.point_2d,colours[index]);
    } else {
      const k = c.reference?.gt_depth_intrinsics || c.reference?.intrinsics;
      const ref = k ? (c.reference?.trajectory || []).filter(p => p[2] > 0).map(([x,y,z]) => [k[0][0]*x/z+k[0][2],k[1][1]*y/z+k[1][2]]) : [];
      const pred = Array.isArray(prediction.answer) ? prediction.answer.filter(p => Array.isArray(p) && p.slice(0,2).every(Number.isFinite)).map(p => [p[0]*w/1000,p[1]*h/1000]) : [];
      shapes = path(ref,'#6be7b0','stroke-dasharray="6 4"') + path(pred,colours[index]);
    }
    const viewBox=c.representation==='polygon'&&!detail&&c.polygonViewBox ? c.polygonViewBox.join(' ') : `0 0 ${w} ${h}`;
    return `<svg viewBox="${viewBox}" role="img" aria-label="${esc(prediction.model)} ${c.representation==='polygon'?'polygon and reference mask':'prediction and reference'} over the original input"><image href="${esc(c.images[0])}" width="${w}" height="${h}"/>${shapes}</svg>`;
  }
  function inputMedia(c, detail=false) {
    if (c.video) return `<div class="gallery-film"><video ${detail ? 'controls' : 'muted loop'} playsinline preload="none" poster="${esc(c.poster)}" ${detail ? `src="${esc(c.video)}"` : `data-video="${esc(c.video)}"`}></video>${detail ? '' : '<span class="gallery-film-label">Opus 5.5 · Click to Watch ↗</span>'}</div>`;
    if (c.animation) return `<div class="gallery-film"><img ${detail ? `src="${esc(c.animation)}"` : `src="${esc(c.poster)}" data-animation="${esc(c.animation)}"`} alt="${esc(c.mediaKind)} recorded for Opus 5.5" loading="lazy"><span class="gallery-film-label">Opus 5.5 · ${esc(c.mediaKind)}</span></div>`;
    const images = detail ? c.images : c.images.slice(0,2);
    return `<div class="gallery-inputs ${images.length > 1 ? 'gallery-inputs-pair' : ''}">${images.map((src,i) => `<figure><img src="${esc(src)}" alt="${esc(c.benchmark)} task input ${i+1}" loading="lazy">${images.length > 1 ? `<figcaption>Input ${i+1}</figcaption>` : ''}</figure>`).join('')}</div>`;
  }
  function executionRecords(c) {
    // Every row comes from one audited task. Never fall back to a benchmark's
    // showcase, an ordinal case number, or legacy outcome-only metadata.
    const rows=(c.runIds||[]).map(id=>mediaData.runs[id]);
    if(rows.length!==mediaData.models.length || rows.some(r=>!r || r.taskId!==c.taskId) || new Set(rows.map(r=>r.model)).size!==mediaData.models.length)throw new Error('Task comparison failed validation.');
    return rows;
  }
  const outcomeText = r => r.executionStarted===false ? 'Failed before execution' : r.demoOnly ? 'Demo · Not evaluated' : r.outcome === 'conflicting_records' ? 'Unresolved outcome' : r.outcome === 'scored' ? 'Continuous reward' : r.success ? 'Completed' : ({budget_exhausted:'Budget reached',timeout:'Timed out',runtime_failure:'Runtime error',task_failure:'Not completed',agent_failure:'Agent error',max_iterations:'Round limit reached'}[r.outcome] || 'Not completed');
  const isScenePreview = r => !!(r && r.previewPolicy!=='records-only' && r.mediaKind!=='Animated execution log' && (r.video||r.animation));
  const previewOf = r => r.updatedRecording ? {...r,...r.updatedRecording} : isScenePreview(r) ? r : isScenePreview(r.demoMedia) ? {...r,...r.demoMedia} : r;
  const hasPreview = r => isScenePreview(r.updatedRecording) || isScenePreview(r) || isScenePreview(r.demoMedia);
  const hasApiError = r => r.executionReview?.reason==='api_failure' || r.auditStatus==='provider_interruption';
  const displayOutcome = r => r.updatedRecording ? outcomeText({...r,...r.updatedRecording}) : hasApiError(r) ? 'API error' : r.auditStatus==='task_failure_after_agent_stop' ? 'Not completed' : outcomeText(r);
  const mediaLabel = r => /Saved observation/.test(r.mediaKind||'') ? 'Observed frames' : r.mediaKind||'Recording';
  function sceneMedia(c,r) {
    if(c.previewPolicy==='records-only') return `<span class="gallery-result-preview"><span class="gallery-scene-only">Execution record</span><span class="gallery-result-facts">${r?.modelRounds!=null?`<span><strong>${num(r.modelRounds,0)}</strong>rounds</span>`:''}${r?.elapsedSeconds!=null?`<span><strong>${num(r.elapsedSeconds/60,1)}</strong>minutes</span>`:''}</span></span>`;
    const scene=c.sceneContext;
    if(scene?.kind==='action_log') return `<span class="gallery-result-preview"><span class="gallery-scene-only">Text-based task</span><span class="gallery-result-facts">${r?.modelRounds!=null?`<span><strong>${num(r.modelRounds,0)}</strong>rounds</span>`:''}${r?.elapsedSeconds!=null?`<span><strong>${num(r.elapsedSeconds/60,1)}</strong>minutes</span>`:''}</span></span>`;
    if(!scene || scene.taskId!==c.taskId || !scene.image) return '<span class="gallery-no-recording">Execution record ↗</span>';
    return `<img src="${esc(scene.image)}" alt="Shared task scene from ${esc(scene.model)}; not this model's execution" loading="lazy"><span class="gallery-scene-only" title="Scene from ${esc(scene.model)} on this exact task. This model has a result and log, but no recording.">Scene only</span>`;
  }
  function runMedia(r, detail=false) {
    if(!hasPreview(r)) return '<span class="gallery-no-recording">No matching recording</span>';
    r=previewOf(r);
    if (r.video) return `<video ${detail ? 'controls' : 'muted loop'} playsinline preload="none" poster="${esc(r.poster)}" ${detail ? `src="${esc(r.video)}"` : `data-video="${esc(r.video)}"`} aria-label="${esc(r.model)}: ${esc(r.mediaKind)}"></video>`;
    if (r.animation) return `<img src="${esc(detail ? r.animation : r.poster)}" ${detail ? '' : `data-animation="${esc(r.animation)}" data-poster="${esc(r.poster)}"`} alt="${esc(r.model)}: ${esc(r.mediaKind)}" loading="lazy">`;
    return '<span class="gallery-no-recording">No matching recording</span>';
  }
  const cameraBadge = r => !r.updatedRecording && r.cameraView==='external' ? '<span class="gallery-camera-label">External camera</span>' : '';
  function runTile(c,r,i) {
    const hasMedia=hasPreview(r);
    const logPreview=sceneMedia(c,r);
    return `<button class="gallery-run ${hasMedia?'':'gallery-run-log'} ${hasApiError(r)?'rerun-required':''}" data-case="${esc(c.id)}" data-model="${i}" aria-label="Inspect ${esc(r.model)} on ${esc(c.taskLabel)}: ${esc(displayOutcome(r))}"><span class="gallery-run-header">${esc(short(r.model))}<span aria-hidden="true">↗</span></span><span class="gallery-run-media">${hasMedia?runMedia(r):logPreview}${hasMedia?cameraBadge(r):''}${r.demoMedia?'<span class="gallery-demo-label">Demo · Not scored</span>':''}${hasMedia?`<span class="gallery-run-kind">${esc(mediaLabel(previewOf(r)))}</span>`:''}</span><span class="gallery-run-footer"><span class="gallery-run-outcome ${hasApiError(r)?'pending':r.success?'success':''}"${hasApiError(r)?' title="API-interrupted attempt; counted as unsuccessful"':''}>${esc(displayOutcome(r))}</span></span></button>`;
  }
  function runBudget(r) {
    const values=Object.entries(r.budgets||{}).filter(([k,v])=>budgetNames[k]&&v!=null);
    return values.length ? `<h3>Recorded Budget Ceilings</h3><ul class="study-budget-list">${values.map(([k,v])=>`<li><span>${esc(budgetNames[k])}</span><strong>${num(v)}</strong></li>`).join('')}</ul>` : '';
  }
  function resultTile(c,p,i, visual=false) {
    const s = summary(c,p);
    return `<button class="gallery-model ${visual ? 'gallery-model-visual' : ''} ${s.state}" data-case="${esc(c.id)}" data-model="${i}" style="--model-colour:${colours[i]}" aria-label="Inspect ${esc(p.model)} on ${esc(c.title)}" title="${esc(p.model)}"><span class="gallery-model-name">${esc(short(p.model))}<span aria-hidden="true">↗</span></span>${visual ? overlay(c,p,i) : ''}<span class="gallery-model-value">${esc(s.value)}</span><span class="gallery-model-caption">${esc(s.label)}</span></button>`;
  }
  function caseSlide(c,index,total) {
    const hasModes=!!c.boxComparison;
    c=representation(c);
    let body;
    if (isOverlay(c)) {
      const polygon=c.representation==='polygon';
      const modes=hasModes ? `<div class="gallery-representations" role="group" aria-label="Affordance output"><button data-representation="polygon" data-representation-case="${esc(c.id)}" aria-pressed="${polygon}">Polygon</button><button data-representation="box" data-representation-case="${esc(c.id)}" aria-pressed="${!polygon}">Box + Point</button><span>${polygon?'Trace the functional region':'Locate a region and a contact point'}</span></div>` : '';
      body = `${modes}<div class="gallery-visual-grid">${c.predictions.map((p,i) => resultTile(c,p,i,true)).join('')}</div><p class="gallery-evidence-note"><span class="${polygon?'gallery-mask-swatch':'gallery-ref-line'}"></span> ${polygon?'Reference mask':'Reference'} <span class="gallery-pred-line"></span> ${polygon?'Predicted polygon · Shared crop; click for full image.':'Model prediction'}${c.benchmark.includes('3D') ? ' · Image-plane projection; depth is scored in 3D.' : ''}</p>`;
    } else if (c.predictions.length) {
      body = `<div class="gallery-comparison"><button class="gallery-scene" data-case="${esc(c.id)}" aria-label="Open ${esc(c.title)} task details">${inputMedia(c)}${referenceLabel(c) ? `<span class="gallery-reference">${esc(referenceLabel(c))}</span>` : ''}<span class="gallery-enlarge">Explore Task ↗</span></button><div class="gallery-score-panel"><p class="gallery-panel-label">First-Run Answers${c.domainKey === 'spatial' ? '' : ' · Individual Case'}</p><div class="gallery-model-grid">${c.predictions.map((p,i) => resultTile(c,p,i)).join('')}</div></div></div>`;
    } else {
      const rows = executionRecords(c);
      const filmed=rows.filter(hasPreview);
      body = `<div class="gallery-run-grid gallery-all-models">${rows.map((r,i)=>runTile(c,r,i)).join('')}</div><div class="gallery-run-controls"><p>${c.cameraNote?esc(c.cameraNote):`${rows.length} models · ${filmed.length} visual previews`}</p>${filmed.length?`<button data-playback="${esc(c.id)}" aria-pressed="false">Pause previews</button>`:''}</div>`;
    }
    return `<article class="gallery-slide" data-gallery-id="${esc(c.id)}" data-domain="${esc(c.domainKey)}" role="group" aria-roledescription="slide" aria-label="Case ${index+1} of ${total}: ${esc(c.benchmark)} — ${esc(c.title)}"><div class="gallery-slide-top"><div><p class="gallery-kicker">${esc(c.benchmark)}</p><h3>${esc(c.title)}</h3></div><button class="gallery-detail-button" data-case="${esc(c.id)}">Compare <span aria-hidden="true">↗</span></button></div>${c.shortTask.trim().toLowerCase()!==c.title.trim().toLowerCase()?`<p class="gallery-task">${esc(c.shortTask)}</p>`:''}${body}</article>`;
  }
  function comparisonOverview(c) {
    const tile=html=>html.replace(/data-case="[^"]*" data-model="(\d+)"/, 'data-detail-model="$1"');
    if(!c.predictions.length) return `${c.cameraNote?`<p class="gallery-camera-note">${esc(c.cameraNote)}</p>`:''}<div class="gallery-run-grid gallery-all-models">${executionRecords(c).map((r,i)=>tile(runTile(c,r,i))).join('')}</div>`;
    if(isOverlay(c)) return `<div class="gallery-visual-grid">${c.predictions.map((p,i)=>tile(resultTile(c,p,i,true))).join('')}</div><p class="gallery-evidence-note">Green: reference · Colored outline: prediction</p>`;
    return `<div class="gallery-comparison"><div>${inputMedia(c)}<div class="gallery-reference">${esc(referenceLabel(c))}</div></div><div class="gallery-model-grid">${c.predictions.map((p,i)=>tile(resultTile(c,p,i))).join('')}</div></div>`;
  }
  function taskInstructions(c) {
    if (!c.predictions.length) return '';
    const original = c.originalInstruction;
    return `<details class="gallery-method-note"><summary>${esc(c.instructionKind || 'Original instruction')}</summary><p class="study-original-instruction">${esc(c.question)}</p>${original ? `<a class="study-download" href="${esc(original.url)}" download>Download original evaluation prompts (JSON) ↓</a>` : ''}</details>`;
  }
  const content = document.getElementById('study-dialog-content');
  async function renderModelDetail(c,index) {
    c=representation(c);
    const staticCase = c.predictions.length > 0;
    const list = staticCase ? c.predictions : executionRecords(c);
    const p = list[index] || list[0];
    const selection = document.getElementById('gallery-model-detail');
    if (!p || !selection) return;
    content.querySelector('#gallery-comparison-overview').hidden=true;
    content.querySelector('.gallery-model-switcher').hidden=false;
    content.querySelector('[data-back-comparison]').hidden=false;
    selection.hidden=false;
    dialog.querySelectorAll('[data-detail-model]').forEach(b => b.setAttribute('aria-pressed',String(Number(b.dataset.detailModel) === index)));
    const switcher=content.querySelector('.gallery-model-switcher');
    const active=switcher.querySelector('[aria-pressed="true"]');
    if(active && switcher.scrollWidth>switcher.clientWidth){
      switcher.scrollLeft+=active.getBoundingClientRect().left-switcher.getBoundingClientRect().left-(switcher.clientWidth-active.offsetWidth)/2;
    }
    if (staticCase) {
      const s = summary(c,p);
      const polygon=c.representation==='polygon';
      selection.innerHTML = `<h3>${esc(p.model)}</h3><div class="gallery-detail-comparison"><div>${isOverlay(c) ? overlay(c,p,index,true) : inputMedia(c,true)}</div><div><p class="gallery-detail-metric ${s.state}"><strong>${esc(s.value)}</strong>${esc(s.label)}</p>${polygon ? '<p>Green: reference region. Colored contour: predicted polygon.</p>' : ''}<details ${polygon?'':'open'}><summary>${polygon?'Polygon coordinates':'Recorded Answer'}</summary><pre>${esc(JSON.stringify(p.answer,null,2))}</pre>${c.reference != null ? `<h4>Reference</h4><pre>${esc(JSON.stringify(c.reference,null,2))}</pre>` : ''}</details></div></div>${!polygon && Object.keys(p.metrics || {}).length ? `<h3>Case Metrics</h3><div class="study-table-wrap"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${Object.entries(p.metrics).map(([key,value]) => `<tr><td>${esc(key)}</td><td>${esc(compact(value))}</td></tr>`).join('')}</tbody></table></div>` : ''}`;
    } else {
      selection.querySelectorAll('video').forEach(v=>v.pause());
      if (p.updatedRecording) {
        const u=p.updatedRecording;
        selection.innerHTML=`<div class="gallery-detail-run-heading"><h3>${esc(p.model)}</h3><span class="study-outcome">${esc(displayOutcome(p))}</span></div><div class="gallery-detail-film">${runMedia(p,true)}</div><p class="recording-caption">Updated recording · 24 October · Robot head camera</p><div class="recording-facts"><span><strong>${num(u.playbackSeconds,1)} s</strong> playback</span><span><strong>${num(u.elapsedSeconds/60,1)} min</strong> episode time</span><span><strong>${num(u.frameCount,0)}</strong> recorded frames</span></div><p class="study-media-note">${esc(u.mediaNote)} <a href="${esc(u.provenance)}">Recording details ↗</a></p><details class="recording-original"><summary>Original evaluation record</summary><p>${esc(outcomeText({...p,updatedRecording:null}))} · ${num(p.modelRounds,0)} rounds · ${num(p.elapsedSeconds/60,1)} min${p.tokens==null?'':` · ${num(p.tokens,0)} tokens`}</p><p>The paper’s scores and execution analysis use this earlier episode.</p><a href="${esc(p.detailUrl)}" download>Download original execution record ↓</a></details>`;
        return;
      }

      selection.innerHTML = `<div class="gallery-detail-run-heading"><h3>${esc(p.model)}</h3><span class="study-outcome">${p.demoMedia?"Original evaluation · ":""}${esc(displayOutcome(p))}</span></div>${hasPreview(p)?`<div class="gallery-detail-film">${p.demoMedia?'<span class="gallery-demo-badge">Same-task demo · Not scored</span>':''}${runMedia(p,true)}${cameraBadge(p)}</div>${p.observationPreview?`<p class="gallery-camera-note"><a href="${esc(p.observationPreview.image)}" target="_blank" rel="noopener">View the robot’s recorded observation · Round ${num(p.observationPreview.round,0)} ↗</a></p>`:''}${previewOf(p).mediaNote?`<details class="study-media-note"><summary>${esc(previewOf(p).mediaKind)}</summary><p>${esc(previewOf(p).mediaNote)}</p></details>`:""}`:`<div class="gallery-detail-film gallery-context-film">${sceneMedia(c,p)}</div><p class="study-media-note">${c.previewPolicy==='records-only'?`Results and decisions are available below.${p.video?` <a href="${esc(p.video)}" download>Download original recording ↓</a>`:''} The archived footage has rendering noise.`:c.sceneContext?.kind==='action_log'?'This text-based task is represented by its recorded results and decisions.':`Task scene from ${esc(c.sceneContext?.model)}. This model's result and log are shown below; no execution recording was supplied.`}</p>`}<p class="study-meta">${num(p.modelRounds,0)} decision rounds · ${num(p.elapsedSeconds,1)} s elapsed${p.tokens==null?'':` · ${num(p.tokens,0)} tokens`}</p>${p.note?`<p>${esc(p.note)}</p>`:''}<div class="gallery-rounds" role="status">Loading recorded rounds…</div><details><summary>Task Identity &amp; Settings</summary><p><code>${esc(c.taskId)}</code></p><p>${esc(p.protocol)}</p>${runBudget(p)}<p>Report SHA-256: <code>${esc(p.sourceReportSha256)}</code></p><a href="${esc(p.detailUrl)}" download>Download execution record ↓</a></details>`;
      const rounds=selection.querySelector('.gallery-rounds');
      try {
        const response=await fetch(p.detailUrl+'?v=toolbox-20261031');if(!response.ok)throw Error();const d=await response.json();
        if(!rounds.isConnected)return;
        // API interruptions count as final failed attempts, including those after
        // useful actions. A budget stop or generated-code error is not an API failure.
        if(hasApiError(p)) {
          const before=p.executionReview.phase==='before_actions';
          selection.querySelector('.study-outcome').textContent='API error';
          const notice=`<div class="gallery-api-failure"><h4>API-interrupted attempt</h4><p>${before?'The initial API request failed before any action was generated.':'An API request failed after execution began.'} This attempt is counted as unsuccessful.</p><p><code>${esc(c.taskId)}</code></p><div class="gallery-api-facts"><a href="data/report/case-api-errors.csv" download>Download error inventory ↓</a><a href="${esc(p.detailUrl)}" target="_blank" rel="noopener">Original record ↗</a></div><details><summary>Recorded error</summary><pre>${esc(p.executionReview.error)}</pre></details></div>`;
          if(before) {
            selection.querySelectorAll('.study-meta').forEach(e=>e.remove());
            rounds.removeAttribute('role');rounds.innerHTML=notice;return;
          }
          rounds.insertAdjacentHTML('beforebegin',notice);
        }
        if(p.detailedTraceAvailable===false){rounds.remove();return;}
        if(p.frames?.length && p.mediaKind==='Saved observations') {
          const box=document.createElement('div');box.className='gallery-observation-picker';
          box.innerHTML=`<label>Observation <output>1</output> / ${p.frames.length}<input type="range" min="0" max="${p.frames.length-1}" value="0" aria-label="Select input observation"></label>`;
          selection.querySelector('.gallery-detail-film').after(box);
          box.querySelector('input').oninput=e=>{const f=p.frames[Number(e.target.value)];selection.querySelector('.gallery-detail-film img').src=f.image;box.querySelector('output').textContent=f.round;};
        }
        const useful=d.steps.filter((s,i)=>s.error||s.success||i===0||i===d.steps.length-1);
        rounds.removeAttribute('role');
        rounds.innerHTML=`<h4>Recorded Decisions</h4>${d.steps.length?`<p>${d.steps.length} rounds · ${d.steps.filter(s=>s.error).length} rounds with errors</p><div class="gallery-round-list">${useful.map(s=>`<details><summary>Round ${s.round}${s.at==null?'':` · ${num(s.at,1)} s`} — ${esc(s.title)}</summary>${s.code?`<pre>${esc(s.code)}</pre>`:''}<pre>${esc(JSON.stringify(s.feedback||{},null,2))}</pre>${s.error?`<pre>${esc(typeof s.error==='string'?s.error:JSON.stringify(s.error,null,2))}</pre>`:''}</details>`).join('')}</div>`:'<p>No per-round trace was recorded.</p>'}`;
      } catch(e){if(rounds.isConnected)rounds.innerHTML=`<a href="${esc(p.detailUrl)}">Open execution record ↗</a>`;}

    }
  }
  async function open(id,button,modelIndex=null) {
    trigger = button; openId = id; selectedModel = modelIndex;
    content.innerHTML='<h2 id="study-dialog-title">Loading Case…</h2>';
    if (!dialog.open) dialog.showModal();
    document.body.classList.add('study-modal-open');
    window.ArenaMedia.refresh();
    try {
      await load();
      const found = records.get(id); if (!found) throw new Error('Case not found.');
      const c=representation(found);
      if (!dialog.open || openId !== id) return;
      const list = c.predictions.length ? c.predictions : executionRecords(c);
      selectedModel=modelIndex;
      const budget = Object.entries(c.predictions.length ? c.budgets||{} : {}).filter(([k,v]) => budgetNames[k] && v!=null).map(([k,v])=>`<li><span>${budgetNames[k]}</span><strong>${num(v)}</strong></li>`).join('');
      content.innerHTML=`<p class="eyebrow">${esc(c.domain)} · ${esc(c.benchmark)}</p><h2 id="study-dialog-title">${esc(c.title)}</h2>${c.shortTask.trim().toLowerCase()!==c.title.trim().toLowerCase()?`<p class="study-question">${esc(c.shortTask)}</p>`:''}<button class="gallery-back" data-back-comparison ${modelIndex===null?'hidden':''}>← All ten models</button><div class="gallery-model-switcher" role="group" aria-label="Select model" ${modelIndex===null?'hidden':''}>${list.map((p,i)=>`<button data-detail-model="${i}" aria-pressed="${i===selectedModel}">${esc(short(p.model))}</button>`).join('')}</div><div id="gallery-comparison-overview" ${modelIndex===null?'':'hidden'}><p class="gallery-comparison-hint">Select a model to inspect its prediction or execution.</p>${comparisonOverview(c)}</div><div id="gallery-model-detail" ${modelIndex===null?'hidden':''}></div>${isOverlay(c) && c.representation !== 'polygon' ? '<p class="gallery-evidence-note">Dashed mint: reference. Solid colour: prediction. Overlays preserve the original image coordinates.</p>' : ''}${c.comparisonFigure ? `<a class="study-download" href="${esc(c.comparisonFigure)}" target="_blank" rel="noopener">Open Full Comparison Figure ↗</a>` : ''}${budget ? `<h3>${c.predictions.length ? '' : 'Opus 5.5 · '}Recorded Budget Ceilings</h3><ul class="study-budget-list">${budget}</ul>` : ''}${taskInstructions(c)}${c.options?.length ? `<ul>${c.options.map(v=>`<li>${esc(compact(v))}</li>`).join('')}</ul>` : ''}<details class="gallery-method-note"><summary>Evaluation details</summary><p class="study-protocol">${esc(c.protocolNote)}</p></details><a class="study-download" id="study-json-download">Download Case Record (JSON) ↓</a>`;
      if(selectedModel!==null) renderModelDetail(c,selectedModel);
      dialog.scrollTop=0;
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl=URL.createObjectURL(new Blob([JSON.stringify({...c,galleryMetadata:metadata[c.id],recordings:c.predictions.length ? undefined : executionRecords(c)},null,2)],{type:'application/json'}));
      const download=document.getElementById('study-json-download'); download.href=currentUrl; download.download=`${c.id}.json`;
    } catch(error) {content.innerHTML=`<h2 id="study-dialog-title">Case Details Unavailable</h2><p>${esc(error.message)} Please reload and try again.</p>`;}
  }
  document.addEventListener('click',event=>{
    if(event.target.closest('[data-back-comparison]')) {
      content.querySelectorAll('video').forEach(v=>v.pause());
      content.querySelector('#gallery-model-detail').hidden=true;
      content.querySelector('.gallery-model-switcher').hidden=true;
      content.querySelector('[data-back-comparison]').hidden=true;
      content.querySelector('#gallery-comparison-overview').hidden=false;
      selectedModel=null; dialog.scrollTop=0; window.ArenaMedia.refresh(); return;
    }
    const mode=event.target.closest('[data-representation]');
    if(mode){
      const id=mode.dataset.representationCase;
      if(mode.dataset.representation==='box')boxCases.add(id);else boxCases.delete(id);
      const slot=mode.closest('.gallery-slot');
      const index=shown.findIndex(c=>c.id===id);
      if(slot&&index>=0){slot.innerHTML=caseSlide(shown[index],index,shown.length);slot.querySelector(`[data-representation="${mode.dataset.representation}"]`)?.focus({preventScroll:true});}
      return;
    }
    const playback=event.target.closest('[data-playback]');
    if(playback){const id=playback.dataset.playback;const playing=!pausedCases.has(id)&&(manualPlay.has(id)||!reduced.matches);if(playing){pausedCases.add(id);manualPlay.delete(id);}else{pausedCases.delete(id);manualPlay.add(id);}updateActiveMedia();}
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
  const atlasMode=document.body.classList.contains('case-atlas');
  const filters=[...document.querySelectorAll('[data-study-filter]')];
  const search=document.getElementById('study-search');
  let active=filters.find(b=>b.getAttribute('aria-pressed')==='true')?.dataset.studyFilter || 'highlights';
  const gallery=document.createElement('div'); gallery.className='case-gallery';
  gallery.innerHTML=`<div class="gallery-track" tabindex="0" role="region" aria-roledescription="carousel" aria-label="Case comparisons; use left and right arrow keys to browse"></div><div class="gallery-navigation"><p class="gallery-help"><span aria-hidden="true">↔</span> Scroll to Explore · Click a Model for Details</p><div class="gallery-position"><output id="gallery-position" aria-live="polite"></output><button data-gallery-step="-1" aria-label="Previous case">←</button><button data-gallery-step="1" aria-label="Next case">→</button></div></div><input class="gallery-scrubber" type="range" min="1" max="15" value="1" aria-label="Choose a case"><p class="gallery-empty" hidden>No matching cases. Try another search.</p>`;
  gallery.hidden=atlasMode;
  grid.before(gallery);
  if(atlasMode) {
    const empty=document.createElement('p');empty.id='atlas-empty';empty.hidden=true;
    empty.textContent='No matching cases. Try a different task or capability.';grid.after(empty);
    const motion=document.querySelector('[data-atlas-motion]');
    motion?.addEventListener('click',()=>{const play=motion.getAttribute('aria-pressed')==='true';
      window.ArenaMedia.toggle(grid.querySelectorAll('video,img[data-animation]'),play);
      motion.setAttribute('aria-pressed',String(!play));motion.textContent=play?'Pause previews':'Play previews';});
  }
  const track=gallery.querySelector('.gallery-track');
  const position=gallery.querySelector('#gallery-position');
  const scrubber=gallery.querySelector('.gallery-scrubber');
  let allCases=[], shown=[], slides=[], current=0, isVisible=false, ticking=false;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  let observer;
  // Cover every supplied benchmark/subset; retain paper cases and contrasting tasks.
  const showcaseIds = ["showcase-capx_behavior1k", "showcase-calvin", "W1-w1_vlm_motion_real-01", "W2-w2_mindcube-01", "W5-w5_umd-03", "showcase-alfworld_visual", "W4-calvin-03", "showcase-robocasa", "W1-w1_tracespatial_3d-01", "W2-w2_3dsrbench-01", "W5-w5_ragnet-02", "showcase-discoveryworld", "showcase-capx_libero_pro", "W1-w1_influx-01", "W2-w2_bop_ask-01", "W5-w5_reasonaff-03", "showcase-alfred_official_visual", "W1-w1_mapfree-01", "W2-w2_mmsi_bench-01", "W5-w5_umd-01", "W1-w1_multispa-01", "W2-w2_robospatial-01", "showcase-humanclaw", "showcase-capx_robosuite", "W1-w1_multispa-02", "W2-w2_vsi_bench-01", "showcase-scienceworld_text", "showcase-cliport", "W1-w1_tracespatial_2d-01", "W2-w2_vsi_bench-02", "showcase-virtualhome_symbolic", "showcase-maniskill", "W1-w1_vlm_depth-01", "showcase-rlbench", "W1-w1_vlm_focal-01", "W1-w1_vlm_intrinsics-01", "W1-w1_vlm_motion_camera-01", "showcase-robocasa365", "W1-w1_vlm_motion_camera_large-01", "showcase-robotwin2", "W1-w1_vlm_motion_fixed-01", "showcase-robowits", "W1-w1_vlm_pose-01", "showcase-vimabench", "W1-w1_vlm_reference-01", "showcase-vlabench", "W1-w1_vlm_reference-03", "W1-w1_vlm_size-01", "W4-calvin-01", "W4-capx_behavior1k-02"];
  const showcase = new Set(showcaseIds);
  const aliases=c=>showcaseIds.filter(id=>id===c.id||records?.get(id)?.id===c.id); const priority=c=>Math.min(200,...aliases(c).map(id=>showcaseIds.indexOf(id)));
  function hydrate(index) {
    for(let i=Math.max(0,index-1);i<=Math.min(slides.length-1,index+1);i++) {
      const slide=slides[i]; if(!slide || slide.dataset.ready) continue;
      slide.innerHTML=caseSlide(shown[i],i,shown.length);slide.dataset.ready='true';
    }
  }
  function updateActiveMedia() {
    slides.forEach((slide,i)=>{
      const id=shown[i].id;
      const playing=!pausedCases.has(id)&&(manualPlay.has(id)||!reduced.matches);
      const visible=i===current && isVisible && !document.hidden && !dialog.open;
      slide.querySelectorAll('video, [data-animation]').forEach(media=>{
        window.ArenaMedia.enable(media,visible && playing,manualPlay.has(id));
      });
      const toggle=slide.querySelector('[data-playback]');
      if(toggle){toggle.textContent=playing?'Pause previews':'Play previews';toggle.setAttribute('aria-pressed',String(!playing));}
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
    shown=allCases.filter(c=>(active==='all'||active==='highlights'&&aliases(c).length>0||c.domainKey===active)&&(!query||[c.benchmark,c.title,c.shortTask,c.question,...c.predictions.map(p=>p.model),...(c.runIds||[]).map(id=>{const r=mediaData.runs[id];return `${r.model} ${r.taskLabel} ${r.instruction}`;})].join(' ').toLowerCase().includes(query))).sort((a,b)=>priority(a)-priority(b));
    if(atlasMode){
      const ids=new Set(shown.map(c=>c.id));
      grid.querySelectorAll('.study-card').forEach(card=>{card.hidden=!ids.has(card.querySelector('[data-case]').dataset.case);});
      const count=document.getElementById('study-count');if(count)count.textContent=`${shown.length} ${shown.length===1?'case':'cases'}${shown.length?' · Select a card to compare':''}`;
      document.getElementById('atlas-empty').hidden=!!shown.length;
      return;
    }
    track.querySelectorAll('video').forEach(v=>v.pause());
    track.innerHTML=shown.map((c,i)=>`<div class="gallery-slot" aria-label="${esc(c.benchmark)}: ${esc(c.title)}"><p class="gallery-loading">${i+1} · ${esc(c.benchmark)}</p></div>`).join('');
    slides=[...track.children];current=0;hydrate(0);track.scrollLeft=0;
    scrubber.max=Math.max(1,shown.length);scrubber.value=1;
    gallery.querySelector('.gallery-empty').hidden=!!shown.length;
    track.hidden=!shown.length;gallery.querySelector('.gallery-navigation').hidden=!shown.length;scrubber.hidden=shown.length<2;
    const count=document.getElementById('study-count');if(count)count.textContent=`${shown.length} case views`;
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
  document.addEventListener('visibilitychange',()=>{if(document.hidden){window.ArenaMedia.refresh();}else updateActiveMedia();});
  window.addEventListener('resize',()=>go(current,false));
  load().then(cases=>{allCases=cases;grid.hidden=!atlasMode;gallery.hidden=atlasMode;if(!atlasMode)gallery.classList.add('is-ready');applyFilter();const target=new URLSearchParams(location.search).get('case');const index=shown.findIndex(c=>c.id===target);if(index>=0){if(atlasMode)open(target,null);else go(index,false);}}).catch(error=>{
    gallery.innerHTML=`<p class="study-protocol">${esc(error.message)} <a href="case-atlas.html">Browse the Case Atlas ↗</a></p>`;
  });
})();
