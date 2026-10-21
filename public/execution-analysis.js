(() => {
  'use strict';
  const root = document.getElementById('execution-analysis');
  if (!root) return;
  const esc = x => String(x ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = n => n == null ? '—' : Number(n).toLocaleString('en-US',{maximumFractionDigits:1});
  const time = n => {if(n==null)return '—';const sec=Math.round(n);return sec<60?`${sec}s`:`${Math.floor(sec/60)}m ${sec%60}s`;};
  const tokens = n => n == null ? '—' : n>=1e6 ? `${(n/1e6).toFixed(2)}M` : `${(n/1000).toFixed(1)}k`;
  const names = {'Claude Fable 5.1':'Fable 5.1','Claude Opus 5.5':'Opus 5.5','Qwen 3.8 Max':'Qwen Max','Qwen 3.5 397B-A17B':'Qwen 397B','Qwen 3.5 27B':'Qwen 27B','DeepSeek 4.1 Flash-1017':'DeepSeek-1017'};
  const short = r => names[r.model] || r.model;
  const stopNames = {max_total_tokens:'Token limit',max_agent_iterations:'Round limit',primitive_calls:'Action-call limit',max_env_steps:'Step limit',max_code_turns:'Code-round limit',max_primitive_calls:'Tool-call limit',max_rounds:'Round limit'};
  const cache = new Map(), panes = {};
  let data,story,generation=0,pendingStory,pendingModels;
  async function get(url){if(!cache.has(url))cache.set(url,fetch(url+'?v=media-20261022').then(r=>{if(!r.ok)throw Error();return r.json();}).catch(e=>{cache.delete(url);throw e;}));return cache.get(url);}
  const limitsOf = r => ({...r.budgets,
    max_rounds: r.loopBudgets?.max_turns ?? r.budgets.max_rounds ?? r.budgets.max_code_turns ?? r.budgets.max_agent_iterations,
    timeout_seconds: r.loopBudgets?.timeout_seconds ?? r.budgets.trial_timeout_seconds ?? r.budgets.timeout_seconds
  });
  const pause=()=>root.querySelectorAll('video').forEach(v=>v.pause());
  // Some exact-task records retain round feedback without primitive-call traces.
  // Normalize that representation; never infer unseen actions or intermediate goals.
  function normalizeTrace(detail, run) {
    const d = {...run, ...detail};
    d.steps = (detail.steps || []).map(s => ({
      ...s,
      calls: s.calls || [],
      event: s.event || {
        kind: s.error ? 'error' : s.success ? 'success' : s.budgetStop ? 'budget' : 'decision',
        label: s.title || 'Decision round',
        detail: s.error ? 'Execution error returned; inspect the feedback below.' :
          s.success ? 'The native evaluator reports task success.' :
          s.budgetStop ? 'An episode limit was reached.' : 'Recorded round feedback is available below.'
      }
    }));
    if (!d.milestones?.length && d.steps.length) {
      const indices = new Set([0, d.steps.length - 1]);
      d.steps.forEach((s, i) => {if (s.error || s.success || s.budgetStop) indices.add(i);});
      d.milestones = [...indices].sort((a,b)=>a-b).map(index => ({
        index, at: d.steps[index].at, label: d.steps[index].event.label,
        detail: d.steps[index].event.detail
      }));
    }
    return d;
  }
  function playbackLabel(){const b=root.querySelector('.exec-play');if(!b)return;const vs=[...root.querySelectorAll('.exec-video')];b.hidden=!vs.length;b.textContent=vs.some(v=>!v.paused)?'Ⅱ Pause':vs.length===2?'▶ Play Both':'▶ Play Clip';}
  const outcome=r=>r.success?'Completed':'Not completed';
  const metric=(value,label,cls='')=>`<div class="${cls}"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`;
  function updateUrl(){const u=new URL(location.href);u.searchParams.set('analysis',story.id);[0,1].forEach(i=>{const r=story.runs.find(r=>r.id===root.querySelector(`[data-pane="${i}"]`)?.dataset.run);if(r)u.searchParams.set(i?'modelB':'modelA',r.modelKey);});u.hash='execution-analysis';history.replaceState(null,'',u);}
  function selectStory(id,models){renderStory(id,models);updateUrl();}
  function renderStory(id,models){
    pause();generation++;story=data.stories.find(s=>s.id===id)||data.stories[0];
    root.querySelectorAll('[data-exec-story]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.execStory===story.id)));
    document.getElementById('exec-library').value=story.id;
    const common=key=>{const v=story.runs.map(r=>limitsOf(r)[key]);return v[0]!=null&&v.every(x=>x===v[0])?v[0]:null;};
    const limits=[common('max_rounds')?`${common('max_rounds')} rounds`:null,common('max_total_tokens')?`${tokens(common('max_total_tokens'))} tokens`:null,common('timeout_seconds')?`${num(common('timeout_seconds')/60)} min`:null].filter(Boolean).join(' · ');
    document.getElementById('exec-stage').innerHTML=`<div class="exec-story-heading"><h3>${esc(story.title)}</h3></div><div class="exec-task"><span>Task</span><p>${esc(story.instruction)}</p></div><div class="exec-compare-toolbar"><button class="exec-play" type="button">▶ Play Both</button></div><div class="exec-compare"><article class="exec-pane" data-pane="0"></article><article class="exec-pane" data-pane="1"></article></div><details class="exec-methods"><summary>Evidence, Settings &amp; Source Records</summary><p>${esc(story.featured ? story.takeaway : `${story.runs.filter(r=>r.success).length} of ${story.runs.length} models complete this task.`)}</p><p><strong>Task:</strong> <code>${esc(story.runs[0].taskId)}</code></p><p>${limits?`Shared ceilings: ${esc(limits)}.`:''} Each model’s settings are listed below. Videos have their own playback clock; round sliders show episode time. Videos are excerpts or saved observations.</p><div class="exec-settings-grid"></div><p><a href="data/execution/identity-audit.json">Task-matching audit ↗</a> <a href="execution-guide.html#${esc(story.id)}">Episode metrics ↗</a></p></details>`;
    const chosen=[0,1].map(i=>story.runs.find(r=>r.modelKey===(models?.[i]||story.default[i]))||story.runs[i]);if(chosen[0].id===chosen[1].id)chosen[1]=story.runs.find(r=>r.id!==chosen[0].id);[0,1].forEach(i=>renderPane(i,chosen[i].id));
    root.querySelector('.exec-play').onclick=()=>{const vs=[...root.querySelectorAll('.exec-video')];window.ArenaMedia.toggle(vs,!vs.some(v=>!v.paused));};
    document.getElementById('exec-announcement').textContent=`${story.benchmark}. ${story.taskLabel}. ${story.runs.length} matching models.`;
  }
  function renderSettings(){
    const area=root.querySelector('.exec-settings-grid');if(!area)return;
    area.innerHTML=[0,1].map(i=>{const r=panes[i];if(!r||!story.runs.some(x=>x.id===r.id))return '';const b=limitsOf(r);return `<div><h4>${esc(r.model)}</h4><p>Observation setting: ${esc(r.protocol)}</p><dl>${[['Rounds',b.max_rounds],['Tokens',tokens(b.max_total_tokens)],['Time ceiling',time(b.timeout_seconds)],['Tool calls',b.max_primitive_calls],['Environment steps',b.max_env_steps]].filter(([,v])=>v!=null&&v!=='—').map(([k,v])=>`<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join('')}<div><dt>Stop</dt><dd>${esc(r.stopLimits.length?r.stopLimits.map(x=>stopNames[x]||x).join(', '):outcome(r))}</dd></div><div><dt>Input / output tokens</dt><dd>${tokens(r.stats.promptTokens)} / ${tokens(r.stats.completionTokens)}</dd></div></dl><a href="${esc(r.detailUrl)}" download>Download trace ↓</a><p class="exec-record-id">${esc(r.recordId)}<br>SHA-256 ${esc(r.reportSha256)}</p></div>`;}).join('');
  }
  function finalMetrics(r){
    const s=r.stats,p=r.progress;
    let result=p.label==='Task completion'?(r.success?'Success':'Incomplete'):`${num(p.value)} / ${num(p.total)}`; let resultLabel=p.label;
    let second=s.actionSubmissions!=null?[num(s.actionSubmissions),'Action submissions']:[num(s.rounds),'Decision rounds'];
    if(story.id==='precision'){result=num(s.nativeSteps);resultLabel='Native steps';second=[`${num(s.finalGoalDistanceCm)} cm`,'Final goal distance'];}
    if(['time','horizon'].includes(story.id)){result=num(s.rounds);resultLabel='Decision rounds';second=[tokens(s.tokens),'Tokens used'];}
    return metric(result,resultLabel,r.success?'is-success':'is-incomplete')+metric(...second)+metric(time(s.elapsedSeconds),'Episode time');
  }
  function episodeLimits(r){
    const b=limitsOf(r);
    const limits=[b.max_rounds?`${num(b.max_rounds)} rounds`:null,b.max_total_tokens?`${tokens(b.max_total_tokens)} tokens`:null,b.timeout_seconds?`${num(b.timeout_seconds/60)} min`:null].filter(Boolean);
    const ending=r.success?'Goal satisfied':r.stopLimits.length?r.stopLimits.map(x=>stopNames[x]||x.replaceAll('_',' ')).join(', '):'Task incomplete';
    return `<div class="exec-limit-strip"><span>Ceilings: ${esc(limits.join(' · '))}</span><strong>Ended: ${esc(ending)}</strong></div>`;
  }
  function actionSummary(step){
    const calls=step.calls||[];
    const action=calls.find(c=>/^(submit_|apply_|move_|grasp_|place_|execute_)/.test(c.name))||calls[0];
    if(!action)return '';
    const entries=Object.entries(action.arguments||{}).filter(([k,v])=>!['evidence_handles','query','prompt','context'].includes(k)&&v!=null).slice(0,3);
    if(!entries.length)return '';
    const val=v=>Array.isArray(v)?v.map(x=>typeof x==='number'?Number(x.toFixed(3)):x).join(', '):typeof v==='object'?JSON.stringify(v):String(v);
    return `<dl class="exec-parameters">${entries.map(([k,v])=>`<div><dt>${esc(k.replaceAll('_',' '))}</dt><dd>${esc(val(v)).slice(0,180)}</dd></div>`).join('')}</dl>`;
  }
  async function renderPane(i,id){
    const r=story.runs.find(r=>r.id===id),pane=root.querySelector(`[data-pane="${i}"]`);if(!r)return;
    pane.querySelector('video')?.pause();pane.dataset.run=id;const current=generation;delete panes[i];
    const isLog=!r.video||/log/i.test(r.mediaKind);
    const style=window.ArenaModelStyle.get(r.modelKey);pane.style.setProperty('--model-color',style.color);pane.style.setProperty('--model-light',style.light);
    pane.innerHTML=`<header class="exec-model"><div class="exec-model-picker"><label class="sr-only" for="exec-model-${i}">Model ${i?'B':'A'}</label><div><img src="${style.logo}" width="22" height="22" alt=""><select id="exec-model-${i}">${story.runs.map(x=>`<option value="${esc(x.id)}" ${x.id===id?'selected':''}>${esc(short(x))}</option>`).join('')}</select></div></div><span class="exec-outcome ${r.success?'is-success':''}">${outcome(r)}</span></header>${isLog&&(r.animation||r.demoMedia)?`<div class="exec-video-frame">${r.demoMedia?`<video class="exec-demo-video" controls muted loop playsinline preload="metadata" poster="${esc(r.demoMedia.poster)}" src="${esc(r.demoMedia.video)}"></video>`:`<img src="${esc(r.animation)}" alt="${esc(r.model)} original input observations">`}<span class="exec-video-kind">${r.demoMedia?'Same-task demo · Separate from the evaluation below':'Observed frames · Original evaluation'}</span></div>`:isLog?`<div class="exec-log-preview"><span>⌘</span><h4>Action Log</h4><p>${r.video?'This episode has a saved action log.':'No matching video for this execution.'}</p>${r.video?`<a href="${esc(r.video)}" target="_blank" rel="noopener">Open log recording ↗</a>`:''}</div>`:`<div class="exec-video-frame"><video class="exec-video" controls muted playsinline preload="metadata" poster="${esc(r.poster)}" src="${esc(r.video)}" aria-label="${esc(r.model+' — '+story.taskLabel)}"></video><span class="exec-video-kind">${esc(r.mediaKind||'Video excerpt')}</span></div>`}<div class="exec-metrics">${finalMetrics(r)}</div>${episodeLimits(r)}${story.goalLabels&&story.id==='sequence'?`<div class="exec-goal-sequence" aria-label="Final subgoal check">${story.goalLabels.map((label,j)=>`<span class="${j<r.progress.value?'done':''}"><b>${j<r.progress.value?'✓':j+1}</b>${esc(label)}</span>`).join('')}</div>`:''}<div class="exec-trace-loading" role="status">Loading decisions…</div>`;
    pane.querySelector('select').onchange=e=>{const newId=e.target.value;renderPane(i,newId);const other=root.querySelector(`[data-pane="${1-i}"]`);if(other.dataset.run===newId)renderPane(1-i,story.runs.find(x=>x.id!==newId).id);updateUrl();};
    const v=pane.querySelector('video');
    if(v){['play','pause','ended'].forEach(type=>v.addEventListener(type,playbackLabel));}
    try{
      playbackLabel();const d=normalizeTrace(await get(r.detailUrl),r);if(current!==generation||pane.dataset.run!==id)return;panes[i]=d;renderSettings();
      const slot=pane.querySelector('.exec-trace-loading');slot.className='exec-decisions';slot.removeAttribute('role');
      if(!d.steps.length){slot.innerHTML='<p>Final outcome available. No per-round trace.</p>';return;}
      const milestones=d.milestones||[];
      const first=milestones.find(m=>d.steps[m.index].event.kind==='error')||milestones[0];
      const next=milestones.find(m=>m.index>(first?.index??-1)&&((d.steps[m.index].score??0)>0||(d.steps[m.index].goalsMet??0)>0))||milestones.find(m=>m.index>(first?.index??-1));
      const shown=[...new Map([first,next,milestones.at(-1)].filter(Boolean).map(m=>[m.index,m])).values()];
      slot.innerHTML=`<div class="exec-decision-heading"><h4>Execution Timeline</h4><span class="exec-round-position"></span></div><div class="exec-timeline" aria-label="Click a decision round">${d.steps.map((s,j)=>`<button data-moment="${j}" class="${s.event.kind}" aria-label="Round ${s.round}: ${esc(s.event.label)}" title="Round ${s.round}: ${esc(s.event.label)}">${d.steps.length<=12?s.round:''}</button>`).join('')}</div><div class="exec-scrubber"><button class="exec-prev" aria-label="Previous decision for ${esc(r.model)}">←</button><label class="sr-only" for="exec-round-${i}">Decision for ${esc(r.model)}</label><input id="exec-round-${i}" type="range" min="0" max="${d.steps.length-1}" step="1" value="0"><button class="exec-next" aria-label="Next decision for ${esc(r.model)}">→</button></div><div class="exec-chart"></div><div class="exec-milestones">${shown.map(m=>`<button type="button" data-moment="${m.index}" title="${esc(m.detail||m.label)}"><span>Round ${d.steps[m.index].round} · ${time(m.at)}</span><strong>${esc(m.label)}</strong></button>`).join('')}</div><div class="exec-selected-event" aria-live="polite"></div><div class="exec-evidence-content"></div>`;
      slot.querySelectorAll('[data-moment]').forEach(b=>b.onclick=()=>selectStep(i,Number(b.dataset.moment)));
      const slider=slot.querySelector('input');slider.oninput=()=>selectStep(i,Number(slider.value));
      slot.querySelector('.exec-prev').onclick=()=>selectStep(i,Math.max(0,Number(slider.value)-1));
      slot.querySelector('.exec-next').onclick=()=>selectStep(i,Math.min(d.steps.length-1,Number(slider.value)+1));
      selectStep(i,shown.length>1?shown[1].index:0);
    }catch{if(current===generation&&pane.dataset.run===id){const slot=pane.querySelector('.exec-trace-loading,.exec-decisions');if(slot)slot.innerHTML='<p>Trace unavailable. <button class="exec-retry">Retry</button></p>';pane.querySelector('.exec-retry')?.addEventListener('click',()=>renderPane(i,id));}}
  }
  function selectStep(i,j){
    const d=panes[i],pane=root.querySelector(`[data-pane="${i}"]`),s=d.steps[j];if(!s)return;
    pane.querySelector('input[type=range]').value=j;pane.querySelector('input[type=range]').setAttribute('aria-valuetext',`Round ${s.round}, ${time(s.at)} elapsed`);pane.querySelector('.exec-round-position').textContent=`Round ${s.round} of ${d.steps.at(-1).round} · ${time(s.at)}`;pane.querySelector('.exec-prev').disabled=j===0;pane.querySelector('.exec-next').disabled=j===d.steps.length-1;
    pane.querySelectorAll('[data-moment]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.moment)===j)));
    const e=s.event,atEnd=j===d.steps.length-1,detail=atEnd?(d.verifier?.message||outcome(d)):s.currentSubgoal||e.detail;
    const progress=s.goalsMet!=null?`${s.goalsMet}/${s.goalsTotal} conditions`:s.score!=null?`Score ${num(s.score)}`:'';
    pane.querySelector('.exec-selected-event').innerHTML=`<div class="exec-event-meta"><span>Round ${s.round} · ${time(s.at)}</span><strong class="${e.kind}">${esc(atEnd?'Final check':progress||e.label)}</strong></div><p>${esc(detail||s.title)}</p>`;
    const chart=pane.querySelector('.exec-chart');
    const hasGoals=d.steps.some(x=>x.goalsMet!=null),hasScore=d.steps.some(x=>x.score!=null);
    if(hasGoals||hasScore){
      const key=hasGoals?'goalsMet':'score',total=hasGoals?d.steps.find(x=>x.goalsTotal)?.goalsTotal||1:d.progress.total;
      const x=k=>34+k/Math.max(1,d.steps.length-1)*438,y=v=>85-v/Math.max(total,0.0001)*65;
      let path='',connected=false;d.steps.forEach((step,k)=>{const val=step[key];if(val==null){connected=false;return;}path+=connected?` H${x(k)} V${y(val)}`:` M${x(k)},${y(val)}`;connected=true;});
      chart.innerHTML=`<div class="exec-chart-label"><span>${hasGoals?'Verified goal conditions':'Task score'}</span><strong>${progress||'No value at this round'}</strong></div><svg viewBox="0 0 500 116" role="img" aria-label="${hasGoals?'Goal conditions':'Task score'} by decision round. Select a point or use the round slider."><line x1="34" y1="20" x2="472" y2="20" class="exec-grid-line"/><line x1="34" y1="85" x2="472" y2="85" class="exec-grid-line"/><text x="24" y="24" text-anchor="end">${num(total)}</text><text x="24" y="89" text-anchor="end">0</text><text x="34" y="108">Round ${d.steps[0].round}</text><text x="472" y="108" text-anchor="end">${d.steps.at(-1).round}</text><path class="exec-plot" d="${path}"/>${d.steps.map((step,k)=>step[key]==null?'':`<circle cx="${x(k)}" cy="${y(step[key])}" r="${k===j?5:2.5}" class="exec-progress-point"><title>Round ${step.round}: ${num(step[key])}/${num(total)}</title></circle>`).join('')}<line x1="${x(j)}" y1="12" x2="${x(j)}" y2="91" class="exec-cursor"/></svg>`;
      chart.querySelector('svg').onclick=e=>{const rect=e.currentTarget.getBoundingClientRect(),pos=(e.clientX-rect.left)/rect.width*500;selectStep(i,Math.max(0,Math.min(d.steps.length-1,Math.round((pos-34)/438*(d.steps.length-1)))));};
    }else chart.innerHTML='';
    const pre=x=>`<pre>${esc(typeof x==='string'?x:JSON.stringify(x,null,2))}</pre>`;
    pane.querySelector('.exec-evidence-content').innerHTML=`<details><summary>Code, Feedback &amp; Tool Calls · Round ${s.round}</summary>${actionSummary(s)}${s.code?pre(s.code):''}${s.stdout?pre(s.stdout):''}${s.error?pre(s.error):''}${Object.keys(s.feedback||{}).length?pre(s.feedback):''}${s.calls.length?`<h5>${s.calls.length} Tool Calls</h5>${s.calls.map((c,k)=>`<details><summary>${k+1}. ${esc(c.name)} · ${c.ok?'Accepted':'Returned error'}</summary>${pre({arguments:c.arguments,output:c.output,...c.error?{error:c.error}:{}})}</details>`).join('')}`:''}</details>`;
  }
  async function init(){
    try{
      data=await get('data/execution/complete-index.json');
      document.getElementById('exec-counts').textContent=`${data.stories.length} Tasks · ${data.modelCount} Models`;
      document.getElementById('exec-questions').innerHTML=data.stories.filter(s=>s.featured).map(s=>`<button type="button" data-exec-story="${s.id}" aria-pressed="false">${esc(s.label)}</button>`).join('');
      root.querySelectorAll('[data-exec-story]').forEach(b=>b.onclick=()=>selectStory(b.dataset.execStory));
      const chooser=document.createElement('div');chooser.className='exec-library-controls';chooser.innerHTML=`<label for="exec-library">Choose a Task</label><select id="exec-library">${[...new Set(data.stories.map(s=>s.benchmark))].map(b=>`<optgroup label="${esc(b)}">${data.stories.filter(s=>s.benchmark===b).map(s=>`<option value="${s.id}">${esc(s.benchmark+' · '+(s.taskLabel||s.title))} (${s.runs.length} models)</option>`).join('')}</optgroup>`).join('')}</select>`;document.getElementById('exec-questions').after(chooser);
      chooser.querySelector('select').onchange=e=>selectStory(e.target.value);
      const params=new URLSearchParams(location.search);renderStory(pendingStory||params.get('analysis')||'sequence',pendingModels||(params.has('modelA')?[params.get('modelA'),params.get('modelB')]:null));
      if(location.hash==='#execution-analysis')requestAnimationFrame(()=>root.scrollIntoView({block:'start',behavior:'instant'}));
    }catch{document.getElementById('exec-stage').innerHTML='<p>Could not load comparisons. <a href="execution-guide.html">Open the case guide ↗</a></p>';}
  }
  document.addEventListener('execution-select',e=>{pendingStory=typeof e.detail==='string'?e.detail:e.detail.storyId;pendingModels=typeof e.detail==='object'?e.detail.models:null;if(data)selectStory(pendingStory,pendingModels);});
  const observer=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){observer.disconnect();init();}},{rootMargin:'500px'});
  if(location.hash==='#execution-analysis'||new URLSearchParams(location.search).has('analysis'))init();else observer.observe(root);

})();
