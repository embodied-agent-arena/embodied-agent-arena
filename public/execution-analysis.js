(() => {
  'use strict';
  const root = document.getElementById('execution-analysis');
  if (!root) return;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pretty = v => esc(JSON.stringify(v, null, 2));
  const number = n => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits: 1});
  const time = n => {if(n==null)return '—';const sec=Math.round(n);return sec<60?`${sec}s`:`${Math.floor(sec/60)}m ${sec%60}s`;};
  const tokens = n => n == null ? '—' : n >= 1e6 ? `${(n/1e6).toFixed(2)}M` : `${(n/1000).toFixed(1)}k`;
  const names = {'GPT-6 Astra':'GPT-6 Astra','GPT-6 Sol':'GPT-6 Sol','Claude Fable 5.1':'Fable 5.1','Claude Opus 5.5':'Opus 5.5','Gemini 3.8 Flash':'Gemini 3.8 Flash','Qwen 3.8 Max':'Qwen Max','Qwen 3.5 397B-A17B':'Qwen 397B','Qwen 3.5 27B':'Qwen 27B','DeepSeek 4.1 Flash-1017':'DeepSeek-1017'};
  const stops = {max_total_tokens:'Token limit',max_agent_iterations:'Round limit',primitive_calls:'Tool-call limit',max_env_steps:'Environment-step limit'};
  const budgetNames = {max_rounds:'Model rounds',max_total_tokens:'Tokens',max_primitive_calls:'Tool calls',max_env_steps:'Environment steps',timeout_seconds:'Time ceiling'};
  let data, story, generation = 0;
  const cache = new Map();
  const paneData = {};
  async function get(url) {if(!cache.has(url)) cache.set(url, fetch(url).then(r => {if(!r.ok) throw Error('Trace unavailable'); return r.json();}).catch(e=>{cache.delete(url);throw e;})); return cache.get(url);}
  function pause() {root.querySelectorAll('video').forEach(v=>v.pause());}
  function outcome(run) {return run.stats.completedSubgoals != null ? `${run.stats.completedSubgoals}/${run.stats.requiredSubgoals} goals` : run.success ? 'Completed' : run.outcome === 'budget_exhausted' ? 'Budget reached' : 'Not completed';}
  function metric(label, value, title='') {return `<div title="${esc(title)}"><strong>${esc(value)}</strong><span>${label}</span></div>`;}
  function bar(label, used, max) {if(used==null || !max) return '';return `<div class="exec-budget"><div><span>${label}</span><span>${number(used)} / ${number(max)}</span></div><meter min="0" max="${max}" value="${Math.min(used,max)}">${number(used)} of ${number(max)}</meter></div>`;}
  function renderStory(id) {
    pause(); generation++; story=data.stories.find(s=>s.id===id) || data.stories[0];
    root.querySelectorAll('[data-exec-story]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.execStory===story.id));});
    const runs=story.runs;
    const common=(key)=>{const values=runs.map(r=>r.budgets[key]);return values[0]!=null && values.every(v=>v===values[0])?values[0]:null;};
    const limits=[common('max_rounds')?`${common('max_rounds')} rounds`:null,common('max_total_tokens')?`${tokens(common('max_total_tokens'))} tokens`:null,common('max_env_steps')?`${common('max_env_steps')} environment steps`:null,common('timeout_seconds')?`${common('timeout_seconds')/60} min`:null].filter(Boolean).join(' · ');
    document.getElementById('exec-stage').innerHTML=`<div class="exec-story-heading"><p class="exec-kicker">${esc(story.benchmark)} <span>•</span> ${esc(story.domain)} <span>•</span> ${runs.length} Models / Same Task</p><h3>${esc(story.title)}</h3><p class="exec-takeaway">${esc(story.takeaway)}</p></div><div class="exec-task"><span>THE TASK</span><div><p>${esc(story.instruction)}</p><p class="exec-common-budget">Per-episode ceilings: ${esc(limits)}</p></div></div><div class="exec-compare-toolbar"><p>Choose two models. Inspect each decision.</p><button type="button" class="exec-play">▶ Play Both Clips</button></div><div class="exec-compare"><article class="exec-pane" data-pane="0"></article><article class="exec-pane" data-pane="1"></article></div><div class="exec-conclusion"><h4>${esc(story.question)}</h4><p>${esc(story.answer)}</p></div><details class="exec-scope"><summary>How to Read This Comparison</summary><p><a href="execution-guide.html">All Findings &amp; Episode Metrics ↗</a></p><p>${esc(data.scope)}</p><p>Round times measure elapsed time within the agent loop. Total duration includes recorded setup when available. Model calls and executed rounds can differ because a response may be retried or rejected. Tokens include input and output tokens. Video excerpts and saved observations play on their own clock; no unrecorded frame-to-log timing is inferred.</p><p>The trace viewer exposes executed code, tool parameters and retained feedback. It does not reconstruct hidden reasoning. “Tool call succeeded” and “task completed” are separate signals.</p></details>`;
    [0,1].forEach(i=>{const run=runs.find(r=>r.modelKey===story.default[i]) || runs[i] || runs[0]; renderPane(i,run.id);});
    root.querySelector('.exec-play').onclick=async()=>{const videos=[...root.querySelectorAll('.exec-video')];if(videos.some(v=>!v.paused)) videos.forEach(v=>v.pause());else await Promise.allSettled(videos.map(v=>v.play()));};
    document.getElementById('exec-announcement').textContent=`${story.benchmark}. ${story.title}`;
  }
  async function renderPane(i,id) {
    const run=story.runs.find(r=>r.id===id);if(!run)return;
    const pane=root.querySelector(`[data-pane="${i}"]`);pane.querySelector('video')?.pause();const token=generation;pane.dataset.run=id;
    const s=run.stats;
    pane.innerHTML=`<header class="exec-model"><label for="exec-model-${i}" class="sr-only">Model ${i+1}</label><select id="exec-model-${i}">${story.runs.map(r=>`<option value="${esc(r.id)}" ${r.id===id?'selected':''}>${esc(names[r.model]||r.model)}</option>`).join('')}</select><span class="exec-outcome ${run.success?'is-success':''}">${esc(outcome(run))}</span></header><div class="exec-video-frame"><video class="exec-video" controls playsinline muted preload="none" poster="${esc(run.poster)}" src="${esc(run.video)}" aria-label="${esc(run.model+' — '+run.instruction)}"></video></div><p class="exec-media-label">${esc(run.videoNote)}</p><div class="exec-metrics">${metric('Recorded Time',time(s.elapsedSeconds),'Elapsed duration, including recorded setup when available')}${metric('Model Calls',number(s.modelCalls))}${metric('Tokens',tokens(s.tokens),'Input plus output tokens')}</div><div class="exec-trace-loading" role="status">Loading execution trace…</div><details class="exec-settings"><summary>Settings &amp; Evidence</summary><p>${esc(run.model)} · ${esc(run.protocol)} observations</p>${bar('Rounds',s.rounds,run.budgets.max_rounds)}${bar('Tokens',s.tokens,run.budgets.max_total_tokens)}${bar('Tool calls',s.toolCalls,run.budgets.max_primitive_calls)}<dl>${Object.entries(budgetNames).filter(([k])=>run.budgets[k]!=null).map(([k,label])=>`<div><dt>${label}</dt><dd>${k==='timeout_seconds'?time(run.budgets[k]):number(run.budgets[k])}</dd></div>`).join('')}<div><dt>Stop condition</dt><dd>${esc(run.stopLimits.length?run.stopLimits.map(s=>stops[s]||s).join(', '):run.success?'Task completed':'Native task failure')}</dd></div>${s.apiSeconds!=null?`<div><dt>Recorded API wait</dt><dd>${time(s.apiSeconds)}</dd></div><div><dt>Mean API latency</dt><dd>${number(s.apiMeanSeconds)}s / call</dd></div>`:''}</dl><a href="${esc(run.detailUrl)}" download>Download This Trace ↓</a><p class="exec-record-id">Record ${esc(run.recordId)}<br>Report SHA-256: ${esc(run.reportSha256)}</p></details>`;
    pane.querySelector('select').onchange=e=>renderPane(i,e.target.value);
    pane.querySelector('video').addEventListener('play',()=>root.querySelector('.exec-play').textContent='Ⅱ Pause Clips');
    ['pause','ended'].forEach(type=>pane.querySelector('video').addEventListener(type,()=>{if([...root.querySelectorAll('.exec-video')].every(v=>v.paused))root.querySelector('.exec-play').textContent='▶ Play Both Clips';}));
    try {
      const detail=await get(run.detailUrl);
      if(token!==generation || pane.dataset.run!==id)return;
      paneData[i]=detail;
      if(detail.verifier?.message){const evidence=document.createElement('p');evidence.className='exec-native-check';evidence.textContent='Native final check: '+detail.verifier.message;pane.querySelector('.exec-settings').append(evidence);}
      const panel=pane.querySelector('.exec-trace-loading');panel.className='exec-trace';panel.removeAttribute('role');
      panel.innerHTML=`<div class="exec-trace-title"><h4>Execution Trace</h4><button class="exec-end" type="button">View Outcome →</button></div><div class="exec-chart"></div><div class="exec-scrubber"><button class="exec-prev" aria-label="Previous round for ${esc(run.model)}">←</button><label class="sr-only" for="exec-round-${i}">Execution round for ${esc(run.model)}</label><input type="range" id="exec-round-${i}" min="0" max="${Math.max(0,detail.steps.length-1)}" value="0" step="1"><button class="exec-next" aria-label="Next round for ${esc(run.model)}">→</button></div><div class="exec-step"></div>`;
      const slider=panel.querySelector('input');slider.oninput=()=>renderStep(i,Number(slider.value));
      panel.querySelector('.exec-end').onclick=()=>{slider.value=detail.steps.length-1;slider.dispatchEvent(new Event('input'));};
      panel.querySelector('.exec-prev').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);slider.dispatchEvent(new Event('input'));};
      panel.querySelector('.exec-next').onclick=()=>{slider.value=Math.min(detail.steps.length-1,Number(slider.value)+1);slider.dispatchEvent(new Event('input'));};
      renderStep(i,0);
    } catch(e) {if(token===generation && pane.dataset.run===id)pane.querySelector('.exec-trace-loading').innerHTML=`The trace could not be loaded. <button class="exec-retry">Try Again</button>`;pane.querySelector('.exec-retry')?.addEventListener('click',()=>renderPane(i,id));}
  }
  function renderStep(i,index) {
    const d=paneData[i],pane=root.querySelector(`[data-pane="${i}"]`);const step=d.steps[index];if(!step)return;
    pane.querySelector('.exec-prev').disabled=index===0;pane.querySelector('.exec-next').disabled=index===d.steps.length-1;
    const hasScore=d.steps.some(s=>s.score!=null),hasEnv=d.steps.some(s=>s.environmentSteps!=null),key=hasScore?'score':hasEnv?'environmentSteps':'cumulativeToolCalls';
    const vals=d.steps.map(s=>s[key] ?? 0),max=hasScore?1:Math.max(1,...vals),x=j=>16+j/Math.max(1,vals.length-1)*468,y=v=>94-v/max*72;
    let path=`M${x(0)},${y(vals[0])}`;for(let j=1;j<vals.length;j++)path+=` H${x(j)} V${y(vals[j])}`;
    pane.querySelector('.exec-chart').innerHTML=`<div class="exec-chart-label"><span>${hasScore?'Native Task Score':hasEnv?'Environment Steps':'Recorded Tool Calls'}</span><strong>${number(vals[index])}${hasScore?' / 1':''}</strong></div><svg viewBox="0 0 500 110" role="img" aria-label="${hasScore?'Task score':hasEnv?'Environment steps':'Cumulative tool calls'} across ${vals.length} decision rounds. Current value ${vals[index]}"><line x1="16" y1="94" x2="484" y2="94" class="exec-grid-line"/><line x1="16" y1="22" x2="484" y2="22" class="exec-grid-line"/><path d="${path}" class="exec-plot"/>${d.steps.map((s,j)=>`<circle cx="${x(j)}" cy="${y(vals[j])}" r="${index===j?6:3}" class="${s.error?'exec-error-dot':s.success?'exec-success-dot':'exec-dot'}"/>`).join('')}<line x1="${x(index)}" y1="12" x2="${x(index)}" y2="100" class="exec-cursor"/></svg><div class="exec-chart-axis"><span>Round 1</span><span>Round ${vals.length}</span></div>`;
    const calls=step.calls||[],count=Object.entries(calls.reduce((r,c)=>(r[c.name]=(r[c.name]||0)+1,r),{}));
    const progress=step.goalsMet!=null?`${step.goalsMet}/${step.goalsTotal} conditions`:step.score!=null?`Score ${number(step.score)}`:step.success?'Task completed':`${step.toolCalls} tool calls`;
    pane.querySelector('.exec-step').innerHTML=`<div class="exec-step-meta"><strong>Round ${step.round}</strong><span>${time(step.at)} elapsed</span><span class="${step.error?'exec-error-text':''}">${step.budgetStop?'Budget reached':step.error?'Execution error':esc(progress)}</span></div><p class="exec-step-title">${esc(step.title)}</p>${step.currentSubgoal?`<p class="exec-target"><span>${step.success?'Last Subgoal':'Current Subgoal'}</span>${esc(step.currentSubgoal)}</p>`:''}${step.error?`<p class="exec-error-brief">${esc(String(step.error).split('\n').filter(Boolean).slice(-1)[0]).slice(0,300)}</p>`:''}<details class="exec-evidence"><summary>Executed Code &amp; Feedback${calls.length?` · ${calls.length} Calls`:''}</summary>${step.code?`<h5>Executed Code${step.codeExcerpt?' · Excerpt':''}</h5><pre>${esc(step.code)}</pre>`:'<p>No executed code was retained for this round.</p>'}${step.stdout?`<h5>Returned Output</h5><pre>${esc(step.stdout)}</pre>`:''}${step.error?`<h5>Error</h5><pre>${esc(step.error)}</pre>`:''}${count.length?`<h5>Calls in This Round</h5><ul class="exec-call-counts">${count.map(([n,c])=>`<li><code>${esc(n)}</code><span>×${c}</span></li>`).join('')}</ul><div class="exec-call-list">${calls.map((c,k)=>`<details><summary>${k+1}. ${esc(c.name)} <span>${c.ok===true?'Call succeeded':c.ok===false?'Call failed':'Recorded'}</span></summary><pre>${pretty({arguments:c.arguments,output:c.output,...(c.error?{error:c.error}:{})})}</pre></details>`).join('')}</div>`:''}${Object.keys(step.feedback||{}).length?`<h5>Task Feedback</h5><pre>${pretty(step.feedback)}</pre>`:''}</details>`;
  }
  async function init() {
    try {
      data=await get('data/execution/index.json');
      document.getElementById('exec-counts').textContent=`${data.stories.length} Tasks · ${data.runCount} Episodes · ${data.modelCount} Models`;
      document.getElementById('exec-questions').innerHTML=data.stories.map((s,j)=>`<button type="button" data-exec-story="${s.id}" aria-pressed="${j===0}"><span>${esc(s.benchmark)}</span>${esc(s.label)}</button>`).join('');
      root.querySelectorAll('[data-exec-story]').forEach(b=>b.onclick=()=>{const url=new URL(location.href);url.searchParams.set('analysis',b.dataset.execStory);url.hash='execution-analysis';history.replaceState(null,'',url);renderStory(b.dataset.execStory);});
      renderStory(new URLSearchParams(location.search).get('analysis')||data.stories[0].id);
    } catch(e) {document.getElementById('exec-stage').innerHTML='<p>Execution analysis could not be loaded. <a href="data/execution/index.json">Open the evidence index ↗</a></p>';}
  }
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){observer.disconnect();init();}},{rootMargin:'500px'});observer.observe(root);
  const videoObserver=new IntersectionObserver(entries=>entries.forEach(e=>{if(!e.isIntersecting)pause();}),{threshold:0});videoObserver.observe(root);
})();
