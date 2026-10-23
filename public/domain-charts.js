/* Native evidence charts. Shared numeric inputs, original paper colour ramp. */
(() => {
  'use strict';
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (v, n=1) => Number(v).toFixed(n).replace(/\.0$/, '');
  const brand = m => window.ArenaModelStyle.get(m);
  let data, pending;
  const load = () => pending ||= fetch('data/report/domain-charts.json?v=humanclaw-20261023').then(r => {if(!r.ok) throw Error('Chart data unavailable');return r.json();}).then(d => (data=d)).catch(error => {pending=null;throw error;});
  const heat = v => data.heatRamp[Math.round(Math.max(0,Math.min(100,v))/100*(data.heatRamp.length-1))];
  const metricText = (z, unit='%') => `${fmt(z.value,unit.includes('cm')?3:unit.includes('episodes')?0:2)}${unit}${z.sd == null ? '' : ` ± ${fmt(z.sd,unit.includes('cm')?3:unit.includes('episodes')?0:2)}${unit} run SD`}`;
  const logo = m => `<img src="${brand(m).logo}" alt="" width="23" height="23">`;
  const model = m => `${logo(m)}<span>${esc(brand(m).short)}</span>`;
  function section(id,title,body,note='') {
    return `<section class="dc-section" data-chart-section="${id}"><h4>${title}</h4>${body}${note?`<p class="dc-note">${note}</p>`:''}</section>`;
  }
  function readout(text) {return `<p class="dc-readout" role="status" aria-live="polite">${text}</p>`;}
  function frame(content,label,height=430) {
    return `<div class="dc-scroll" tabindex="0" role="region" aria-label="${esc(label)}; scroll horizontally on narrow screens"><svg class="dc-svg dc-plot" viewBox="0 0 1120 ${height}" role="group" aria-label="${esc(label)}">${content}</svg></div>`;
  }
  function markAttrs(detail) {return `class="dc-mark" tabindex="0" role="button" aria-label="${esc(detail)}" data-detail="${esc(detail)}"`;}
  function bars(rows,{max=100,min=0,ticks=[0,50,100],unit='%',axis='Accuracy (%)',description='Select a bar to inspect its value.',height=430}={}) {
    const x0=65,x1=1100,top=42,bottom=height-90,w=(x1-x0)/rows.length,y=v=>bottom-(v-min)/(max-min)*(bottom-top),zero=y(0);
    let s=`<text x="${x0}" y="23" class="dc-axis-title">${esc(axis)}</text>`;
    for(const v of ticks)s+=`<path class="dc-grid" d="M${x0} ${y(v)}H${x1}"/><text class="dc-tick" x="${x0-12}" y="${y(v)+6}" text-anchor="end">${fmt(v)}</text>`;
    rows.forEach((r,i)=>{
      const b=r.model?brand(r.model):null,col=r.color||b?.color||'#547A65',x=x0+w*(i+.5),bw=Math.min(58,w*.55),value=r.value,yy=y(value);
      const name=r.label||b?.short,detail=r.detail||`${b?.name||name}: ${metricText(r,unit)}. ${r.definition||''}`;
      s+=`<g ${markAttrs(detail)}><title>${esc(detail)}</title><rect x="${x-w*.44}" y="${top}" width="${w*.88}" height="${bottom-top+75}" fill="transparent"/><rect class="dc-bar" x="${x-bw/2}" y="${Math.min(yy,zero)}" width="${bw}" height="${Math.abs(zero-yy)}" rx="5" fill="${col}" style="transform-origin:${x}px ${zero}px"/>`;
      if(value===0)s+=`<path d="M${x-bw/2} ${zero}h${bw}" stroke="${col}" stroke-width="3"/>`;
      if(r.sd!=null){const lo=y(value-r.sd),hi=y(value+r.sd);s+=`<path class="dc-whisker" d="M${x} ${lo}V${hi}M${x-6} ${lo}h12M${x-6} ${hi}h12"/>`;}
      if(r.dot!=null){const lo=y(r.dot-(r.dotSd||0)),hi=y(r.dot+(r.dotSd||0));s+=`<path class="dc-comparison" d="M${x} ${lo}V${hi}M${x-7} ${lo}h14M${x-7} ${hi}h14"/><circle class="dc-comparison" cx="${x}" cy="${y(r.dot)}" r="5"/>`;}
      const labelTop=Math.min(yy,y(value+(r.sd||0)),r.dot==null?yy:y(r.dot+(r.dotSd||0)));
      s+=`<text class="dc-value" x="${x}" y="${value<0?yy+24:labelTop-12}" text-anchor="middle">${fmt(value,unit.includes('cm')?2:1)}</text>`;
      if(b)s+=`<image href="${b.logo}" x="${x-11}" y="${bottom+17}" width="22" height="22"/>`;
      s+=`<text x="${x}" y="${bottom+(b?65:37)}" class="dc-label" text-anchor="middle">${esc(name)}</text></g>`;
    });
    return frame(s,axis,height)+readout(description);
  }
  function heatmap(columns, rows, title, {unit='%',max=100,selectable=false,selected=0}={}) {
    return `<div class="dc-scroll dc-heat-scroll" tabindex="0" role="region" aria-label="${esc(title)}; scroll horizontally on narrow screens"><table class="dc-heatmap"><caption class="sr-only">${esc(title)}</caption><thead><tr><th scope="col">Model</th>${columns.map((c,i)=>`<th scope="col">${selectable?`<button type="button" data-column="${i}" aria-pressed="${i===selected}" title="${esc(c.definition)}" data-detail="${esc(c.definition)}">${esc(c.label)} <span aria-hidden="true">ⓘ</span></button>`:`<span title="${esc(c.definition)}">${esc(c.label)}</span>`}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr><th scope="row" title="${esc(brand(r.model).name)}">${model(r.model)}</th>${r.values.map((z,i)=>{const detail=`${brand(r.model).name} · ${columns[i].title||columns[i].label}: ${fmt(z.value,unit==='%'?2:0)}${unit}${z.sd==null?'':` ± ${fmt(z.sd,2)}${unit} run SD`}. ${z.cases?`${z.cases} cases${z.runs?` × ${z.runs} ${z.runs===1?'run':'runs'}`:''}. `:''}${columns[i].definition}`;return `<td><button type="button" class="dc-heat-cell" style="background:${heat(z.value/max*100)};color:${z.value/max>=.65?'#fff':'#30343b'}" data-detail="${esc(detail)}" aria-label="${esc(detail)}" data-heat-column="${i}">${fmt(z.value,0)}</button></td>`;}).join('')}</tr>`).join('')}</tbody></table></div><div class="dc-scale"><span>${esc(unit==='%'?'Accuracy (%)':unit)}</span><span>0</span><i style="background:linear-gradient(90deg,${data.heatRamp.join(',')})"></i><span>${max}</span></div>`;
  }
  function geometry() {
    const g=data.geometry;
    const reduction=bars(g.reductions.map(r=>({...r,color:r.value>=0?'#202020':'#686868',detail:`${r.label}: ${fmt(r.value)}% error reduction relative to ${brand(r.peer).name}. Astra error ${fmt(r.astra,3)}%; peer error ${fmt(r.peerMean,3)}%. ${r.cases} assigned cases. ${r.definition}`})),{min:-20,max:80,ticks:[-20,0,40,80],axis:'Error reduction relative to the best original peer (%)',description:'Positive values mean lower error for Astra. Select a bar for the metric definition and comparator.'});
    const controls=`<div class="dc-controls" role="group" aria-label="Camera condition">${g.motion.map((x,i)=>`<button type="button" data-motion-condition="${i}" aria-pressed="${i===0}">${x.label}</button>`).join('')}</div><div data-motion-chart></div>`;
    const x=v=>95+(v-.25)/.55*960,y=v=>460-v/50*400;
    let scatter='<text x="95" y="25" class="dc-axis-title">Joint tracing success (%) ↑</text>';
    for(const v of [0,20,40])scatter+=`<path class="dc-grid" d="M95 ${y(v)}H1055"/><text x="78" y="${y(v)+6}" text-anchor="end">${v}</text>`;
    for(const v of [.3,.4,.5,.6,.7,.8])scatter+=`<text x="${x(v)}" y="494" text-anchor="middle">${v}</text>`;
    scatter+='<path class="dc-axis" d="M95 60V460H1055"/><text x="575" y="535" text-anchor="middle">3D path RMSE (m) · lower is better ←</text>';
    const offsets={sol:[18,-4],qwen:[18,4],deepseek:[18,14],'qwen35-397b':[-18,-14],'qwen35-27b':[18,22],opus55:[18,-8]};
    g.trace.forEach(r=>{const b=brand(r.model),[dx,dy]=offsets[r.model]||[18,3],detail=`${b.name}: ${fmt(r.rmse,3)} m path RMSE; ${fmt(r.success)}% joint success on ${r.cases} paths. Success requires initiation, arrival, and collision constraints to hold together.`;scatter+=`<g ${markAttrs(detail)}><rect x="${x(r.rmse)+Math.min(-17,dx<0?dx-b.short.length*11:dx)}" y="${y(r.success)+Math.min(-20,dy-23)}" width="${b.short.length*11+40}" height="${Math.max(20,dy+6)-Math.min(-20,dy-23)}" fill="transparent"/><circle cx="${x(r.rmse)}" cy="${y(r.success)}" r="17" fill="transparent"/><image class="dc-point" href="${b.logo}" x="${x(r.rmse)-9}" y="${y(r.success)-9}" width="18" height="18"/><text x="${x(r.rmse)+dx}" y="${y(r.success)+dy}" text-anchor="${dx<0?'end':'start'}" style="color:${b.ink}">${esc(b.short)}</text></g>`;});
    return section('reduction','(a) Geometry error reduction',reduction)+section('motion','(b) Object motion under camera changes',controls,'MAE = mean absolute displacement error in centimetres. Whiskers show sample SD over five run means; lower is better.')+section('trace','(c) Path accuracy and task completion',frame(scatter,'3D path RMSE versus joint constraint success',560)+readout('Select a model marker. RMSE measures path error; joint success checks every required path constraint.'),'One run on each of 56 paths per model; values retain the paper table’s precision.');
  }
  function motionChart(host,index) {
    const r=data.geometry.motion[index];
    host.querySelector('[data-motion-chart]').innerHTML=bars(Object.entries(r.results).map(([model,z])=>({model,value:z.mean,sd:z.sd,definition:r.definition})),{max:3.5,min:-.25,ticks:[0,1,2,3],unit:' cm',axis:'Displacement MAE (cm) ↓',description:r.definition});
  }
  function spatial() {
    const columns=data.spatial.panels.a.concat(data.spatial.panels.b),rows=data.models.map(m=>({model:m,values:columns.map(c=>({...c.results[m],value:c.results[m].mean}))}));
    return section('spatial-matrix','(a–b) Relations, viewpoints, and depth',`<p class="dc-instruction">Select a column heading to understand the task and compare models.</p>${heatmap(columns,rows,'Task-level spatial accuracy',{selectable:true,selected:3})}<div class="dc-definition" data-task-definition></div><div data-task-bars></div>`,'Heatmap labels round to integers. Selected-task bars and readouts retain the measured means and sample SD. All ten models share the same cases.')+section('spatial-sources','(c) Performance across sources',`<div class="dc-controls" role="group" aria-label="Spatial source">${data.spatial.sourceInfo.map((s,i)=>`<button type="button" data-source="${i}" aria-pressed="${i===0}">${s.label}</button>`).join('')}</div><div data-source-chart></div>`,'Native Pass@1 includes numerical questions scored by each source’s original evaluator. Five runs per case; Fable: three.');
  }
  function taskChart(host,index) {
    const c=data.spatial.panels.a.concat(data.spatial.panels.b)[index];
    host.querySelectorAll('[data-column]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.column===index)));
    host.querySelector('[data-task-definition]').innerHTML=`<strong>${esc(c.label)} · ${esc(c.title)}</strong><span>${esc(c.definition)} <b>${c.cases} cases · ${esc(data.spatial.sourceInfo.find(s=>s.key===c.source).label)}</b></span>`;
    host.querySelector('[data-task-bars]').innerHTML=bars(data.models.map(m=>({model:m,value:c.results[m].mean,sd:c.results[m].sd,definition:`${c.cases} cases × ${c.results[m].runs} runs. ${c.definition}`})),{min:-25,max:125,ticks:[0,50,100],axis:`${c.title} · accuracy (%)`,description:'Select any bar for its exact mean, run SD, and model name.'});
  }
  function sourceChart(host,index) {
    const s=data.spatial.sourceInfo[index];
    host.querySelector('[data-source-chart]').innerHTML=bars(data.models.map(m=>{const z=data.spatial.sources[s.key][m];return {model:m,value:z.mean,sd:z.sd,definition:`${z.cases} cases × ${z.runs} ${z.runs===1?'run':'runs'}. ${s.definition}`};}),{max:115,ticks:[0,50,100],axis:`${s.label} · native Pass@1 (%)`,description:s.definition});
  }
  function affordance() {
    return section('regions','(a) Polygon and box overlap',bars(data.affordance.regions.map(r=>({model:r.model,value:r.polygon,dot:r.box,dotSd:r.sd,detail:`${brand(r.model).name}: polygon IoU ${fmt(r.polygon,1)}%; box–mask IoU ${fmt(r.box,2)}% ± ${fmt(r.sd,2)}% run SD. Same ${r.cases} cases; polygons: one run; boxes: ${r.runs} runs.`})),{max:55,ticks:[0,25,50],axis:'Mask intersection over union (%) ↑',description:'Colour bars: polygons. White dots with whiskers: boxes ± run SD. Select a model to compare both.'}),'IoU = intersection area divided by union area with the ground-truth mask. Invalid or absent polygons score zero.')+section('affordance-sources','(b) Tools versus interaction scenes',bars(data.affordance.sources,{max:100,ticks:[0,50,100],axis:'Gemini 4 Argon · polygon IoU (%) ↑',description:'Select a source to see what kind of grounding it tests.'}),'UMD tests functional tool regions. RAGNet and ReasonAff require action-conditioned region grounding in interaction scenes.');
  }
  function planning() {
    const p=data.planning;
    return section('planning-models','(a) Complete task success',bars(p.models.map(r=>({...r,detail:`${brand(r.model).name}: ${r.count} of ${p.total} assigned tasks completed (${fmt(r.value,2)}%). One recorded episode per task.`})),{axis:'Planning task success (%) ↑',description:'Select a model for its completion count across all 157 assigned tasks.'}))+section('planning-coverage','(b) Shared successes and failures',bars(p.histogram.map((n,i)=>({label:String(i),value:n,color:i===0?'#A64B50':i===10?'#547A65':'#D3A163',detail:`${n} of ${p.total} tasks are completed by exactly ${i} of the ten models.${i===0?' Shared failures.':i===10?' Shared successes.':' Mixed outcomes across models.'}`})),{max:30,ticks:[0,10,20,30],unit:' tasks',axis:'Number of tasks',description:'Horizontal axis: number of models completing the same task, from 0 to 10.'}),'Each task contributes once. Combined coverage is the union of recorded successes: 142 of 157; 15 are unsolved by all ten models.');
  }
  function manipulation() {
    const c=data.manipulation,columns=[1,2,3,4,5].map(k=>({label:`≥ ${k} ${k===1?'goal':'goals'}`,definition:`Episodes completing the first ${k} consecutive ${k===1?'subgoal':'subgoals'} in the prescribed order. Each model has ten assigned episodes.`}));
    const rows=c.models.map(r=>({model:r.model,values:r.reached.map(value=>({value,cases:10,runs:1}))}));
    return section('calvin-goals','(a) Progress through the full sequence',heatmap(columns,rows,'CALVIN consecutive goals reached',{unit:' episodes',max:10})+readout('Select a cell. Reaching an early goal does not mean all five goals were completed.'),'Counts out of ten episodes per model. Six records with no observed prefix receive no goal credit.')+section('calvin-recovery','(b) What happens after an execution error?',bars([
      {label:'Execution error',value:c.total.error_episodes,color:'#A64B50',definition:'Episodes with a failed code-execution round. Transport failures, timeouts, and budget guards are excluded.'},
      {label:'Then executes',value:c.total.resumed,color:'#D3A163',definition:'Error episodes with at least one later round of executable code.'},
      {label:'Then completes',value:c.total.completed_after_error,color:'#547A65',definition:'Error episodes that later reach native success for the complete five-goal sequence.'}
    ],{max:70,ticks:[0,20,40,60],unit:' episodes',axis:'Episodes · all 100 assigned CALVIN runs',description:'These are nested counts: an error, a later executable round, and eventual complete success.'}),'Ten sequences × ten models, one adopted episode each. Recorded observation schedules and execution budgets are retained.');
  }
  function progress() {
    const x=r=>95+(r-1)*127,y=v=>355-v*280;
    let svg='<text x="95" y="25" class="dc-axis-title">Native task progress</text>';
    for(const v of [0,.5,1])svg+=`<path class="dc-grid" d="M95 ${y(v)}H1055"/><text x="75" y="${y(v)+6}" text-anchor="end">${v}</text>`;
    for(let r=1;r<=8;r++)svg+=`<text x="${x(r)}" y="390" text-anchor="middle">${r}</text>`;
    svg+='<text x="560" y="430" text-anchor="middle">Decision round</text>';
    data.progress.forEach((series,index)=>{
      const b=brand(series.model),points=series.rounds;
      let path=`M${x(points[0].round)} ${y(points[0].score)}`;
      points.slice(1).forEach(p=>{path+=`H${x(p.round)}V${y(p.score)}`;});
      svg+=`<path d="${path}" fill="none" stroke="${b.color}" stroke-width="3" ${series.model==='gemini4argon'?'stroke-dasharray="7 4"':''}/>`;
      points.forEach(p=>{const detail=`${b.name} · round ${p.round}: native task progress ${fmt(p.score,1)} of 1.0. Progress is the source evaluator’s recorded score after the decision round.`;svg+=`<g ${markAttrs(detail)}><circle cx="${x(p.round)}" cy="${y(p.score)}" r="14" fill="transparent"/><circle class="dc-point" cx="${x(p.round)}" cy="${y(p.score)}" r="5" fill="${series.model==='gemini4argon'?'white':b.color}" stroke="${b.color}" stroke-width="2"/></g>`;});
      const end=points.at(-1),dy=-22;
      svg+=`<image href="${b.logo}" x="${x(end.round)-45}" y="${y(end.score)+dy-17}" width="22" height="22"/><text x="${x(end.round)-17}" y="${y(end.score)+dy}" style="color:${b.ink}">${b.short}</text>`;
    });
    return section('execution-progress','Archaeology Dating · task progress',frame(svg,'Native task progress by decision round',455)+readout('Select a recorded point to inspect its model, round, and native score.'),'Steps follow recorded decision rounds. A score of 1.0 indicates native task completion.');
  }
  const renderers={geometry,spatial,affordance,planning,manipulation,progress};
  function selectControl(host,button,key,callback) {
    host.querySelectorAll(`[${key}]`).forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
    callback(host,Number(button.getAttribute(key)));
  }
  function bind(host) {
    const tip=document.createElement('div');tip.className='dc-tooltip';tip.hidden=true;tip.setAttribute('role','tooltip');host.appendChild(tip);
    function hide(){tip.hidden=true;}
    function detail(target,event){
      const mark=target.closest('[data-detail]');if(!mark)return;
      const section=mark.closest('.dc-section'),out=section?.querySelector('.dc-readout');
      if(out)out.textContent=mark.dataset.detail;
      section?.querySelectorAll('.dc-selected').forEach(n=>n.classList.remove('dc-selected'));mark.classList.add('dc-selected');
      tip.textContent=mark.dataset.detail;tip.hidden=false;
      const box=mark.getBoundingClientRect(),rect=tip.getBoundingClientRect();
      const px=event?.clientX??(box.left+box.width/2),py=event?.clientY??box.bottom;
      tip.style.left=`${Math.max(10,Math.min(px+16,innerWidth-rect.width-10))}px`;
      tip.style.top=`${Math.max(85,Math.min(py+20,innerHeight-rect.height-12))}px`;
    }
    host.addEventListener('pointerover',e=>{if(e.pointerType!=='touch')detail(e.target,e);});
    host.addEventListener('pointerout',e=>{const mark=e.target.closest('[data-detail]');if(mark&&!mark.contains(e.relatedTarget))hide();});
    host.addEventListener('focusin',e=>detail(e.target));
    host.addEventListener('focusout',hide);
    host.addEventListener('click',e=>{
      const t=e.target.closest('button,[role="button"]');if(!t){hide();return;}detail(t,e);
      if(t.hasAttribute('data-column'))taskChart(host,+t.dataset.column);
      if(t.hasAttribute('data-motion-condition'))selectControl(host,t,'data-motion-condition',motionChart);
      if(t.hasAttribute('data-source'))selectControl(host,t,'data-source',sourceChart);
    });
    host.addEventListener('keydown',e=>{if(e.key==='Escape')hide();if(e.target.matches('.dc-mark')&&['Enter',' '].includes(e.key)){e.preventDefault();detail(e.target);}});
    // Capture scrolls from the local chart overflow as well as the document.
    host.addEventListener('scroll',hide,true);
  }
  async function mount(host) {
    if(host.dataset.chartMounted)return;host.dataset.chartMounted='loading';
    try{await load();if(!host.isConnected)return;const kind=host.dataset.domainChart;host.innerHTML=renderers[kind]();host.dataset.chartMounted='ready';bind(host);if(kind==='spatial'){taskChart(host,3);sourceChart(host,0);}if(kind==='geometry')motionChart(host,0);}
    catch(error){host.dataset.chartMounted='error';host.innerHTML='<p>Chart data could not load. <button type="button" class="dc-retry">Retry</button> · <a href="paper.pdf">Read the paper figures</a></p>';host.querySelector('button').onclick=()=>{delete host.dataset.chartMounted;mount(host);};}
  }
  function scan(root){if(root.nodeType!==1)return;if(root.matches('[data-domain-chart]'))mount(root);root.querySelectorAll('[data-domain-chart]').forEach(mount);}
  document.addEventListener('scroll',()=>document.querySelectorAll('.dc-tooltip').forEach(t=>t.hidden=true),{passive:true});
  document.addEventListener('pointerdown',e=>document.querySelectorAll('.dc-tooltip').forEach(t=>{if(!t.parentElement.contains(e.target))t.hidden=true;}));
  scan(document.documentElement);
  new MutationObserver(records=>records.forEach(r=>r.addedNodes.forEach(scan))).observe(document.body,{subtree:true,childList:true});
  window.ArenaDomainCharts={mount};
})();
