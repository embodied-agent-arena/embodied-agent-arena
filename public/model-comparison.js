// The crawlable results table is the single source for this interactive view.
(() => {
  'use strict';
  const root=document.getElementById('results-explorer'),table=document.getElementById('results-table');
  if(!root||!table)return;
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const groups=[['geometry','Geometry',[0,1]],['spatial','Spatial Reasoning',[2,3]],['affordance','Affordance',[4,5]],['planning','Planning',[6]],['manipulation','Manipulation',[7]]];
  const specs=[['Rotation error','degrees',false,60],['Translation error','cm',false,300],['Pass@1','%',true,100],['AbsRel','relative error',false,.4],['Box–mask IoU','IoU',true,.5],['Polygon IoU','IoU',true,.5],['Pass@1','%',true,100],['Task success','%',true,100]];
  const rows=[...table.tBodies[0].rows].map(tr=>({name:tr.cells[0].textContent.trim(),scores:[...tr.cells].slice(1).map(td=>{const label=td.textContent.trim(),v=label.split('±').map(Number);return{label,mean:v[0],sd:v[1]||0};})}));
  if(rows.length!==10||rows.some(r=>r.scores.length!==8||r.scores.some(s=>!Number.isFinite(s.mean))))return;
  let group='spatial',metric=2;
  function render(){
    const current=groups.find(g=>g[0]===group),[title,unit,higher,max]=specs[metric];
    const ranked=rows.slice().sort((a,b)=>(a.scores[metric].mean-b.scores[metric].mean)*(higher?-1:1));
    const modelStyle=name=>window.ArenaModelStyle.get(name);
    const best=ranked[0].scores[metric].mean,withSD=metric<5;
    root.innerHTML=`<div class="metric-tabs" role="tablist" aria-label="Choose a capability">${groups.map(([key,name])=>`<button type="button" role="tab" id="metric-tab-${key}" aria-controls="metric-panel" aria-selected="${group===key}" tabindex="${group===key?0:-1}" data-metric-group="${key}">${name}</button>`).join('')}</div><div id="metric-panel" role="tabpanel" aria-labelledby="metric-tab-${group}" tabindex="0"><div class="metric-toolbar"><div><h3>${esc(title)} <span>${higher?'Higher':'Lower'} Is Better</span></h3><p>${withSD?'Mean ± run SD · 5 runs; Fable: 3':'One run per task'}${metric===7?' · Source-specific observation schedules':''}</p></div>${current[2].length>1?`<div class="metric-switch" role="group" aria-label="Choose metric">${current[2].map(i=>`<button type="button" data-metric="${i}" aria-pressed="${metric===i}">${esc(specs[i][0])}</button>`).join('')}</div>`:''}</div><div class="ranking" aria-label="All ten models ranked by ${esc(title)}"><div class="ranking-axis"><span>Model</span><span><i>0</i><i>${max} ${esc(unit)}</i></span><span>${withSD?'Mean ± SD':'Score'}</span></div>${ranked.map((r,i)=>{const s=r.scores[metric],lo=Math.max(0,s.mean-s.sd),hi=Math.min(max,s.mean+s.sd),leader=s.mean===best,brand=modelStyle(r.name);return `<div class="ranking-row${leader?' is-leader':''}${r.name==='Gemini 4 Argon'?' is-argon':''}" data-result-model="${esc(r.name)}" style="--model-color:${brand.color};--model-light:${brand.light};--model-ink:${brand.ink}"><span class="ranking-name"><span class="rank">${i+1}</span><img class="model-logo" src="${brand.logo}" alt="" width="22" height="22"><strong>${esc(r.name)}</strong></span><span class="ranking-track" aria-hidden="true"><i class="ranking-fill" style="width:${s.mean/max*100}%"></i>${withSD?`<i class="ranking-sd" style="left:${lo/max*100}%;width:${(hi-lo)/max*100}%"></i><i class="ranking-mean" style="left:${s.mean/max*100}%"></i>`:''}</span><span class="ranking-value">${esc(s.label)}</span></div>`}).join('')}</div><p class="metric-note">${withSD?'Whiskers show the recorded SD at the plotted scale. ':''}<a href="#evaluation-details">Protocol &amp; budgets ↓</a></p></div>`;
    root.querySelectorAll('[data-metric-group]').forEach((b,i)=>{b.onclick=()=>choose(b.dataset.metricGroup,true);b.onkeydown=e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();const n=e.key==='Home'?0:e.key==='End'?groups.length-1:(i+(e.key==='ArrowRight'?1:-1)+groups.length)%groups.length;choose(groups[n][0],true);};});
    root.querySelectorAll('[data-metric]').forEach(b=>b.onclick=()=>{metric=Number(b.dataset.metric);render();root.querySelector(`[data-metric="${metric}"]`)?.focus();});
  }
  function choose(key,focus=false){const g=groups.find(g=>g[0]===key);if(!g)return;group=key;metric=g[2][0];render();const tab=root.querySelector(`[data-metric-group="${key}"]`);tab.parentElement.scrollLeft=Math.max(0,tab.offsetLeft-tab.parentElement.offsetLeft-20);if(focus)tab.focus();}
  render();root.hidden=false;document.getElementById('full-results').open=false;
  document.addEventListener('click',e=>{const a=e.target.closest('[data-results-domain]');if(!a||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;choose(a.dataset.resultsDomain);});
})();
