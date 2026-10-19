const domains=[
  {
    "id": "geometry",
    "label": "Geometry",
    "title": "Metric Precision Is Only Part of Geometric Competence.",
    "text": "Astra has the lowest pose errors. The tracing comparison shows why a close path can still miss the endpoint or violate a task constraint.",
    "evidence": [
      [
        "17.2 ± 6.7°",
        "Astra rotation error",
        "DeepSeek-1017 · 30.2 ± 3.9°"
      ],
      [
        "106.9 ± 36.0 cm",
        "Astra translation error",
        "DeepSeek-1017 · 182.0 ± 29.0 cm"
      ]
    ],
    "caption": "(a) Astra’s error reduction against its original six peers. (b) Object displacement under fixed or moving cameras, including DeepSeek-1017. (c) Nine-model 3D path error and joint tracing success."
  },
  {
    "id": "spatial",
    "label": "Spatial Reasoning",
    "title": "Spatial Strengths Depend on the Question.",
    "text": "Astra leads Spatial Pass@1, while Gemini has the lowest numerical error. Source-level comparisons expose gaps hidden by an overall score.",
    "evidence": [
      [
        "69.4 ± 1.0%",
        "Astra Spatial Pass@1",
        "DeepSeek-1017 · 51.6 ± 1.5%"
      ],
      [
        "0.161 ± 0.028",
        "Gemini numerical AbsRel",
        "Astra · 0.214 ± 0.016"
      ]
    ],
    "caption": "(a–b) Relation, viewpoint, and depth diagnostics for the original seven agents. (c) Native Pass@1 for all nine agents across six sources."
  },
  {
    "id": "affordance",
    "label": "Affordance",
    "title": "Finding a Contact Is Not the Same as Tracing a Region.",
    "text": "Opus 5.5 achieves the highest box–mask and polygon IoU. Direct polygons test region shape beyond the rectangular coverage of a predicted box.",
    "evidence": [
      [
        "0.397",
        "Opus 5.5 polygon IoU",
        "DeepSeek-1017 · 0.249"
      ],
      [
        "0.311 ± 0.020",
        "Opus 5.5 box–mask IoU",
        "DeepSeek-1017 · 0.217 ± 0.011"
      ]
    ],
    "caption": "(a) UMD contact validity for all nine agents. (b) Direct-polygon IoU for all nine models on 70 cases, one run per case. (c) Astra’s contact validity by interaction; 1H/2H denotes one/two hands."
  },
  {
    "id": "planning",
    "label": "Task Planning",
    "title": "High Planning Scores Still Leave Physical Goals Unfinished.",
    "text": "Astra leads overall Planning at 80.3%. Opus 5.5 completes 3/13 HumanCLAW tasks and DeepSeek-1017 2/13; those successes involve navigation, while all eight sitting tasks remain incomplete.",
    "evidence": [
      [
        "80.3%",
        "Astra Planning success",
        "Opus 5.5 · 48.4%"
      ],
      [
        "31.2%",
        "DeepSeek-1017 Planning success",
        "157 tasks per agent"
      ]
    ],
    "caption": "(a) Source-level success for Astra, Fable, Opus 5.5, and DeepSeek-1017. (b–c) Astra and Fable’s household-goal and ScienceWorld diagnostics. Each assignment uses one episode."
  },
  {
    "id": "manipulation",
    "label": "Manipulation",
    "title": "Local Control Does Not Ensure Complete Household Tasks.",
    "text": "Opus 5.5 completes 47.8% of assigned Manipulation tasks; DeepSeek-1017 completes 20.9%. No model completes any of the five BEHAVIOR-1K tasks within the source limits.",
    "evidence": [
      [
        "47.8%",
        "Opus 5.5 task success",
        "Astra · 41.8%"
      ],
      [
        "20.9%",
        "DeepSeek-1017 task success",
        "38 / 182 tasks"
      ]
    ],
    "caption": "(a) Nine-model success across thirteen suites. (b) Consecutive CALVIN goals, out of five. (c) ManiSkill condition satisfaction. Filled/open markers show Baseline/additional visual feedback; teal triangles show DeepSeek-1017’s per-round RGB setting."
  }
];
const cases=[
 {id:'household',label:'Household Manipulation',title:'Local Skills Do Not Yet Add Up to Complete Tasks.',value:'0%',metric:'BEHAVIOR-1K complete-task success',scope:'0 / 5 tasks for each of the nine agents',explanation:'No agent satisfied all required final household conditions in any of the five evaluated tasks. This score counts complete tasks; intermediate actions can succeed while the task remains unfinished.',budget:'180 min trial time; 24 model rounds; 256 tool calls; 640k tokens; 1,000 environment control steps.',protocol:'One episode per task and agent, with source-native observations and tools. Trial time includes model calls, execution, and feedback. Runs can terminate unsuccessfully before the time limit.',text:'The BEHAVIOR-1K row shows 0/5 complete tasks for all nine agents. Intermediate progress does not satisfy the full set of required final conditions. Other rows show suites with their own source budgets.',image:'manipulation',alt:'Task success across thirteen manipulation suites. The BEHAVIOR-1K row shows 0 of 5 completed tasks per agent under the specified evaluation budgets.'},
 {id:'body',label:'Body–Object Interaction',title:'Finding the Seat Does Not Complete the Sitting Task.',value:'0 / 8',metric:'HumanCLAW sitting tasks completed',scope:'0 / 8 tasks for each of the nine agents',explanation:'Success requires the environment to confirm the required sitting state. These eight sitting tasks form a subset of HumanCLAW’s 13 evaluated tasks; the 0/8 result applies only to that subset.',budget:'180 min episode time (11,000 s outer task timeout); 101 code rounds; 100 environment steps; 400 primitive calls; 2M tokens.',protocol:'One episode per task and agent, with source-native observations and tools. Runs can terminate unsuccessfully before the time limit.',text:'Astra found the seat in six cases and approached within 20 cm while issuing a sit action in five. None achieved the required sitting state. These records distinguish target finding and attempted execution from the final condition that determines success.',diagnostic:[['6','Target Found'],['5','Within 20 cm + Sit Action Issued'],['0','Sitting State Confirmed']]}
];
function buildTabs(items,prefix,render){
 const list=document.querySelector(`#${prefix}-tabs`),panel=document.querySelector(`#${prefix}-panel`);
 items.forEach((item,i)=>{const b=document.createElement('button');b.type='button';b.id=`${prefix}-tab-${item.id}`;b.setAttribute('role','tab');b.setAttribute('aria-controls',panel.id);b.style.setProperty('--tab-color',prefix==='domain'?`var(--${item.id})`:'var(--ink)');b.textContent=item.label;b.addEventListener('click',()=>select(i));list.appendChild(b);});
 function select(index,focus=false){[...list.children].forEach((b,i)=>{b.setAttribute('aria-selected',String(i===index));b.tabIndex=i===index?0:-1;});panel.setAttribute('aria-labelledby',list.children[index].id);panel.innerHTML=render(items[index]);if(focus)list.children[index].focus();}
 list.addEventListener('keydown',e=>{const i=[...list.children].indexOf(document.activeElement);let n=i;if(e.key==='ArrowRight')n=(i+1)%items.length;else if(e.key==='ArrowLeft')n=(i-1+items.length)%items.length;else if(e.key==='Home')n=0;else if(e.key==='End')n=items.length-1;else return;e.preventDefault();select(n,true);});select(0);
}
// Reserve figure space before lazy loading so in-page links keep their position.
const figureSizes={"geometry": [1875, 512], "spatial": [2063, 499], "affordance": [1875, 512], "planning": [1875, 512], "manipulation": [2100, 697]};
function figureSizeAttrs(name){const size=figureSizes[name];return size?`width="${size[0]}" height="${size[1]}"`:'';}
buildTabs(domains,'domain',d=>`<div class="domain-text"><h3>${d.title}</h3><ul class="evidence">${d.evidence.map(([value,label,peer])=>`<li><strong>${value}</strong><span>${label}</span><span class="peer">${peer}</span></li>`).join('')}</ul></div><figure><button class="figure-button" data-zoom="assets/${d.id}.webp?v=run-sd-20261019" aria-label="Enlarge ${d.label} results"><img ${figureSizeAttrs(d.id)} src="assets/${d.id}.webp?v=run-sd-20261019" alt="${d.label}: ${d.caption}" loading="lazy"></button><figcaption><p class="figure-takeaway">${d.text}</p><p class="figure-caption"><strong>${d.label}.</strong> ${d.caption}</p></figcaption></figure>`);
function renderCompletionCase(c){
 const evidence=c.diagnostic?`<div class="sitting-diagnostic"><h4>Astra on the Same Eight Sitting Tasks</h4><dl>${c.diagnostic.map(([count,label])=>`<div><dt>${label}</dt><dd>${count}<span> / 8</span></dd></div>`).join('')}</dl></div>`:`<button class="figure-button" data-zoom="assets/${c.image}.webp?v=run-sd-20261019" aria-label="Enlarge ${c.label} results"><img ${figureSizeAttrs(c.image)} src="assets/${c.image}.webp?v=run-sd-20261019" alt="${c.alt}" loading="lazy"></button>`;
 return `<h3>${c.title}</h3><div class="completion-highlight"><strong>${c.value}</strong><div><span>${c.metric}</span><span class="completion-scope">${c.scope}</span></div></div><p class="case-explanation">${c.explanation}</p><div class="case-setting"><p><strong>Per-episode limits.</strong> ${c.budget}</p><p>${c.protocol}</p></div><figure class="case-figure">${evidence}<figcaption class="figure-takeaway">${c.text}</figcaption></figure>`;
}
buildTabs(cases,'case',renderCompletionCase);
const citation=document.querySelector('#bibtex').textContent.trim();
document.querySelector('#copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(citation);document.querySelector('#copy-status').textContent='Citation copied.';document.querySelector('#copy-citation').textContent='Copied';setTimeout(()=>document.querySelector('#copy-citation').textContent='Copy BibTeX',2000);}catch{const range=document.createRange();range.selectNodeContents(document.querySelector('#bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);document.querySelector('#copy-status').textContent='Citation selected. Use your browser’s copy command.';}});
const dialog=document.querySelector('#figure-dialog'),dialogImage=document.querySelector('#dialog-image');
document.addEventListener('click',e=>{const button=e.target.closest('[data-zoom]');if(!button)return;dialogImage.src=button.dataset.zoom;dialogImage.alt=button.querySelector('img').alt;dialog.showModal();});
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});

// Load motion only when visible. Respect reduced motion and explicit pause.
const motionPreference=window.matchMedia('(prefers-reduced-motion: reduce)');
const motionClips=[...document.querySelectorAll('video[data-src]')].map(video=>({video,button:document.querySelector(`[data-motion="${video.id}"]`),visible:false,userPaused:motionPreference.matches}));
function setMotionIcon(clip){
 const paused=clip.video.paused;
 const purpose=clip.video.id==='ambient-video'?'background motion':'task demonstrations';
 clip.button.setAttribute('aria-label',`${paused?'Play':'Pause'} ${purpose}`);
 clip.button.title=`${paused?'Play':'Pause'} ${purpose}`;
 clip.button.innerHTML=`<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="${paused?'M8 5v14l11-7z':'M7 5h3v14H7zM14 5h3v14h-3z'}"/></svg>`;
}
function syncMotion(clip){
 if(clip.visible&&!clip.userPaused&&!document.hidden){
  if(!clip.video.getAttribute('src'))clip.video.src=clip.video.dataset.src;
  clip.video.play().catch(()=>setMotionIcon(clip));
 }else clip.video.pause();
 setMotionIcon(clip);
}
const motionObserver=new IntersectionObserver(entries=>{entries.forEach(entry=>{const clip=motionClips.find(c=>c.video===entry.target);clip.visible=entry.isIntersecting;syncMotion(clip);});},{threshold:.08});
motionClips.forEach(clip=>{
 clip.video.muted=true;
 clip.button.addEventListener('click',()=>{clip.userPaused=!clip.video.paused;syncMotion(clip);});
 clip.video.addEventListener('play',()=>setMotionIcon(clip));
 clip.video.addEventListener('pause',()=>setMotionIcon(clip));
 motionObserver.observe(clip.video);
});
document.addEventListener('visibilitychange',()=>motionClips.forEach(syncMotion));
motionPreference.addEventListener('change',e=>{motionClips.forEach(clip=>{clip.userPaused=e.matches;syncMotion(clip);});});





// Scene information is revealed on hover, keyboard focus, or a touch tap.
const sceneTasks = [{"id": "arena-1-1", "benchmark": "ALFRED", "model_label": "GPT-6 Astra (low)", "display_instruction": "Place the salt shaker in the cabinet."}, {"id": "arena-1-2", "benchmark": "ALFWorld", "model_label": "GPT-6 Astra (low)", "display_instruction": "Fill up a mug with water and take to the coffee machine."}, {"id": "arena-1-3", "benchmark": "DiscoveryWorld", "model_label": "GPT-6 Astra (low)", "display_instruction": "Excavate three sites, date the artifacts, and mark the site with the oldest artifact."}, {"id": "arena-1-4", "benchmark": "HumanCLAW", "model_label": "GPT-6 Astra (low)", "display_instruction": "Find the chair and move until your body is touching the target, with zero distance to the target object."}, {"id": "arena-1-5", "benchmark": "CALVIN", "model_label": "GPT-6 Astra (low)", "display_instruction": ""}, {"id": "arena-1-6", "benchmark": "RLBench", "model_label": "GPT-6 Astra (low)", "display_instruction": "Hockey"}, {"id": "arena-1-7", "benchmark": "ManiSkill", "model_label": "GPT-6 Astra (low)", "display_instruction": "Anymal C-Spin"}, {"id": "arena-1-8", "benchmark": "CLIPort", "model_label": "GPT-6 Astra (low)", "display_instruction": "Push the pile of orange blocks into the purple square"}, {"id": "arena-1-9", "benchmark": "RoboCasa", "model_label": "GPT-6 Astra (low)", "display_instruction": "Sort the plastic and glass bottles into their matching groups."}, {"id": "arena-1-10", "benchmark": "RoboCasa365", "model_label": "GPT-6 Astra (low)", "display_instruction": "Gather milk, a banana, and honey to prepare a banana milkshake."}, {"id": "arena-1-11", "benchmark": "BEHAVIOR-1K", "model_label": "GPT-6 Astra (low)", "display_instruction": "Bring water."}, {"id": "arena-2-1", "benchmark": "ALFRED", "model_label": "GPT-6 Sol", "display_instruction": ""}, {"id": "arena-2-2", "benchmark": "ALFWorld", "model_label": "GPT-6 Sol", "display_instruction": "Hold the clock and turn on the lamp."}, {"id": "arena-2-3", "benchmark": "DiscoveryWorld", "model_label": "GPT-6 Sol", "display_instruction": "Identify the rust-removing chemical, clean the key, and escape from the shed."}, {"id": "arena-2-4", "benchmark": "HumanCLAW", "model_label": "GPT-6 Sol", "display_instruction": ""}, {"id": "arena-2-5", "benchmark": "CALVIN", "model_label": "GPT-6 Sol", "display_instruction": ""}, {"id": "arena-2-6", "benchmark": "RLBench", "model_label": "GPT-6 Sol", "display_instruction": "Close Laptop Lid"}, {"id": "arena-2-7", "benchmark": "ManiSkill", "model_label": "GPT-6 Sol", "display_instruction": "Peg Insertion Side"}, {"id": "arena-2-8", "benchmark": "CLIPort", "model_label": "GPT-6 Sol", "display_instruction": "Push the pile of blue blocks into the brown square"}, {"id": "arena-2-9", "benchmark": "RoboCasa", "model_label": "GPT-6 Sol", "display_instruction": "Open the left drawer."}, {"id": "arena-2-10", "benchmark": "RoboCasa365", "model_label": "GPT-6 Sol", "display_instruction": "Place the lemon and carrot in the pan without overlap."}, {"id": "arena-2-11", "benchmark": "BEHAVIOR-1K", "model_label": "GPT-6 Sol", "display_instruction": "Pack books into boxes for storage."}, {"id": "arena-3-1", "benchmark": "ALFRED", "model_label": "Claude Fable 5.1", "display_instruction": ""}, {"id": "arena-3-2", "benchmark": "ALFWorld", "model_label": "Claude Fable 5.1", "display_instruction": "Put two sets of keys in the safe."}, {"id": "arena-3-3", "benchmark": "DiscoveryWorld", "model_label": "Claude Fable 5.1", "display_instruction": "Your task is to drop the flag inside the following location: science lab."}, {"id": "arena-3-4", "benchmark": "HumanCLAW", "model_label": "Claude Fable 5.1", "display_instruction": ""}, {"id": "arena-3-5", "benchmark": "CALVIN", "model_label": "Claude Fable 5.1", "display_instruction": ""}, {"id": "arena-3-6", "benchmark": "RLBench", "model_label": "Claude Fable 5.1", "display_instruction": ""}, {"id": "arena-3-7", "benchmark": "ManiSkill", "model_label": "Claude Fable 5.1", "display_instruction": "Plug Charger"}, {"id": "arena-3-8", "benchmark": "CLIPort", "model_label": "Claude Fable 5.1", "display_instruction": "Put the orange blocks in a red bowl"}, {"id": "arena-3-9", "benchmark": "RoboCasa", "model_label": "Claude Fable 5.1", "display_instruction": "Load the cup and bowl into the dishwasher, then close the door."}, {"id": "arena-3-10", "benchmark": "RoboCasa365", "model_label": "Claude Fable 5.1", "display_instruction": "Clear the broccoli and steak from the cutting board."}, {"id": "arena-3-11", "benchmark": "BEHAVIOR-1K", "model_label": "Claude Fable 5.1", "display_instruction": "Wash a baseball cap."}, {"id": "arena-4-1", "benchmark": "ALFRED", "model_label": "Gemini 3.8 Flash", "display_instruction": "Put the green spray bottles on the toilet"}, {"id": "arena-4-2", "benchmark": "ALFWorld", "model_label": "Gemini 3.8 Flash", "display_instruction": "Put a cardboard box with keys in it on the seat of a blue chair."}, {"id": "arena-4-3", "benchmark": "DiscoveryWorld", "model_label": "Gemini 3.8 Flash", "display_instruction": "Enter the building, open all the doors, and pick up the flag."}, {"id": "arena-4-4", "benchmark": "HumanCLAW", "model_label": "Gemini 3.8 Flash", "display_instruction": ""}, {"id": "arena-4-5", "benchmark": "CALVIN", "model_label": "Gemini 3.8 Flash", "display_instruction": ""}, {"id": "arena-4-6", "benchmark": "RLBench", "model_label": "Gemini 3.8 Flash", "display_instruction": ""}, {"id": "arena-4-7", "benchmark": "ManiSkill", "model_label": "Gemini 3.8 Flash", "display_instruction": ""}, {"id": "arena-4-8", "benchmark": "CLIPort", "model_label": "Gemini 3.8 Flash", "display_instruction": "Push the pile of blue blocks into the brown square"}, {"id": "arena-4-9", "benchmark": "RoboCasa", "model_label": "Gemini 3.8 Flash", "display_instruction": ""}, {"id": "arena-4-10", "benchmark": "RoboCasa365", "model_label": "Gemini 3.8 Flash", "display_instruction": "Take the steak from the fridge and place it in the marinade."}, {"id": "arena-4-11", "benchmark": "Manipulation", "model_label": "Gemini 3.8 Flash", "display_instruction": ""}, {"id": "arena-5-1", "benchmark": "ALFRED", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-2", "benchmark": "ALFWorld", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-3", "benchmark": "DiscoveryWorld", "model_label": "Qwen 3.8 Max", "display_instruction": "Decode the village elder’s request."}, {"id": "arena-5-4", "benchmark": "HumanCLAW", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-5", "benchmark": "CALVIN", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-6", "benchmark": "RLBench", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-7", "benchmark": "ManiSkill", "model_label": "Qwen 3.8 Max", "display_instruction": "Pull Cube"}, {"id": "arena-5-8", "benchmark": "CLIPort", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-9", "benchmark": "RoboCasa", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-10", "benchmark": "RoboCasa365", "model_label": "Qwen 3.8 Max", "display_instruction": ""}, {"id": "arena-5-11", "benchmark": "RoboCasa365", "model_label": "Qwen 3.8 Max", "display_instruction": "Clear the broccoli and steak from the cutting board."}, {"id": "arena-6-1", "benchmark": "ALFRED", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": ""}, {"id": "arena-6-2", "benchmark": "ALFWorld", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": ""}, {"id": "arena-6-3", "benchmark": "DiscoveryWorld", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": "Your task is to give Yugo the spectrometer."}, {"id": "arena-6-4", "benchmark": "HumanCLAW", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": ""}, {"id": "arena-6-5", "benchmark": "CALVIN", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": ""}, {"id": "arena-6-6", "benchmark": "RLBench", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": "Take Toilet Roll Off Stand"}, {"id": "arena-6-7", "benchmark": "ManiSkill", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": "Plug Charger"}, {"id": "arena-6-8", "benchmark": "CLIPort", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": ""}, {"id": "arena-6-9", "benchmark": "RoboCasa", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": "Place the toasted item on a plate."}, {"id": "arena-6-10", "benchmark": "RoboCasa365", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": ""}, {"id": "arena-6-11", "benchmark": "RoboCasa365", "model_label": "Qwen 3.5 397B-A17B", "display_instruction": "Place the lemon and carrot in the pan without overlap."}, {"id": "arena-7-1", "benchmark": "ALFRED", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-2", "benchmark": "ALFWorld", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-3", "benchmark": "DiscoveryWorld", "model_label": "Qwen 3.5 27B", "display_instruction": "Identify the nutrient and concentration that make the plant grow."}, {"id": "arena-7-4", "benchmark": "HumanCLAW", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-5", "benchmark": "CALVIN", "model_label": "Qwen 3.5 27B", "display_instruction": "Press the button to turn off the led light"}, {"id": "arena-7-6", "benchmark": "RLBench", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-7", "benchmark": "ManiSkill", "model_label": "Qwen 3.5 27B", "display_instruction": "Plug Charger"}, {"id": "arena-7-8", "benchmark": "CLIPort", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-9", "benchmark": "RoboCasa", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-10", "benchmark": "RoboCasa365", "model_label": "Qwen 3.5 27B", "display_instruction": ""}, {"id": "arena-7-11", "benchmark": "Manipulation", "model_label": "Qwen 3.5 27B", "display_instruction": ""}];
const sceneTooltip=document.querySelector('#scene-tooltip');
let activeScene=null,hideSceneTimer=null;
const tooltipBenchmark=sceneTooltip.querySelector('.scene-tooltip-benchmark');
const tooltipModel=sceneTooltip.querySelector('.scene-tooltip-model');
const tooltipGoal=sceneTooltip.querySelector('.scene-tooltip-goal');
function positionSceneTooltip(button){
 const r=button.getBoundingClientRect(),t=sceneTooltip.getBoundingClientRect(),pad=12;
 let left=Math.min(innerWidth-t.width-pad,Math.max(pad,r.left+(r.width-t.width)/2));
 let top=r.bottom+10;
 if(top+t.height>innerHeight-pad)top=r.top-t.height-10;
 top=Math.min(innerHeight-t.height-pad,Math.max(84,top));
 sceneTooltip.style.left=`${left}px`;sceneTooltip.style.top=`${top}px`;
}
function revealScene(button,trigger){
 clearTimeout(hideSceneTimer);
 const task=sceneTasks.find(t=>t.id===button.dataset.scene);if(!task)return;
 if(activeScene&&activeScene!==button)activeScene.classList.remove('is-active');
 activeScene=button;button.classList.add('is-active');
 tooltipBenchmark.textContent=task.benchmark;tooltipModel.textContent=task.model_label||'';
 tooltipGoal.textContent=task.display_instruction||'';tooltipGoal.hidden=!task.display_instruction;
 sceneTooltip.classList.toggle('has-instruction',Boolean(task.display_instruction));
 sceneTooltip.hidden=false;sceneTooltip.dataset.trigger=trigger;positionSceneTooltip(button);
}
function closeScene(){clearTimeout(hideSceneTimer);sceneTooltip.hidden=true;if(activeScene)activeScene.classList.remove('is-active');activeScene=null;}
function deferSceneClose(){clearTimeout(hideSceneTimer);hideSceneTimer=setTimeout(closeScene,100);}
document.querySelectorAll('.scene-hotspot').forEach(button=>{
 button.addEventListener('pointerenter',e=>{if(e.pointerType!=='touch')revealScene(button,'hover');});
 button.addEventListener('pointermove',e=>{if(e.pointerType!=='touch'){clearTimeout(hideSceneTimer);if(activeScene!==button||sceneTooltip.hidden)revealScene(button,'hover');}});
 button.addEventListener('pointerleave',e=>{if(e.pointerType!=='touch')deferSceneClose();});
 button.addEventListener('focus',()=>revealScene(button,'focus'));
 button.addEventListener('blur',deferSceneClose);
 button.addEventListener('click',()=>revealScene(button,matchMedia('(hover:none)').matches?'tap':'click'));
});
sceneTooltip.addEventListener('pointerenter',()=>clearTimeout(hideSceneTimer));
sceneTooltip.addEventListener('pointerleave',deferSceneClose);
document.addEventListener('pointerdown',e=>{if(!e.target.closest('.scene-hotspot')&&!sceneTooltip.contains(e.target))closeScene();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeScene();});
window.addEventListener('scroll',closeScene,{passive:true});
window.addEventListener('resize',closeScene);
