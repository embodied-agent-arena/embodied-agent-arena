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
const figureSizes={"geometry": [2250, 614], "spatial": [2400, 566], "affordance": [2355, 645], "planning": [2250, 614], "manipulation": [2400, 796]};
function figureSizeAttrs(name){const size=figureSizes[name];return size?`width="${size[0]}" height="${size[1]}"`:'';}
buildTabs(domains,'domain',d=>`<div class="domain-text"><h3>${d.title}</h3><ul class="evidence">${d.evidence.map(([value,label,peer])=>`<li><strong>${value}</strong><span>${label}</span><span class="peer">${peer}</span></li>`).join('')}</ul></div><figure><button class="figure-button" data-zoom="assets/${d.id}.webp?v=model-logos-20261020" aria-label="Enlarge ${d.label} results"><img ${figureSizeAttrs(d.id)} src="assets/${d.id}.webp?v=model-logos-20261020" alt="${d.label}: ${d.caption}" loading="lazy"></button><figcaption><p class="figure-takeaway">${d.text}</p><p class="figure-caption"><strong>${d.label}.</strong> ${d.caption}</p></figcaption></figure>`);
function renderCompletionCase(c){
 const evidence=c.diagnostic?`<div class="sitting-diagnostic"><h4>Astra on the Same Eight Sitting Tasks</h4><dl>${c.diagnostic.map(([count,label])=>`<div><dt>${label}</dt><dd>${count}<span> / 8</span></dd></div>`).join('')}</dl></div>`:`<button class="figure-button" data-zoom="assets/${c.image}.webp?v=model-logos-20261020" aria-label="Enlarge ${c.label} results"><img ${figureSizeAttrs(c.image)} src="assets/${c.image}.webp?v=model-logos-20261020" alt="${c.alt}" loading="lazy"></button>`;
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
const sceneTasks = [{"id":"arena-1-1","benchmark":"ALFRED","model_label":"GPT-6 Astra (low)","display_instruction":"Place the salt shaker in the cabinet."},{"id":"arena-1-2","benchmark":"ALFWorld","model_label":"GPT-6 Astra (low)","display_instruction":"Fill up a mug with water and take to the coffee machine."},{"id":"arena-1-3","benchmark":"DiscoveryWorld","model_label":"GPT-6 Astra (low)","display_instruction":"Excavate three sites, date the artifacts, and mark the site with the oldest artifact."},{"id":"arena-1-4","benchmark":"HumanCLAW","model_label":"GPT-6 Astra (low)","display_instruction":"Find the chair and move until your body is touching the target, with zero distance to the target object."},{"id":"arena-1-5","benchmark":"CALVIN","model_label":"GPT-6 Astra (low)","display_instruction":""},{"id":"arena-1-6","benchmark":"RLBench","model_label":"GPT-6 Astra (low)","display_instruction":"Hockey"},{"id":"arena-1-7","benchmark":"ManiSkill","model_label":"GPT-6 Astra (low)","display_instruction":"Anymal C-Spin"},{"id":"arena-1-8","benchmark":"CLIPort","model_label":"GPT-6 Astra (low)","display_instruction":"Push the pile of orange blocks into the purple square"},{"id":"arena-1-9","benchmark":"RoboCasa","model_label":"GPT-6 Astra (low)","display_instruction":"Sort the plastic and glass bottles into their matching groups."},{"id":"arena-1-10","benchmark":"RoboCasa365","model_label":"GPT-6 Astra (low)","display_instruction":"Gather milk, a banana, and honey to prepare a banana milkshake."},{"id":"arena-1-11","benchmark":"BEHAVIOR-1K","model_label":"GPT-6 Astra (low)","display_instruction":"Bring water."},{"id":"arena-2-1","benchmark":"ALFRED","model_label":"GPT-6 Sol","display_instruction":""},{"id":"arena-2-2","benchmark":"ALFWorld","model_label":"GPT-6 Sol","display_instruction":"Hold the clock and turn on the lamp."},{"id":"arena-2-3","benchmark":"DiscoveryWorld","model_label":"GPT-6 Sol","display_instruction":"Identify the rust-removing chemical, clean the key, and escape from the shed."},{"id":"arena-2-4","benchmark":"HumanCLAW","model_label":"GPT-6 Sol","display_instruction":""},{"id":"arena-2-5","benchmark":"CALVIN","model_label":"GPT-6 Sol","display_instruction":""},{"id":"arena-2-6","benchmark":"RLBench","model_label":"GPT-6 Sol","display_instruction":"Close Laptop Lid"},{"id":"arena-2-7","benchmark":"ManiSkill","model_label":"GPT-6 Sol","display_instruction":"Peg Insertion Side"},{"id":"arena-2-8","benchmark":"CLIPort","model_label":"GPT-6 Sol","display_instruction":"Push the pile of blue blocks into the brown square"},{"id":"arena-2-9","benchmark":"RoboCasa","model_label":"GPT-6 Sol","display_instruction":"Open the left drawer."},{"id":"arena-2-10","benchmark":"RoboCasa365","model_label":"GPT-6 Sol","display_instruction":"Place the lemon and carrot in the pan without overlap."},{"id":"arena-2-11","benchmark":"BEHAVIOR-1K","model_label":"GPT-6 Sol","display_instruction":"Pack books into boxes for storage."},{"id":"arena-3-1","benchmark":"ALFRED","model_label":"Claude Fable 5.1","display_instruction":""},{"id":"arena-3-2","benchmark":"ALFWorld","model_label":"Claude Fable 5.1","display_instruction":"Put two sets of keys in the safe."},{"id":"arena-3-3","benchmark":"DiscoveryWorld","model_label":"Claude Fable 5.1","display_instruction":"Your task is to drop the flag inside the following location: science lab."},{"id":"arena-3-4","benchmark":"HumanCLAW","model_label":"Claude Fable 5.1","display_instruction":""},{"id":"arena-3-5","benchmark":"CALVIN","model_label":"Claude Fable 5.1","display_instruction":""},{"id":"arena-3-6","benchmark":"RLBench","model_label":"Claude Fable 5.1","display_instruction":""},{"id":"arena-3-7","benchmark":"ManiSkill","model_label":"Claude Fable 5.1","display_instruction":"Plug Charger"},{"id":"arena-3-8","benchmark":"CLIPort","model_label":"Claude Fable 5.1","display_instruction":"Put the orange blocks in a red bowl"},{"id":"arena-3-9","benchmark":"RoboCasa","model_label":"Claude Fable 5.1","display_instruction":"Load the cup and bowl into the dishwasher, then close the door."},{"id":"arena-3-10","benchmark":"RoboCasa365","model_label":"Claude Fable 5.1","display_instruction":"Clear the broccoli and steak from the cutting board."},{"id":"arena-3-11","benchmark":"BEHAVIOR-1K","model_label":"Claude Fable 5.1","display_instruction":"Wash a baseball cap."},{"id":"arena-4-1","benchmark":"ALFRED","model_label":"Gemini 3.8 Flash","display_instruction":"Put the green spray bottles on the toilet"},{"id":"arena-4-2","benchmark":"ALFWorld","model_label":"Gemini 3.8 Flash","display_instruction":"Put a cardboard box with keys in it on the seat of a blue chair."},{"id":"arena-4-3","benchmark":"DiscoveryWorld","model_label":"Gemini 3.8 Flash","display_instruction":"Enter the building, open all the doors, and pick up the flag."},{"id":"arena-4-4","benchmark":"HumanCLAW","model_label":"Gemini 3.8 Flash","display_instruction":""},{"id":"arena-4-5","benchmark":"CALVIN","model_label":"Gemini 3.8 Flash","display_instruction":""},{"id":"arena-4-6","benchmark":"RLBench","model_label":"Gemini 3.8 Flash","display_instruction":""},{"id":"arena-4-7","benchmark":"ManiSkill","model_label":"Gemini 3.8 Flash","display_instruction":""},{"id":"arena-4-8","benchmark":"CLIPort","model_label":"Gemini 3.8 Flash","display_instruction":"Push the pile of blue blocks into the brown square"},{"id":"arena-4-9","benchmark":"RoboCasa","model_label":"Gemini 3.8 Flash","display_instruction":""},{"id":"arena-4-10","benchmark":"RoboCasa365","model_label":"Gemini 3.8 Flash","display_instruction":"Take the steak from the fridge and place it in the marinade."},{"id":"arena-4-11","benchmark":"Manipulation","model_label":"Gemini 3.8 Flash","display_instruction":""},{"id":"arena-5-1","benchmark":"ALFRED","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-2","benchmark":"ALFWorld","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-3","benchmark":"DiscoveryWorld","model_label":"Qwen 3.8 Max","display_instruction":"Decode the village elder’s request."},{"id":"arena-5-4","benchmark":"HumanCLAW","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-5","benchmark":"CALVIN","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-6","benchmark":"RLBench","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-7","benchmark":"ManiSkill","model_label":"Qwen 3.8 Max","display_instruction":"Pull Cube"},{"id":"arena-5-8","benchmark":"CLIPort","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-9","benchmark":"RoboCasa","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-10","benchmark":"RoboCasa365","model_label":"Qwen 3.8 Max","display_instruction":""},{"id":"arena-5-11","benchmark":"RoboCasa365","model_label":"Qwen 3.8 Max","display_instruction":"Clear the broccoli and steak from the cutting board."},{"id":"arena-6-1","benchmark":"ALFRED","model_label":"Qwen 3.5 397B-A17B","display_instruction":""},{"id":"arena-6-2","benchmark":"ALFWorld","model_label":"Qwen 3.5 397B-A17B","display_instruction":""},{"id":"arena-6-3","benchmark":"DiscoveryWorld","model_label":"Qwen 3.5 397B-A17B","display_instruction":"Your task is to give Yugo the spectrometer."},{"id":"arena-6-4","benchmark":"HumanCLAW","model_label":"Qwen 3.5 397B-A17B","display_instruction":""},{"id":"arena-6-5","benchmark":"CALVIN","model_label":"Qwen 3.5 397B-A17B","display_instruction":""},{"id":"arena-6-6","benchmark":"RLBench","model_label":"Qwen 3.5 397B-A17B","display_instruction":"Take Toilet Roll Off Stand"},{"id":"arena-6-7","benchmark":"ManiSkill","model_label":"Qwen 3.5 397B-A17B","display_instruction":"Plug Charger"},{"id":"arena-6-8","benchmark":"CLIPort","model_label":"Qwen 3.5 397B-A17B","display_instruction":""},{"id":"arena-6-9","benchmark":"RoboCasa","model_label":"Qwen 3.5 397B-A17B","display_instruction":"Place the toasted item on a plate."},{"id":"arena-6-10","benchmark":"RoboCasa365","model_label":"Qwen 3.5 397B-A17B","display_instruction":""},{"id":"arena-6-11","benchmark":"RoboCasa365","model_label":"Qwen 3.5 397B-A17B","display_instruction":"Place the lemon and carrot in the pan without overlap."},{"id":"arena-7-1","benchmark":"ALFRED","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-2","benchmark":"ALFWorld","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-3","benchmark":"DiscoveryWorld","model_label":"Qwen 3.5 27B","display_instruction":"Identify the nutrient and concentration that make the plant grow."},{"id":"arena-7-4","benchmark":"HumanCLAW","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-5","benchmark":"CALVIN","model_label":"Qwen 3.5 27B","display_instruction":"Press the button to turn off the led light"},{"id":"arena-7-6","benchmark":"RLBench","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-7","benchmark":"ManiSkill","model_label":"Qwen 3.5 27B","display_instruction":"Plug Charger"},{"id":"arena-7-8","benchmark":"CLIPort","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-9","benchmark":"RoboCasa","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-10","benchmark":"RoboCasa365","model_label":"Qwen 3.5 27B","display_instruction":""},{"id":"arena-7-11","benchmark":"Manipulation","model_label":"Qwen 3.5 27B","display_instruction":""},{"row":8,"column":1,"id":"arena-8-1","model_label":"Claude Opus 5.5","benchmark":"ALFRED","display_instruction":"wash the pink bar by the sink, put the pink bar in the tub","caseId":"W3-alfred_official_visual-opus55-01","taskId":"alfred_official_visual--alfred_official_visual--official_eval_v1--trial_T20190908_064808_086925--0--36b8ba3ce8cf","sourceVideo":"assets/model-cases/W3-alfred_official_visual-opus55-01/clip.mp4","sourceSha256":"432ca57d653d2315bb899cd35bc1e7212a6cf7ca09f5e418d56cdb72e426e24c","reportSha256":"bf5482562fc4507bc8b406e05ac104a731e94b37af5bdafa317f44e22b9ed568","compositionSource":"assets/row-8-col-1.mp4","mediaKind":"Saved observation sequence"},{"row":8,"column":2,"id":"arena-8-2","model_label":"Claude Opus 5.5","benchmark":"ALFWorld","display_instruction":"Turn on a lamp with a metal box. ","caseId":"W3-alfworld_visual-opus55-01","taskId":"alfworld_visual--alfworld_visual--official_eval_v1--trial_T20190908_222951_616606--0--67ab2def7ff7","sourceVideo":"assets/model-cases/W3-alfworld_visual-opus55-01/clip.mp4","sourceSha256":"477f5033f70c8c5b29077ef7b9c64361c02ce55a3adeae0b78650ead74190a78","reportSha256":"75dc6480ce72b59240fdaec964c1862b7cd4dd8a927f4f61056825ac542dc8b3","compositionSource":"assets/row-8-col-2.mp4","mediaKind":"Saved observation sequence"},{"row":8,"column":3,"id":"arena-8-3","model_label":"Claude Opus 5.5","benchmark":"DiscoveryWorld","display_instruction":"You are on an archaeological dig on Planet X.  3 ancient sites have been found. Your task is to excavate the sites, and date any artifacts with the radiocarbon meter.  Then, once completed, place the red flag beside the sign of the dig site with the oldest artifact.","caseId":"W3-discoveryworld-opus55-01","taskId":"discoveryworld--discoveryworld--full_v1--Archaeology_Dating-Normal-s4--4--987dc450e2c9","sourceVideo":"assets/model-cases/W3-discoveryworld-opus55-01/clip.mp4","sourceSha256":"64a8059f77ec00f33f8f1964a0d5d1530845051ee83c6fdff010dca7e651935c","reportSha256":"c69112a98751e2dcfdfde9c09bd26f0677fe984e082823f3afa6029d7a934293","compositionSource":"assets/row-8-col-3.mp4","mediaKind":"Saved observation sequence"},{"row":8,"column":4,"id":"arena-8-4","model_label":"Claude Opus 5.5","benchmark":"HumanCLAW","display_instruction":"Find the potted plant and move until your body is touching the target, with zero distance to the target object.","caseId":"W3-humanclaw-opus55-01","taskId":"humanclaw--humanclaw--fullval--104862501_172226556-17-potted_plant--42--0bbe551ad27b","sourceVideo":"assets/model-cases/W3-humanclaw-opus55-01/clip.mp4","sourceSha256":"2234c6b5e242eaa33dcbf3b7d71f9d7e8a5990cd74c0ed921e89cfc6a2a9fc0e","reportSha256":"1bf3ba498b0096903617a3aad6e91d8ec6b288eae1648b500221ccf80104519d","compositionSource":"assets/row-8-col-4.mp4","mediaKind":"Saved observation sequence"},{"row":8,"column":5,"id":"arena-8-5","model_label":"Claude Opus 5.5","benchmark":"CALVIN","display_instruction":"Complete five subgoals in sequence. Recorded subgoals: lift the blue block from the sliding cabinet; store the grasped block in the drawer; Take the blue block from the drawer; stack the grasped block; use the switch to turn off the light bulb.","caseId":"W4-calvin-opus55-01","taskId":"calvin--calvin_native_turn_off_led--test--calvin_sequence_0526--0--8934498e94bb","sourceVideo":"assets/model-cases/W4-calvin-opus55-01/clip.mp4","sourceSha256":"692bac9d4771050dd3584af87a9f5112ceb81131dfc60f3833595f9cb160415c","reportSha256":"3d444d3e161da916afa32b06d85c13af0db632f069227494291dbbc29737df9b","compositionSource":"assets/row-8-col-5.mp4","mediaKind":"Recorded video excerpt"},{"row":8,"column":6,"id":"arena-8-6","model_label":"Claude Opus 5.5","benchmark":"RLBench","display_instruction":"Take Cup Out From Cabinet","caseId":"W4-rlbench-opus55-01","taskId":"rlbench--rlbench_reach_target_motion--test--TakeCupOutFromCabinet-variation_0--0--e03c522b2ac5","sourceVideo":"assets/model-cases/W4-rlbench-opus55-01/clip.mp4","sourceSha256":"3e22d347a9c20674a5a84d968841b57a8be8920cc7f516d892603b4b7bfba643","reportSha256":"98528ae266f3a3b21930b0d56f8fa7dae0668ef33f5b7d140c51f7f28bcaf401","compositionSource":"assets/row-8-col-6.mp4","mediaKind":"Recorded video excerpt"},{"row":8,"column":7,"id":"arena-8-7","model_label":"Claude Opus 5.5","benchmark":"ManiSkill","display_instruction":"PlaceSphere-v1","caseId":"W4-maniskill-opus55-01","taskId":"maniskill--maniskill_pickcube_pd_ee_delta_pos--test--PlaceSphere-v1--0--7547d80c0054","sourceVideo":"assets/model-cases/W4-maniskill-opus55-01/clip.mp4","sourceSha256":"0e6311f2c1352bfa54bdf2b21e34b9b05c8f75c5e5ce4d6192b3bf122f87f496","reportSha256":"366db72bcc1223ec98e30a153afddcf16e5ece60d6a1813a0da35da8bcfc3068","compositionSource":"assets/row-8-col-7.mp4","mediaKind":"Recorded video excerpt"},{"row":8,"column":8,"id":"arena-8-8","model_label":"Claude Opus 5.5","benchmark":"CLIPort","display_instruction":"packing boxes pairs unseen colors","caseId":"W4-cliport-opus55-01","taskId":"cliport--cliport_place_red_in_green--test--packing-boxes-pairs-unseen-colors-test_seed_1012--10127--09f4daa51ae7","sourceVideo":"assets/model-cases/W4-cliport-opus55-01/clip.mp4","sourceSha256":"23be3c184ea29b5168a739d69dcb6bc3e3ffabb435b2a1b9d14bd53162367807","reportSha256":"4b9bab1dae91f107019662566a361c11f45323c3e9211c0bd306913f82d4135d","compositionSource":"assets/row-8-col-8.mp4","mediaKind":"Recorded video excerpt"},{"row":8,"column":9,"id":"arena-8-9","model_label":"Claude Opus 5.5","benchmark":"RoboCasa","display_instruction":"Turn on the sink faucet.","caseId":"W4-robocasa-opus55-01","taskId":"robocasa--robocasa_start_coffee_machine_button--test--target-TurnOnSinkFaucet--0--8545661d52a6","sourceVideo":"assets/model-cases/W4-robocasa-opus55-01/clip.mp4","sourceSha256":"a9948be25346d21908e9f3700122f8ab889e8f12765673ec85daf45cf087afc0","reportSha256":"f0e82bdf845764d97b442925f857171f7db5302d3206669457e18e90ca6816fb","compositionSource":"assets/row-8-col-9.mp4","mediaKind":"Recorded video excerpt"},{"row":8,"column":10,"id":"arena-8-10","model_label":"Claude Opus 5.5","benchmark":"RoboCasa365","display_instruction":"Close the left drawer.","caseId":"W4-robocasa365-opus55-01","taskId":"robocasa365--robocasa365_turn_on_microwave_button--test--pretrain-CloseDrawer--0--67030c8fe6d0","sourceVideo":"assets/model-cases/W4-robocasa365-opus55-01/clip.mp4","sourceSha256":"3e9c7e5270bd2e7e86d8325e663bb5dd5f68d2c5620262e457fdc1f740ca20d4","reportSha256":"df45ff3914798227a72b27f67fda096e4599ad2832614adf79e51261edf9b411","compositionSource":"assets/row-8-col-10.mp4","mediaKind":"Recorded video excerpt"},{"row":8,"column":11,"id":"arena-8-11","model_label":"Claude Opus 5.5","benchmark":"CaPX–BEHAVIOR-1K","display_instruction":"outfit a basic toolbox","caseId":"W4-capx_behavior1k-opus55-01","taskId":"capx--capx_current_interface_live--test--env_configs-r1pro-b1k_outfit_a_basic_toolbox.yam--0--1cd2784f91bf","sourceVideo":"assets/model-cases/W4-capx_behavior1k-opus55-01/clip.mp4","sourceSha256":"b019b7ebf19bad29f150e972608c27c5287ccb52b09fcb52f3834d777d4dd8b4","reportSha256":"dc1082e3797be9493dfdf93254510fe7e58df88a7062c9d369d3c2f4b21379eb","compositionSource":"assets/row-8-col-11.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":1,"id":"arena-9-1","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"ALFRED","display_instruction":"wash the pink bar by the sink, put the pink bar in the tub","caseId":"W3-alfred_official_visual-deepseek-01","taskId":"alfred_official_visual--alfred_official_visual--official_eval_v1--trial_T20190908_064808_086925--0--36b8ba3ce8cf","sourceVideo":"assets/model-cases/W3-alfred_official_visual-deepseek-01/clip.mp4","sourceSha256":"ed461339b5dc19b0e73d95ef7a5c6adaef42d1a9dfce6566a96425fd05f27c80","reportSha256":"83165814a0683ca5fff84b2b6acba6343d952be2cd7956f26bb643eb6265c1b7","compositionSource":"assets/row-9-col-1.mp4","mediaKind":"Saved observation sequence"},{"row":9,"column":2,"id":"arena-9-2","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"ALFWorld","display_instruction":"Put a cleaned bar of soap on a counter.","caseId":"W3-alfworld_visual-deepseek-01","taskId":"alfworld_visual--alfworld_visual--official_eval_v1--trial_T20190907_074124_966890--0--937f07d75987","sourceVideo":"assets/model-cases/W3-alfworld_visual-deepseek-01/clip.mp4","sourceSha256":"ba2f6a607b5c012ae84214b90e2a97a323ed70a2d31ccc8c016e08b0e8e12310","reportSha256":"2359f472c528f1da3057d410d1c73dde533afa84847b8db36aa6177441d83297","compositionSource":"assets/row-9-col-2.mp4","mediaKind":"Saved observation sequence"},{"row":9,"column":3,"id":"arena-9-3","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"DiscoveryWorld","display_instruction":"You are on an archaeological dig on Planet X.  3 ancient sites have been found. Your task is to excavate the sites, and date any artifacts with the radiocarbon meter.  Then, once completed, place the red flag beside the sign of the dig site with the oldest artifact.","caseId":"W3-discoveryworld-deepseek-01","taskId":"discoveryworld--discoveryworld--full_v1--Archaeology_Dating-Normal-s4--4--987dc450e2c9","sourceVideo":"assets/model-cases/W3-discoveryworld-deepseek-01/clip.mp4","sourceSha256":"624659d26d56d7a11a5c7d3da43241249985f4427a1ed7c1fedbde8ababebf48","reportSha256":"b87667680e2752fb60f59610c51ef599a25e58b68350459b31d36186eefe3985","compositionSource":"assets/row-9-col-3.mp4","mediaKind":"Saved observation sequence"},{"row":9,"column":4,"id":"arena-9-4","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"HumanCLAW","display_instruction":"Find the couch and move until your body is touching the target, with zero distance to the target object. Finally, sit on the couch.","caseId":"W3-humanclaw-deepseek-01","taskId":"humanclaw--humanclaw--fullval--102816036-2-couch--42--657a2443a8bd","sourceVideo":"assets/model-cases/W3-humanclaw-deepseek-01/clip.mp4","sourceSha256":"3011b711920309fb1d8f8d209c068a2eed0b2fec05bd26c5b38f2d8f5ab0e46d","reportSha256":"54e1ea7f440550b14f22a2348b4d2dd5b629406de6dc5e89ad573ea27cbfd628","compositionSource":"assets/row-9-col-4.mp4","mediaKind":"Saved observation sequence"},{"row":9,"column":5,"id":"arena-9-5","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"CALVIN","display_instruction":"Complete five subgoals in sequence. Recorded subgoals: lift the blue block from the sliding cabinet; store the grasped block in the drawer; Take the blue block from the drawer; stack the grasped block; use the switch to turn off the light bulb.","caseId":"W4-calvin-deepseek-02","taskId":"calvin--calvin_native_turn_off_led--test--calvin_sequence_0526--0--8934498e94bb","sourceVideo":"assets/model-cases/W4-calvin-deepseek-02/clip.mp4","sourceSha256":"ad8764d5df75393668060de7a74c2bf85fb4fa86969362cf3c9213b36114f013","reportSha256":"5defe359986dc3000abe5d5818156093e27332b1d2122656450a5d66a20686cb","compositionSource":"assets/row-9-col-5.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":6,"id":"arena-9-6","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"RLBench","display_instruction":"Close Laptop Lid","caseId":"W4-rlbench-deepseek-02","taskId":"rlbench--rlbench_reach_target_motion--test--CloseLaptopLid-variation_0--0--f9bb8dd76402","sourceVideo":"assets/model-cases/W4-rlbench-deepseek-02/clip.mp4","sourceSha256":"fc30bae5bf1e1abc69802fec764c1337e191537b73fa33ad7fd1f76c4ece9a15","reportSha256":"086a73ba07f1b8664706a75e6fd6d37bdbe5c62b803c540f1a27585a43b06faa","compositionSource":"assets/row-9-col-6.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":7,"id":"arena-9-7","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"ManiSkill","display_instruction":"PlaceSphere-v1","caseId":"W4-maniskill-deepseek-03","taskId":"maniskill--maniskill_pickcube_pd_ee_delta_pos--test--PlaceSphere-v1--0--7547d80c0054","sourceVideo":"assets/model-cases/W4-maniskill-deepseek-03/clip.mp4","sourceSha256":"85b176f7cb32eda05245d67ca8891e0e88e14976e1ec35513c400fb4a9cf01e3","reportSha256":"5e273ef41a148ca854a9e9b53ca32a330b75ec75aca88fec287ae644830b7a1e","compositionSource":"assets/row-9-col-7.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":8,"id":"arena-9-8","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"CLIPort","display_instruction":"packing boxes pairs unseen colors","caseId":"W4-cliport-deepseek-01","taskId":"cliport--cliport_place_red_in_green--test--packing-boxes-pairs-unseen-colors-test_seed_1012--10127--09f4daa51ae7","sourceVideo":"assets/model-cases/W4-cliport-deepseek-01/clip.mp4","sourceSha256":"1d50e4fbff3765c2d4ba9e4e29f1fb764430db1408b0df4505dc90f9a3a80110","reportSha256":"4cbda27e5df56382b8d76251c194fb89f8182d1d0087ecc80311f81f7c1b1ab1","compositionSource":"assets/row-9-col-8.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":9,"id":"arena-9-9","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"RoboCasa","display_instruction":"Open the left drawer.","caseId":"W4-robocasa-deepseek-01","taskId":"robocasa--robocasa_start_coffee_machine_button--test--target-OpenDrawer--0--201936251412","sourceVideo":"assets/model-cases/W4-robocasa-deepseek-01/clip.mp4","sourceSha256":"f7c8e72edb47c976b18b524c5a8fb5ef295c95847edc88023231d6d5eafd8628","reportSha256":"0fffc6da9fd567dc324bd4ed09f621cd37f1d5684a304b6c12cb31445dc8e8d1","compositionSource":"assets/row-9-col-9.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":10,"id":"arena-9-10","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"RoboCasa365","display_instruction":"Place the bread and cheese on the cutting board.","caseId":"W4-robocasa365-deepseek-01","taskId":"robocasa365--robocasa365_turn_on_microwave_button--test--pretrain-BreadAndCheese--0--55cb7c23465c","sourceVideo":"assets/model-cases/W4-robocasa365-deepseek-01/clip.mp4","sourceSha256":"ce90469a7e633f75b815978dad4e6b5d4d088623b5f2c1434e44b0d590a04ef1","reportSha256":"0a7a7ef0878ec862aabb119481373fbda4260e9710b5eb7abb863598911a5385","compositionSource":"assets/row-9-col-10.mp4","mediaKind":"Recorded video excerpt"},{"row":9,"column":11,"id":"arena-9-11","model_label":"DeepSeek 4.1 Flash-1017","benchmark":"CaPX–BEHAVIOR-1K","display_instruction":"outfit a basic toolbox","caseId":"W4-capx_behavior1k-deepseek-01","taskId":"capx--capx_current_interface_live--test--env_configs-r1pro-b1k_outfit_a_basic_toolbox.yam--0--1cd2784f91bf","sourceVideo":"assets/model-cases/W4-capx_behavior1k-deepseek-01/clip.mp4","sourceSha256":"94cdbb04a159061b2515997a1b0f3119ed55ae794f63c4480524ed7d72b55077","reportSha256":"6bee5fd19473b2ccfb416a561699e79b2318a4d36bdb083b7e3e9979d386c051","compositionSource":"assets/row-9-col-11.mp4","mediaKind":"Recorded video excerpt"}];
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
