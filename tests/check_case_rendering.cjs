const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const pub=path.resolve(__dirname,'../public');
const context={document:{getElementById:()=>({})},fetch:async url=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(pub,url.split('?')[0]),'utf8'))})};
vm.createContext(context);
const source=fs.readFileSync(pub+'/case-studies.js','utf8').split("  const content = document.getElementById('study-dialog-content');")[0];
vm.runInContext(source+'globalThis.checks={load,caseSlide,taskInstructions,executionRecords,runTile,hasPreview,displayOutcome,boxCases,lookup:id=>records.get(id)};})();',context);
(async()=>{
 const cs=await context.checks.load();let executions=0,staticCount=0;
 for(const c of cs){
  const html=context.checks.caseSlide(c,0,cs.length);
  assert(!html.includes('NaN'));assert(!html.includes('undefined'));
  assert(!/\p{Script=Han}/u.test(html),`${c.id}: non-English gallery text`);
  const instructions=context.checks.taskInstructions(c);
  assert(!/\p{Script=Han}/u.test(instructions),`${c.id}: non-English instruction text`);
  if(c.originalInstruction){
   assert(instructions.includes('English translation'));
   assert(instructions.includes('Download original evaluation prompts'));
  }
  if(!c.predictions.length){
   executions++;assert.equal((html.match(/class="gallery-run /g)||[]).length,10,c.id);
   assert.equal(context.checks.executionRecords(c).length,10);
   assert(!html.includes('gallery-record-strip'));
   assert(!html.includes('>Action log<'),`${c.id}: a log animation is not execution footage`);
   for(const [i,r] of context.checks.executionRecords(c).entries()){
    const tile=context.checks.runTile(c,r,i);
    if(r.previewPolicy==='records-only'){
     assert(!context.checks.hasPreview(r));
     assert(!tile.includes('<video'));
     assert(tile.includes('Execution record'));
    }else if(r.mediaKind==='Animated execution log'){
     assert(!context.checks.hasPreview(r));
     assert(!tile.includes(r.animation));
     assert(tile.includes(c.sceneContext.kind==='scene'?'Scene only':'Text-based task'));
    }
    if(r.executionReview) assert(tile.includes('API error'));
    if(r.outcome==='runtime_failure' && !r.executionReview) assert.equal(context.checks.displayOutcome(r),'Runtime error');
   }
  }else {staticCount++;assert.equal((html.match(/class="gallery-model /g)||[]).length,10,c.id);}
 }
 const original=JSON.parse(fs.readFileSync(pub+'/data/case-studies.json'));
 const toolbox=cs.find(c=>c.id==='showcase-capx_behavior1k');
 const toolboxHTML=context.checks.caseSlide(toolbox,0,cs.length);
 assert.equal((toolboxHTML.match(/class="gallery-camera-label"/g)||[]).length,0);
 assert(toolboxHTML.includes('Updated recordings · 24 October · Robot head camera'));
 for(const r of context.checks.executionRecords(toolbox)){
  const updated=r.updatedRecording;
  assert.equal(updated.taskId,r.taskId);
  assert.equal(updated.modelKey,r.modelKey);
  assert.equal(updated.countInEvaluation,false);
  assert(toolboxHTML.includes(updated.video));
  assert(!toolboxHTML.includes(r.video));
 }
 assert(context.checks.executionRecords(toolbox).every(r=>r.taskId===toolbox.taskId));
 const calvin=context.checks.lookup('task-video-720f338150');
 const ds=context.checks.executionRecords(calvin).find(r=>r.modelKey==='deepseek');
 const dsTile=context.checks.runTile(calvin,ds,8);
 assert(dsTile.includes('Scene only'));assert(dsTile.includes('Budget reached'));
 assert(dsTile.includes(calvin.sceneContext.image));assert(!dsTile.includes('API error'));
 for(const c of original.filter(c=>!c.predictions.length))assert.equal(context.checks.lookup(c.id).taskId,c.sample);
 assert.throws(()=>context.checks.executionRecords({runIds:[],taskId:'wrong'}));
 console.log(`Rendered ${executions} execution tasks + ${staticCount} fixed-input tasks: exactly ten model cells each. All old links resolve to their exact task.`);
})().catch(e=>{console.error(e);process.exitCode=1;});
