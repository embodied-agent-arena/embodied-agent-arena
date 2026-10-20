const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const pub=path.resolve(__dirname,'../public');
const context={document:{getElementById:()=>({})},fetch:async url=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(pub,url.split('?')[0]),'utf8'))})};
vm.createContext(context);
const source=fs.readFileSync(pub+'/case-studies.js','utf8').split("  const content = document.getElementById('study-dialog-content');")[0];
vm.runInContext(source+'globalThis.checks={load,caseSlide,executionRecords,boxCases,lookup:id=>records.get(id)};})();',context);
(async()=>{
 const cs=await context.checks.load();let executions=0,staticCount=0;
 for(const c of cs){
  const html=context.checks.caseSlide(c,0,cs.length);
  assert(!html.includes('NaN'));assert(!html.includes('undefined'));
  if(!c.predictions.length){
   executions++;assert.equal((html.match(/class="gallery-run /g)||[]).length,10,c.id);
   assert.equal(context.checks.executionRecords(c).length,10);
   assert(!html.includes('gallery-record-strip'));
  }else {staticCount++;assert.equal((html.match(/class="gallery-model /g)||[]).length,10,c.id);}
 }
 const original=JSON.parse(fs.readFileSync(pub+'/data/case-studies.json'));
 for(const c of original.filter(c=>!c.predictions.length))assert.equal(context.checks.lookup(c.id).taskId,c.sample);
 assert.throws(()=>context.checks.executionRecords({runIds:[],taskId:'wrong'}));
 console.log(`Rendered ${executions} execution tasks + ${staticCount} fixed-input tasks: exactly ten model cells each. All old links resolve to their exact task.`);
})().catch(e=>{console.error(e);process.exitCode=1;});
