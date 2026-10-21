// No browser dependency: exercise viewport gating, user intent, and cleanup.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Video extends EventTarget {
 constructor(){super();this.nodeType=1;this.tagName='VIDEO';this.dataset={video:'example.mp4'};this.paused=true;this.isConnected=true;this.plays=0;this.attrs={};this.inert=false;}
 matches(){return true} querySelectorAll(){return []} closest(selector){return selector==='[inert]'&&this.inert?this:null}
 getAttribute(key){return this.attrs[key]||null}set src(value){this.attrs.src=value}
 play(){this.plays++;this.paused=false;this.dispatchEvent(new Event('play'));return Promise.resolve()}
 pause(){this.paused=true;this.dispatchEvent(new Event('pause'))}
}
const first=new Video(),doc=new EventTarget(),preference=new EventTarget();preference.matches=false;
doc.hidden=false;let modal=null,intersection,mutations;const observed=new Set();
doc.querySelector=()=>modal;doc.documentElement={nodeType:1,matches:()=>false,querySelectorAll:()=>[first]};
const sandbox={document:doc,window:{},matchMedia:()=>preference,
 IntersectionObserver:class{constructor(fn){intersection=fn}observe(el){observed.add(el)}unobserve(el){observed.delete(el)}},
 MutationObserver:class{constructor(fn){mutations=fn}observe(){}}};
vm.runInNewContext(fs.readFileSync('public/viewport-media.js','utf8'),sandbox);
const api=sandbox.window.ArenaMedia;
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const enter=(el,ratio=1)=>intersection([{target:el,isIntersecting:ratio>0,intersectionRatio:ratio}]);
(async()=>{
 assert.equal(first.plays,0);assert.equal(first.getAttribute('src'),null);
 enter(first,.19);assert.equal(first.plays,0);enter(first);await settle();assert.equal(first.paused,false);assert.equal(first.getAttribute('src'),'example.mp4');assert(first.muted&&first.loop&&first.playsInline);
 enter(first,0);assert(first.paused);enter(first);await settle();assert(!first.paused);
 first.pause();enter(first,0);enter(first);assert(first.paused,'native controls preserve user pause');
 api.toggle([first],true);await settle();assert(!first.paused);
 doc.hidden=true;doc.dispatchEvent(new Event('visibilitychange'));assert(first.paused);doc.hidden=false;doc.dispatchEvent(new Event('visibilitychange'));await settle();assert(!first.paused);
 preference.matches=true;preference.dispatchEvent(new Event('change'));assert(first.paused);api.toggle([first],true);await settle();assert(!first.paused,'manual play overrides reduced motion');
 api.enable(first,false);assert(first.paused);api.enable(first,true);await settle();assert(!first.paused);
 first.inert=true;api.refresh();assert(first.paused);first.inert=false;
 modal={contains:()=>false};api.refresh();assert(first.paused);modal=null;api.refresh();await settle();assert(!first.paused);
 const second=new Video();mutations([{addedNodes:[second]}]);assert(observed.has(second));assert.equal(second.plays,0);
 first.isConnected=false;mutations([{addedNodes:[]}]);assert(first.paused);assert(!observed.has(first));
 console.log('Viewport media: clipping, lazy loading, explicit pause, reduced motion, hidden tabs, modal/inert gating, dynamic registration and cleanup passed.');
})().catch(e=>{console.error(e);process.exit(1)});
