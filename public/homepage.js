// The decorative wall uses the same viewport and reduced-motion policy as cases.
(() => {
  'use strict';
  const button=document.querySelector('[data-wall-motion]');
  if(!button)return;
  const videos=[...document.querySelectorAll('.hero-wall video')];
  function label(){
    const playing=videos.some(video=>!video.paused);
    button.textContent=playing?'Ⅱ Pause background':'▶ Play background';
    button.setAttribute('aria-label',playing?'Pause background motion':'Play background motion');
    button.setAttribute('aria-pressed',String(!playing));
  }
  button.addEventListener('click',()=>{
    window.ArenaMedia.toggle(videos,!videos.some(video=>!video.paused));
    label();
  });
  videos.forEach(video=>['play','pause'].forEach(event=>video.addEventListener(event,label)));
  label();
})();
