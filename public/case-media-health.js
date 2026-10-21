// Preserve a visible, honest fallback if a recording cannot be loaded or decoded.
(() => {
  'use strict';
  function recover(el) {
    if(!el.matches?.('video,img') || (el.dataset.mediaFailed && !el.dataset.posterRecovery)) return;
    if(!el.closest('.study-media,.gallery-run-media,.gallery-detail-film,.overview-cell,.gallery-inputs,.exec-video-frame')) return;
    el.dataset.mediaFailed='true';
    const poster=el.getAttribute('poster') || el.dataset.poster;
    if(poster && poster!==el.getAttribute('src') && !el.dataset.posterRecovery) {
      if(el.tagName==='IMG') {el.dataset.posterRecovery='true';el.src=poster;return;}
      el.pause();el.hidden=true;
      const still=document.createElement('img');still.src=poster;
      still.alt='Recording preview; video playback unavailable';
      el.after(still);
    } else {
      delete el.dataset.posterRecovery;
      el.hidden=true;
      const message=document.createElement('span');message.className='case-media-error';
      message.textContent='Preview unavailable · Open case for the recorded result ↗';
      el.after(message);
    }
  }
  document.addEventListener('error', event => recover(event.target),true);
  document.querySelectorAll('img[src]').forEach(el=>{if(el.complete&&!el.naturalWidth)recover(el);});
})();
