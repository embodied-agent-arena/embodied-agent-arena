// Keep evidence links on the current page and select the promised comparison.
(() => {
  'use strict';
  const menu = document.querySelector('.mobile-nav');
  menu?.addEventListener('click', event => {
    if (event.target.closest('a')) menu.open = false;
  });
  menu?.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      menu.open = false;
      menu.querySelector('summary').focus();
    }
  });
  document.addEventListener('click', event => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest('[data-domain-link], [data-execution-link], [data-report-link]');
    if (!link) return;
    let target;
    if (link.dataset.domainLink) {
      target = document.getElementById('findings');
      document.getElementById(`domain-tab-${link.dataset.domainLink}`)?.click();
    } else if (link.dataset.executionLink) {
      target = document.getElementById('execution-analysis');
      document.dispatchEvent(new CustomEvent('execution-select', {detail: link.dataset.executionLink}));
    } else {
      target = document.getElementById('technical-report');
      document.dispatchEvent(new CustomEvent('report-select', {detail: link.dataset.reportLink}));
    }
    if (!target) return;
    event.preventDefault();
    const url = new URL(location.href);
    url.hash = target.id;
    if (link.dataset.executionLink) url.searchParams.set('analysis', link.dataset.executionLink);
    history.replaceState(null, '', url);
    target.scrollIntoView({behavior: link.dataset.reportLink || matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  });
})();
