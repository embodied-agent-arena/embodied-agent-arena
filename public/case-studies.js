(() => {
  const dialog = document.getElementById('study-dialog');
  if (!dialog) return;
  const cards = [...document.querySelectorAll('.study-card')];
  const filters = [...document.querySelectorAll('[data-study-filter]')];
  const search = document.getElementById('study-search');
  let active = filters.find(b => b.getAttribute('aria-pressed') === 'true')?.dataset.studyFilter || 'all';
  const applyFilter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    let n = 0;
    cards.forEach(card => {
      const show = (active === 'all' || card.dataset.domain === active) && (!query || card.dataset.search.toLowerCase().includes(query));
      card.hidden = !show;
      if (show) n++;
    });
    const count = document.getElementById('study-count');
    if (count) count.textContent = `${n} ${n === 1 ? 'case' : 'cases'}`;
  };
  filters.forEach(button => button.addEventListener('click', () => {
    active = button.dataset.studyFilter;
    filters.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    applyFilter();
  }));
  search?.addEventListener('input', applyFilter);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const showValue = value => typeof value === 'object' ? JSON.stringify(value) : String(value);
  const budgetNames = {
    max_code_turns:'Code rounds',max_agent_iterations:'Model rounds',max_total_tokens:'Tokens',trial_timeout_seconds:'Time ceiling (s)',timeout_seconds:'Time ceiling (s)',max_env_steps:'Environment steps',max_primitive_calls:'Primitive calls',max_verifier_calls:'Verifier calls',code_timeout_seconds:'Per-cell time (s)',max_agent_attempts:'Agent attempts',llm_num_retries:'Request retries'
  };
  let records, currentUrl, trigger;
  const load = async () => {
    if (!records) {
      const response = await fetch('data/case-studies.json');
      if (!response.ok) throw new Error('Case details could not be loaded.');
      records = new Map((await response.json()).map(c => [c.id, c]));
    }
    return records;
  };
  const open = async (id, button) => {
    trigger = button;
    const content = document.getElementById('study-dialog-content');
    content.innerHTML = '<h2 id="study-dialog-title">Loading Case…</h2>';
    dialog.showModal();
    document.body.classList.add('study-modal-open');
    try {
      const c = (await load()).get(id);
      if (!c) throw new Error('Case not found.');
      let media = c.video ? `<video controls playsinline preload="metadata" poster="${esc(c.poster)}" src="${esc(c.video)}"></video>` : `<div class="study-dialog-media">${(c.animation ? [c.animation] : c.images).map((src,i) => `<img src="${esc(src)}" alt="${esc(c.benchmark)} ${c.animation ? esc(c.mediaKind) : `input ${i+1}`}">`).join('')}</div>`;
      if (c.comparisonFigure) media = `<a class="study-comparison-link" href="${esc(c.comparisonFigure)}" target="_blank" rel="noopener" aria-label="Enlarge task, reference, and model comparison"><img src="${esc(c.comparisonFigure)}" alt="${esc(c.benchmark)}: task input, reference, and first-run model comparison"></a>`;
      const rows = (c.comparisonFigure ? [] : c.predictions).map(p => `<tr><td>${esc(p.model)}</td><td>${esc(showValue(p.answer))}</td></tr>`).join('');
      const budget = Object.entries(c.budgets || {}).filter(([k,v]) => budgetNames[k] && v !== null).map(([k,v]) => `<li><span>${budgetNames[k]}</span><strong>${Number(v).toLocaleString()}</strong></li>`).join('');
      const runtime = c.elapsedSeconds != null ? ` · ${Number(c.elapsedSeconds).toFixed(1)} s elapsed` : '';
      const rounds = c.modelRounds != null ? ` · ${c.modelRounds} model responses` : '';
      content.innerHTML = `<p class="eyebrow">${esc(c.domain)} · ${esc(c.benchmark)}</p><h2 id="study-dialog-title">${esc(c.title)}</h2><p class="study-meta">${esc(c.model)} · ${esc(c.mediaKind)}</p><p class="study-question">${esc(c.shortTask)}</p>${c.options?.length ? `<p>${c.options.map(v => esc(showValue(v))).join(' · ')}</p>` : ''}${media}<p class="study-outcome">${esc(c.result)}${esc(rounds + runtime)}</p>${c.reference != null && !c.comparisonFigure ? `<h3>Reference</h3><pre>${esc(JSON.stringify(c.reference,null,2))}</pre>` : ''}${rows ? `<h3>First-Run Predictions</h3><div class="study-table-wrap"><table><thead><tr><th>Model</th><th>Answer</th></tr></thead><tbody>${rows}</tbody></table></div>` : ''}${budget ? `<h3>Recorded Budget Ceilings</h3><ul class="study-budget-list">${budget}</ul>` : ''}<h3>${esc(c.instructionKind || "Original Instruction")}</h3><p class="study-original-instruction">${esc(c.question)}</p><p class="study-protocol">${esc(c.protocolNote)}</p><a class="study-download" id="study-json-download">Download Case Details (JSON) ↓</a>`;
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl = URL.createObjectURL(new Blob([JSON.stringify(c,null,2)],{type:'application/json'}));
      const download = document.getElementById('study-json-download');
      download.href = currentUrl;
      download.download = c.id + '.json';
    } catch (error) {
      content.innerHTML = `<h2 id="study-dialog-title">Case Details Unavailable</h2><p>${esc(error.message)} Please reload the page and try again.</p>`;
    }
  };
  document.querySelectorAll('[data-case]').forEach(button => button.addEventListener('click', () => open(button.dataset.case, button)));
  dialog.addEventListener('close', () => {
    dialog.querySelectorAll('video').forEach(v => v.pause());
    document.getElementById('study-dialog-content').replaceChildren();
    document.body.classList.remove('study-modal-open');
    trigger?.focus();
  });
  applyFilter();
})();
