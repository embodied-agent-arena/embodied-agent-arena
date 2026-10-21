/* Task-level set operations shared by the explorer and its regression tests. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PlanningComparison = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';
  const completes = (task, model) => task.successModels.includes(model);
  // Importers may assign different record aliases to identical report bytes.
  const matchesEpisode = (run, episode) => Boolean(episode && run.modelKey === episode.model &&
    run.taskId === episode.taskId && run.reportSha256 && run.reportSha256 === episode.reportSha256);
  function category(task, pair) {
    const a = completes(task, pair[0]), b = completes(task, pair[1]);
    return a ? (b ? 'both' : 'a-only') : (b ? 'b-only' : 'neither');
  }
  function matches(task, filter, pair, modelCount) {
    if (filter === 'all') return true;
    if (filter === 'shared') return task.successModels.length === 0;
    if (filter === 'universal') return task.successModels.length === modelCount;
    if (filter === 'solo') return task.successModels.length === 1 && completes(task, pair[0]);
    if (filter === 'disagree') return completes(task, pair[0]) !== completes(task, pair[1]);
    return category(task, pair) === filter;
  }
  function summarize(tasks, pair, modelCount) {
    const counts = {all: tasks.length, both: 0, 'a-only': 0, 'b-only': 0, neither: 0, shared: 0, universal: 0, solo: 0};
    for (const task of tasks) {
      counts[category(task, pair)]++;
      for (const key of ['shared', 'universal', 'solo']) if (matches(task, key, pair, modelCount)) counts[key]++;
    }
    return counts;
  }
  function page(tasks, index, size) {
    const pages = Math.max(1, Math.ceil(tasks.length / size));
    index = Math.max(0, Math.min(index, pages - 1));
    return {index, pages, start: index * size, tasks: tasks.slice(index * size, (index + 1) * size)};
  }
  return {completes, matchesEpisode, category, matches, summarize, page};
});
