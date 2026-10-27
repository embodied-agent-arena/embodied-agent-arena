// Retire the old workers.dev website without leaving a stale research version online.
const canonicalBase = 'https://embodied-agent-arena.github.io/embodied-agent-arena/';

export default {
  fetch(request) {
    const incoming = new URL(request.url);
    const target = new URL(canonicalBase);
    const path = incoming.pathname.replace(/^\/+/, '');
    target.pathname += path === 'index.html' ? '' : path;
    target.search = incoming.search;
    return Response.redirect(target.href, 301);
  },
};
