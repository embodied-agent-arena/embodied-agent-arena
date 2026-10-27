# Research discovery and citation maintenance

The canonical website is https://embodied-agent-arena.github.io/embodied-agent-arena/.
The paper cited by the current research pages is arXiv:2610.00854v3, published
23 October 2026. Page maintenance on 27 October does not change experimental results.

## Sources and citation exports

- `public/data/study.json` defines the paper identity, published version, authors,
  model identifiers, and main result values.
- `public/data/citation.bib` defines the preferred paper citation, including its
  DOI and version-specific arXiv URL.
- Run `python3 scripts/build_public_results.py` after changes to these sources.
  It synchronizes HTML, README, CFF, CSV provenance, and the downloadable
  `public/citation.bib` and `public/citation.csl.json`.
- `public/geoprobe.html` describes the 168-query / 130-image / ten-protocol subset.
  Its counts and definitions cite arXiv v3 Table A2 and Appendix B.3–B.4.
- `public/findings.html` gives stable section links, population definitions,
  evidence downloads, and a fixed Git commit for the cited data.
- Update `public/sitemap.xml` when a page's substantive content changes. Preserve
  the previous date on unchanged pages. Do not replace experimental dates with a
  website deployment date.
- Keep `public/sitemap.txt` identical to the XML's ordered canonical URL list,
  one absolute URL per line in UTF-8. Tests enforce this correspondence. It is a
  supported alternative submission format for diagnosing sitemap ingestion;
  it does not replace the XML's modification dates or change canonical URLs.

## Canonical and legacy hosting

The main website is deployed from `public/` by `.github/workflows/pages.yml`.
`wrangler.jsonc` deploys only `scripts/legacy-redirect.mjs` to the former
`embodied-agent-arena.haojianhuang927.workers.dev` site. This Worker returns a
permanent 301 to the corresponding canonical path and preserves query strings.
Deploy it with the existing Cloudflare account using `wrangler deploy`.

A project-level `/embodied-agent-arena/robots.txt` is not a host-root robots file.
The checked-in `deployment/github-root/` files are published by the organization
site repository `embodied-agent-arena/embodied-agent-arena.github.io` from its main
branch root. Its `/robots.txt` allows crawling and advertises the project sitemap;
the root index points visitors to the canonical project page. Keep both copies of
robots.txt identical. These files require no new account or secret.

## Validation

The homepage's `google-site-verification` meta tag verifies this URL-prefix
property in Google Search Console. Preserve the tag across rebuilds and future
deployments, including after verification succeeds. Submit `sitemap.xml` through
that property and use URL Inspection for page-specific indexing requests.
The verified root URL-prefix property (`https://embodied-agent-arena.github.io/`)
also exposes Google's robots.txt report; project-path properties cannot show it.
If XML ingestion fails despite a successful live fetch, `sitemap.txt` provides a
bounded alternative-format submission. Only a successful Sitemaps report with
discovered URLs proves ingestion; submission acknowledgements and live URL
Inspection alone do not. Avoid repeated submissions or filename changes.
The adjacent `msvalidate.01` tag verifies the same site in Bing Webmaster Tools;
preserve it so Bing URL Inspection and sitemap submission remain available.

IndexNow ownership files live at the host root in the separate organization
site repository. Keep its public key file available after submitting URLs;
an accepted submission is a crawl notification, not proof of indexing.

Run the Pages workflow's public Python tests and Node rendering checks, including
`tests/check_legacy_redirect.mjs`. `test_public_discovery.py` checks static links and
anchors, canonical URLs, sitemap entries, citations, author metadata, research
entry points and the root robots source.

After deployment, verify the canonical homepage, GeoProbe, findings, citation
downloads and sitemap return HTTP 200; the host-root robots file returns 200;
and the legacy domain returns HTTP 301 to the matching canonical path. Check
the resulting HTML without JavaScript, including requests identifying as
Googlebot, bingbot and OAI-SearchBot. User-agent checks are HTTP smoke tests,
not proof of actual search indexing or future AI citations. Verify indexing in
the relevant webmaster console when authenticated access is available.
