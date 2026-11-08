# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-11-08
- Primary surfaces: homepage, GeoProbe, research findings, research topics.
- Evidence reviewed: public/styles.css, homepage.css, research-pages.css, index.html, geoprobe.html, findings.html, research-topics.html; tests/test_public_discovery.py; published screenshots in outputs/discovery-followup-2026-11-08.
- Observed: the homepage has rich experimental media; research pages rely on long prose and tables. Inference: research readers need faster visual orientation and evidence-to-citation paths.

## Brand
- Calm, precise, open scientific work. Preserve the existing robot identity.
- Trust signals: paper v3, ten evaluated models, explicit sample denominators, original evidence, stable links and citation metadata.
- Avoid: marketing superlatives, invented results, badge clutter, keyword stuffing, citation guarantees.

## Product goals
- Help researchers identify relevant evidence, understand its scope, inspect methods/data and cite the appropriate paper and page.
- Make four adjacent areas discoverable: robot harnesses, recursive self-improvement, robot in-context learning and agentic robotics.
- Non-goals: claim new RSI/ICL algorithms, alter experimental conclusions, replace measured results with generated imagery, force indexation.
- Success signals: readers can find a research question, its quantitative evidence and citation within the first two interactions; desktop and mobile remain readable.

## Personas and jobs
- Robotics researchers surveying a topic; reviewers checking claims; engineers assessing agent evaluation protocols.
- Jobs: understand the contribution, distinguish diagnostics from robot outcomes, inspect denominators, copy an accurate citation, compare source data.
- Contexts: desktop deep reading and mobile paper-link visits; search crawlers must receive the same meaningful static HTML.

## Information architecture
- Keep existing canonical routes, section IDs, tables and all published research content.
- Homepage: existing presentation plus a compact research entry path before deep results, without reordering top-level sections.
- Research pages: identity and version, clear question, visual evidence summary, visible page contents, complete methods/findings, citation.
- Topic page: distinguish eight measured topics from four adjacent research connections; surface both near the top.

## Design principles
- Evidence before decoration: diagrams explain real counts and do not add an inference.
- Progressive reading: concise visible overview with detailed sources still accessible and crawlable.
- Tradeoff: prioritize legibility and stable deep links over animation or complex interactions.

## Visual language
- Reuse white/off-white backgrounds, ink #203542, purple #706c9f, sage #739272 from existing tokens.
- Manrope body and Space Grotesk identity; restrained heading scale and ample line height.
- Keep prose near 65-80 characters, wider data figures/tables; a consistent spacing rhythm and light borders.
- Rounded panels only when they group evidence; minimal shadows and no arbitrary gradients.
- Motion is optional enhancement and respects reduced motion.
- Use existing scientific figures or semantic HTML/SVG diagrams; no stock robot art.

## Components
- Reuse existing button, table-scroll, scope-note, research-grid and evidence-card patterns.
- Add concise overview metrics, evidence diagrams, page contents and citation actions via research-pages.css and a small defensive research-pages.js.
- Home entry components remain scoped to homepage.css. Do not add homepage app.js to independent pages.
- Static links and citation text work without JavaScript; copying is an enhancement.

## Accessibility
- Target WCAG 2.2 AA where applicable: readable contrast, visible focus, semantic headings, meaningful link names.
- All navigation and copy actions keyboard operable. Status messages use a polite live region, with usable failure fallback.
- Figures have captions and textual count equivalents; distinguish counts with text, not color alone.
- Respect reduced motion; no autoplay added to research pages.

## Responsive behavior
- Validate approximately 390px, 768px and 1440px widths.
- Stack overview/evidence columns on small screens; tables may scroll within their container, never overflow the page.
- Page contents must remain reachable without covering content; anchor offsets account for fixed navigation.
- Touch controls remain easy to target; no content revealed only on hover.

## Interaction states
- Loading: content is static; avoid a loading dependency for important facts.
- Empty/error: failed copy leaves source text and downloadable BibTeX available; do not falsely show success.
- Success: announce copied citation or section link briefly; repeated clicks work.
- Disabled: do not display nonfunctional actions as active controls.
- Slow/offline: text, counts, source links and captions remain available in loaded HTML without script execution.

## Content voice
- Clear English scientific prose, precise labels, short research questions and explicit boundaries.
- Counts: 168 queries, 130 unique RGB images and ten protocols are different quantities, not a funnel.
- CALVIN: 100 retained episodes = ten sequence IDs times ten models; 65 have execution errors, 62 resume, three complete all five goals. Do not imply 62 completed or a causal recovery effect.
- Planning: 126/137/142 of 157 are retrospective success unions, not a learned routing policy.
- Adjacent topic cards describe relevance of measured evidence, not experimental claims about those fields.

## Implementation constraints
- Static HTML/CSS, small vanilla JavaScript, no new dependencies.
- Preserve paper version, canonical/verification metadata, schema meaning and source data.
- Keep XML/TXT canonical URL sets equal; only change lastmod for actual content changes.
- Run existing public Python and JavaScript checks, plus focused meaningful tests for new copy behavior and quantitative diagrams.
- Capture actual desktop/mobile pages and test links/copy behavior before publication; verify deployment and live bytes afterward.

## Open questions
- None blocking. Working assumption: preserve the established English academic identity and use actual data rather than broaden experimental claims to adjacent fields.
