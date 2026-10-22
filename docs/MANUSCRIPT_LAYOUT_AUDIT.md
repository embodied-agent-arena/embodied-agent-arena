# Manuscript narrative and pagination — 23 October 2026

The current manuscript has 63 pages. All pages were rendered and inspected after the final edit. The preceding 65-page release contained several large gaps caused by fixed supplementary tables, a float barrier, and a short final page.

## Narrative

The abstract, Introduction, §§4.2–4.4 and Conclusion now connect metric-dependent rankings and complementary coverage to the requirements that remain unmet: geometric constraints, functional regions, dependent subgoals, and precision contact. The existing CALVIN prefixes and five-task ManiSkill diagnostic support the sequence and contact discussion. Feedback is interpreted through verified task progress and its resource cost. The claim-to-evidence map is in [EXPERIMENT_FINDINGS_AUDIT.md](EXPERIMENT_FINDINGS_AUDIT.md).

## Pagination

Supplementary tables can now share pages with adjacent explanations. Redundant float barriers were removed, the case-study introduction uses ordinary paragraph spacing, and the Affordance interpretation precedes its detailed table. The final case-reading guide separates static predictions from execution media. Existing appendix prose words, numerical values, formulas, captions and bibliography entries are preserved.

| Check | Final result |
|---|---|
| Entire PDF | 63 pages, visually reviewed |
| Main Table 2 / Figure 3 | Together on page 7, in that order |
| Figures 4–8 | One main figure at the top of each page, pages 8–12 |
| Experimental discussion / Conclusion | Discussion ends on page 12; Conclusion starts on page 13 |
| Main-page bottom spacing | At most 40 pt beyond the normal bottom margin |
| Large appendix gaps | Former third-page and half-page gaps eliminated; nonfinal pages at most 78 pt beyond the normal bottom margin |
| Final page | About 89 pt beyond the normal bottom margin; complete coverage section and budget table |
| Typography | Body and figure font sizes unchanged; no negative spacing or filler prose |
| Source archive | Same 102 files and 79 references; isolated compilation matches all 63 PDF pages |

The spacing figures exclude the running header and footer. Ordinary margins and space needed to keep headings with their paragraphs remain. Numeric checks locate suspect pages; they do not replace inspection of rendered pages.

## Check after future changes

1. Compile until references and pagination are stable.
2. Run `python scripts/check_manuscript_layout.py public/paper.pdf --output layout-audit.json` with PyMuPDF installed. The default 120 pt threshold flags large page-bottom gaps; the report also lists short main-text tails for review.
3. Render every page, inspect flagged pages at reading size, and check table continuations, captions, headings and page-top figures.
4. Recompile after any correction and repeat the whole-document check. Synchronize PDF page links, page counts, source archives and release hashes only after the final check.

The arXiv replacement package is prepared; no arXiv submission was performed.
