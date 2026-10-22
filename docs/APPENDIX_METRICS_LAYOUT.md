# Appendix metric typography — 23 October 2026

Complete metric calculations now use display math in the existing manuscript. This is a presentation revision; evaluated records, formulas, scoring denominators, invalid-record handling, thresholds and units are unchanged.

| Appendix | Displayed calculations | PDF pages |
|---|---|---|
| B.3 Scoring and Repeated Evaluation | Exact-match criterion; residual, MAE, RMSE, AbsRel, MAPE, depth threshold accuracy; relative motion, translation and rotation errors; principal-point and trace errors; box, point and polygon grounding | 24–26 |
| C.2 Sensitivity to Source Composition | Equal-source mean | 31 |
| C.6 Interpreting the Capability Profiles | Geometry, Spatial Reasoning and Affordance indices, aligned vertically | 36 |
| D.1 Geometry | Relative error reduction | 37 |
| D.5 Manipulation diagnostics | Task success, average verified prefix, per-case and aggregate auxiliary scores, BEHAVIOR-1K scoring example | 46–48 |

Eighteen unnumbered display blocks replace embedded calculations. Previously numbered equations keep their numbers and references. Short symbol references, simple thresholds, and table entries remain inline. Related expressions share aligned blocks instead of adding separate equation numbers or dense standalone explanations.

## Validation

- The canonical PDF and source-only archive compile successfully. The independently compiled archive has identical extracted text on all 65 pages.
- No overfull boxes, missing characters or unresolved references occur.
- Main-text pages 1–13 are pixel-identical to the previous release. Table 2 and Figure 3 remain together on page 7; Figures 4–8 remain on pages 8–12.
- All figure sources/captions, measurements, abstract, bibliography and main-text source files are byte-identical to the previous release. The source inventory contains the same 102 files and 79 references.
- The formula pages were rendered and visually inspected. The project-page PDF links now target the reflowed benchmark-source appendix on page 53 and polygon protocol on page 35.
- PDF, source ZIP, arXiv preparation metadata and manuscript pointers are synchronized. arXiv submission status remains unchanged.
