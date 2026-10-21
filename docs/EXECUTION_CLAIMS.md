# Claim ledger

| Claim | Value | Scope / calculation | Data field | Manuscript | Website |
|---|---|---|---|---|---|
| Shared failures | 15/157; 8/157 common successes; union 142/157 | 1,570 Planning report outcomes; exact task_id match | planning-outcomes.json | Execution appendix | Shared Failures |
| Complementarity | 142 − 126 = 16 tasks beyond Astra | Same-task union; descriptive, not a tested selector | planning-outcomes.json | Execution appendix | Shared Failures |
| Failed-episode token share | 55.5–86.9%; Astra 55.5% | Sum observed input/output tokens; denominator only records with usage | planning.models | Execution appendix | Time & Tokens |
| Recovery | Opus&nbsp;5.5: 48/105; Gemini 4 Argon: 16/28 error episodes eventually complete | Native success after earlier execution error, exclude budget guards/timeouts | planning.models | Execution appendix | Error Recovery |
| CALVIN progress | Opus 5/5,338 action calls; Gemini 4 Argon 5/5,475 action calls; Gemini&nbsp;3.8 0/5,501 action calls | Same sequence0526, per-record settings, native verifier and gateway traces | data/execution | Execution appendix | Execution explorer |
| Representation | Polygon IoU greater for9/10 models | 70 identical task inputs; one polygon run vs repeated boxes | profiles | Existing Affordance appendix + report | Representation & Feedback |
