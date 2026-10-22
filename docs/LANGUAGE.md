# Language and original records

The project page, Case Atlas, README, and public documentation use English. GeoProbe instructions originally written in Chinese have English display translations in the case viewer, including their coordinate conventions, units, output fields, and task-specific inputs.

These translations do not change the evaluation. The [original prompts](../public/data/source-instructions.json) retain the exact task IDs and SHA-256 hashes of the original question strings. The corresponding case viewer labels the translated instructions and provides a download link to the originals. Model answers, reference values, scores, and report hashes remain unchanged.

The code repository preserves the original evaluation prompts in `src/embodied_harness/vlm_protocols/` and multilingual answer-parser fixtures. Execution-log downloads also preserve model-generated code and errors exactly, including any non-English tokens. These are source records, not editorial explanations or interface labels.

Rebuild the display translations and search index from the repository root:

```bash
python3 scripts/build_english_case_instructions.py
python3 scripts/build_case_atlas.py
python3 -m unittest discover -s tests -p 'test_public_language.py'
node tests/check_case_rendering.cjs
```
