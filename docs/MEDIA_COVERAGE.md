# Media coverage

## Homepage overview

**11 fixed tasks × 10 models = 110 evaluated results.** All supplied material is matched by model, native task ID, record ID and original report SHA-256.

- **104 original-evaluation previews**: footage, saved observations, verified action replay or saved-state render. The type is shown in the case details.
- **3 same-task demos**: Astra on CALVIN, RLBench and CLIPort. Each is visibly labeled, has a separate report digest, and is excluded from evaluation. Original scores and timelines remain unchanged.
- **3 API failures before action**: Qwen Max, Qwen 397B and Qwen 27B on HumanCLAW's “Reach the chair” task. All three initial requests returned `Model API transport failure: URLError`, with no generated action or tool call. Their cells show **API error · 0 actions**, and link to the original error records. These are connection failures, not video-loading errors or observed navigation failures. The latest homepage supplement explicitly excludes these three records.

Original records: [Qwen Max](../public/data/case-records/record-8ffa102eebf9c3e0efa5.json) · [Qwen 397B](../public/data/case-records/record-bef806032b8506e12bc2.json) · [Qwen 27B](../public/data/case-records/record-6a0505ce02890f7e79e2.json).

The Qwen&#8209;27B RoboCasa365 preview is a slideshow of **24 original input RGB observations**. Its case detail lets readers select a round; intervening actions are not depicted.

[Overview manifest](../public/data/report/overview-provenance.json) · [Supplement audit](../public/data/homepage-supplement-audit.json)

## Remaining execution-library material

The wider 46-task, 460-record execution library is larger than the 11-task homepage overview. It still has **240 records without original scene footage**: 230 awaiting original recordings, 3 with separate demos only, and 7 stopped before actions. Available outcomes and logs remain accessible. No other task or model's recording is substituted.

[Exact missing-media list (CSV)](../public/data/execution/missing-original-media.csv) gives the model, benchmark, native task ID, record and report digest for each item. To supply further footage, match those identities; an independent rerun must remain labeled as a separate demonstration.
