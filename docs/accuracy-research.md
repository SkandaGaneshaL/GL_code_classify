# Segment 3 accuracy research basis

## What is measured

Segment 3 accuracy is exact agreement between the predicted natural account and
a Finance-approved final posting. Model self-reported confidence, retrieval
similarity, and token logprobs are diagnostic signals; none is reported as
accuracy without held-out validation.

The scorecard reports strict accuracy (including abstentions), selective
accuracy, coverage, accepted precision, per-class precision/recall/F1, balanced
accuracy, confidence calibration, and retrieval ranking metrics. Every rate
must display its numerator, denominator, and Wilson 95% interval.

## Operating model

Oracle's Intelligent Account Combination Defaulting documentation separates
coverage from prediction accuracy, uses complete invoice header/line attributes,
and leaves a segment blank when its internal confidence threshold is not met.
This project follows the same review-safe pattern. See the
[Oracle IACD FAQ](https://docs.oracle.com/en/cloud/saas/financials/26b/fappp/faqs-for-intelligent-account-combination-defaulting.html).

OpenAI's logprob guidance explicitly distinguishes model confidence from
correctness and requires task-specific evaluation before thresholding. See the
[OpenAI logprobs cookbook](https://developers.openai.com/cookbook/examples/using_logprobs)
and [Evals guide](https://developers.openai.com/api/docs/guides/evals).

## Source audit

Run `research_ingestion.py` with all supplied source files to create a complete
machine-readable audit. The job preserves each URL's originating file, redirect,
HTTP result, content hash, retry count, and evidence grade. Invalid and blocked
URLs remain in the report; they are never silently discarded.

Only primary vendor documentation, official product documentation, peer-reviewed
research, or first-party engineering material can justify a production policy.
Landing pages and vendor marketing claims are tagged as discovery material until
their underlying methodology is independently supported.
