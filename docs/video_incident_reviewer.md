# Video Incident Reviewer

This is the honest product direction if the goal is "upload one video and get useful work back."

The MOSEI sentiment models in this repo do not solve that. They classify aligned sentiment features. They are not a general uploaded-video worker.

## What This Path Does

Input:
- one video file

Output:
- a strict JSON report with:
  - factual summary
  - timestamped events
  - detected violations
  - recommended actions
  - a human-review flag when the evidence is ambiguous

## Current Scope

The first narrow use case is incident / safety / compliance review. That is specific enough to be useful and generic enough to adapt to warehouse, site-safety, dashcam, or operations footage.

## Run It

Install the optional dependency:

```bash
pip install -e '.[video]'
```

Set your API key:

```bash
export GEMINI_API_KEY=...
```

Run the reviewer:

```bash
python scripts/review_incident_video.py \
  --video path/to/video.mp4 \
  --focus safety \
  --checklist "Worker wears required PPE" \
  --checklist "No blocked exit" \
  --checklist "No unsafe lifting"
```

This writes `path/to/video.incident_report.json` by default.

## Hard Limits

- This path depends on Gemini video understanding, not the local MOSEI models.
- The output is structured, not guaranteed true. High-stakes use still needs human review.
- Fast action can be missed if the provider downsamples video aggressively.

## Why This Exists

Because "real work from one uploaded video" is a product problem, not a sentiment-benchmark problem.
