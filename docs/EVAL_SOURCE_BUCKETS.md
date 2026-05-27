# Evaluation Source Buckets

The evaluation harness tracks source manifests and bucket intent, not image files.
All benchmark images, RAW files, user photos, and generated renders stay in ignored
paths such as `outputs/` or `data/eval_sources/`.

## Seed Set

`configs/eval_buckets.yaml` points to the current raw.pixls.us seed benchmark:

```text
outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/manifest.json
```

That manifest was generated from CC0/Public Domain RAW files rendered neutrally
with camera white balance, then used for the Velvia 50 deterministic preview.
The paths are local references only and are not committed.

## Manual Private Slot

Private validation photos should go under:

```text
data/eval_sources/private_manual/
```

The repository ignores `data/**`, so these images and manifests stay local. Any
manual/private manifest should use the same basic fields as generated manifests:
`before`, optional `after`, source notes, bucket labels, and license/privacy notes.
