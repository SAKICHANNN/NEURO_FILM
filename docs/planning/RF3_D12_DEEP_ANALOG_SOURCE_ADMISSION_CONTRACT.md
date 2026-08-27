# RF3.D12 — Deep Analog source-admission contract

## Question

Does the August 2026 *Deep Analog* paper add a rights-usable, controlled,
stock-identifying observation that can reopen a three-stock K=1 experiment, or
is it only a method/baseline precedent for single-reference LUT transfer?

## Frozen inputs

- arXiv v1 `2608.14702`, submitted 2026-08-10;
- official repository `EtonMu/deep-analog`, exact `main` commit
  `df49ad5176a1b995a33d6ffb45872f5cf781bcec`;
- only the paper PDF and public repository README/LICENSE text may be read;
- no checkpoint, demo image, dataset, model execution, training, render, or
  target score is allowed.

The source files are inspected in an owned temporary directory and are not
retained. The report binds their hashes and the exact repository identities.

## Admission gates

All gates are required:

1. the observation contains real physical-film targets rather than only
   procedural colour transforms or author-selected qualitative references;
2. it provides same-scene digital/film observations for Velvia 50, Portra 400
   and Ektar 100, or an equally discriminating multi-stock design;
3. roll, process and scanner/source groups support independent holdout;
4. an exact public data manifest and payload identities exist;
5. data and implementation rights permit the intended research and eventual
   commercial-product route;
6. a runnable public checkpoint or exact operator artifact exists;
7. the observation is materially new relative to the closed single-reference,
   after-only and image-adaptive LUT families.

## Decisions

- `PASS_NEW_CONTROLLED_STOCK_OBSERVATION`: only a separately frozen pixel and
  integrity leaf may open.
- `RETAIN_RELATED_WORK_BASELINE_ONLY`: cite the method and its reported
  limitations; do not download model/data, train, reproduce, or reopen a
  single-reference operator experiment.

This audit cannot promote a stock, model, profile, product capability, or
calibration claim.
