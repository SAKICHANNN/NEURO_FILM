# U7.8C — Canon real-scale three-look transaction/resource contract

Status: **FROZEN BEFORE ANY SIX-SOURCE BATCH EXECUTION**

## Question

Can the unchanged U7.8A multi-input transaction process the exact six CC0
Canon sRAW/mRAW files already admitted by P313 and P314, render all three
available deterministic Look Approximations once per input, and publish one
complete directory while remaining replay-exact, source-safe, cleanup-safe and
within a bounded Windows CPU resource envelope?

This is not a mixed-format compatibility experiment.  The six inputs are one
Canon compressed-CR2 cohort.  P313 already proves their current generic
WorkingImage ingress and P314 already proves one Ektar product render plus
strict replay.  U7.8C is admissible only because U7.8A used 100 tiny synthetic
PNGs and did not measure a real-size multi-input transaction.

## Frozen implementation and inputs

- Use `render_three_stock_input_batch_to_directory` unchanged.
- Use `safe-rich-product-v1`, current film statistics and current guardrails.
- Use the six exact P313/P314 files, paths, byte lengths, SHA-256 identities and
  decoded geometries frozen in the U7.8C config.
- Render exactly `velvia_50`, `portra_400`, and `ektar_100` with look amount
  `1.0`, seed `31`, tile size `256`, one tile worker and PNG compression `0`.
- Run two fresh controllers, one supplied forward manifest order and one
  reverse manifest order.  The U7.8A canonical job ordering remains unchanged.
- Do not modify the product renderer, loader, colour math, stock parameters,
  profile, recipe schema, U7.8A transaction core or source files.

## Success evidence

The accepted formal result requires all of the following:

1. All six source hashes and byte lengths match before execution and after all
   success and failure controls.
2. Instrumentation observes exactly one `load_working_image` call for each
   source in the successful transaction and no other source decode.
3. The published transaction contains exactly six child manifests, eighteen
   RGB16 PNG outputs and eighteen strict recipes; every recipe binds its source
   hash and retains `film-inspired` / `look-approximation` claims.
4. Forward and reverse runs have identical canonical receipt, output, recipe
   and child-manifest identities.  A stable scientific identity excludes
   elapsed-time and RSS measurements.
5. A real first-child then injected second-child failure publishes nothing and
   removes the owned outer stage.
6. A complete real batch followed by an injected late foreign destination
   preserves only the foreign claim and removes the owned outer stage.
7. Final owned stage/file residue is zero; network requests are zero.
8. Each complete controller, including the two failure controls, finishes in
   at most `1200` seconds and peaks at no more than `4,294,967,296` bytes of
   process-tree RSS.  Per-job wall times are reported but have no separate
   pass threshold.

## Stop rule

The first committed-source formal result is final for this exact six-source,
profile, output-format, process, resource-gate and failure-control family.  A
failure closes U7.8C without changing sources, order, thresholds, compression,
tile settings, instrumentation, failure position or renderer.  Do not rescue a
result by adding concurrency, reducing resolution, dropping a source, changing
the loader, changing output format or excluding a slow/failing control.

## Claim ceiling

Private Windows/Python real-scale transaction and resource evidence for six
exact CC0 Canon sRAW/mRAW files and the three existing deterministic
film-inspired Look Approximations only.  No mixed-format or general RAW/Canon
support, vendor-exact colour, calibrated or physical stock response, image
quality, product value, public API/package/release, candidate-three or device
performance claim.
