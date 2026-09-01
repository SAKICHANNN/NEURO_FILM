# U7.8G Input-Batch Stage Ownership Repair Contract

## Question

Can the existing multi-input three-stock batch preserve a foreign directory
that replaces its private outer stage path during a child failure, while
retaining exact successful batch bytes and ordinary owned-stage cleanup?

## Trigger

`render_three_stock_input_batch_to_directory` creates one UUID-named sibling
stage and recursively removes whatever occupies that path after a failure. A
deterministic current-code injection renamed the owned stage away, created a
foreign directory at the original stage path, and then failed the child
render. The failure cleanup deleted the foreign directory and its payload while
leaving the actual owned stage at its moved path.

## Scope

- Bind the outer stage directory's filesystem identity immediately after its
  successful creation.
- On failure, recursively remove the stage only while the path still resolves
  to that exact owned directory identity and remains a normal directory.
- Preserve the existing child renderer, output transaction, receipt schema,
  output bytes, recipes, manifests, ordering and CLI behavior.

## Required semantics

1. An identity-replaced stage path and every foreign member inside it are
   preserved byte exact after the injected child failure.
2. An ordinary injected child failure still removes the still-owned outer
   stage and publishes no final destination.
3. Successful one-job forward/reverse executions retain exact receipt and tree
   bytes.
4. Input, profile, statistics and guardrail files remain immutable.
5. No stage cleanup follows a symlink, junction or other reparse-point
   replacement.
6. Existing U7.8A parent behavior and evidence remain historical and are not
   rewritten as a new result.

## Non-goals and claim ceiling

This is one private process-level ownership repair for the existing outer
multi-input batch stage. It adds no renderer, format, recipe field, recovery
protocol, preview, wrapper, installer, release, calibrated-stock, physical-film
or public-product claim. It does not generalize to arbitrary foreign additions
inside a still-owned private UUID directory.

## Verification

- deterministic pre-repair destructive-race witness;
- focused replacement, ordinary-failure and successful-byte regressions;
- committed-head forward/reverse audit with exact report identity;
- relevant U7.8A tests, Ruff, compile, JSON, scoped diff and worktree ownership
  review.

## Stop rule

One prospective repair and committed-head audit only. Do not chain adjacent
preview, resumable, child-stage, wrapper, format or source leaves from this
result.
