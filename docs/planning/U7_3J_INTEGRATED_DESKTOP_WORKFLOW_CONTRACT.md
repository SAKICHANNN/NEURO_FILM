# U7.3J — Integrated desktop three-look workflow contract

## Question

Can the already verified recipe history, existing-output previews and strict
recipe replay operate as one bounded local browser workflow for Velvia 50,
Portra 400 and Ektar 100, without adding another renderer or weakening the
Look Approximation claim?

## Frozen implementation boundary

- Reuse the existing strict recipe-history, hash-before-decode preview and
  recipe-export implementations.
- The browser receives only a session token and opaque request identity. It
  never chooses or receives input, output, profile or repository paths.
- Each session is IPv4-loopback-only and permits exactly one successful
  create-only export.
- Existing U7.2 recipes, renderer semantics, output bytes and profile assets
  remain unchanged.
- No network origin other than the owned loopback session is allowed.

## Formal matrix

Three independent fresh-browser sessions select exactly one of:

1. Fujifilm Velvia 50;
2. Kodak Portra 400;
3. Kodak Ektar 100.

The page must embed one preview per stock only after the complete referenced
RGB16 output has passed its frozen SHA-256, PNG and decoded-pixel bounds. Each
selection must reproduce the corresponding frozen 24 MP output SHA-256.

## Required gates

- exactly three valid, distinct styles and request identities;
- exactly three distinct, hash-bound preview identities;
- no machine-local path appears in the browser payload;
- labelled stock cards, claim text, keyboard-operable submit controls and a
  live result region are present;
- desktop and narrow browser viewports preserve all three selectable cards;
- each independent selection publishes only its expected file and exact bytes;
- invalid token, unknown request, repeat submission and occupied destination
  remain fail-closed;
- output publication is create-only and owned scratch is empty after exit;
- existing U7.3I default page and strict export behavior remain compatible.

## Claim ceiling and stop rule

Passing establishes only a private local desktop workflow for the three
existing film-inspired Look Approximation baselines. It does not establish
stock accuracy, stock distinguishability, calibration, installer readiness or
public release. Any failed gate closes U7.3J without changing the renderer,
recipe, profile, stock labels or thresholds.
