# P164 — unsupported-media batch atomicity contract

## Question

When the second source in a two-source reference-match transaction is rejected
by an existing media-ingress rule, does the first source's already staged
render remain non-durable and does the whole transaction leave prior output,
recipe and report bytes unchanged?

## Frozen matrix

Each case uses one valid reference, one valid first source and one invalid
second source:

1. generated PNG with one transparent RGBA pixel;
2. generated two-page RGB TIFF;
3. exact pinned libultrahdr `apple_gainmap_new.jpg` fixture
   (`492a94bd...dc078`), rejected through the existing multi-frame boundary.

Each case runs twice against consumer commit `f2ea6f7`. Before invocation,
both output targets, recipe and report contain deterministic sentinel bytes.

## Gates

- the exact existing rejection type and substring are observed;
- the valid first source reaches render/encode staging before source two fails;
- every pre-existing durable target retains its exact hash;
- no new durable output, recipe or report appears;
- no stage, backup or temporary artifact remains;
- normalized case results reproduce across both runs.

## Boundary

This leaf changes no decoder or media policy. It cannot establish complete
alpha, multi-frame, MPO, Ultra HDR or ISO 21496-1 detection, and it adds no
format, HDR, gain-map or product support. It tests only the current consumer
transaction boundary for the three exact fixtures.
