# U2.6A profile evidence inspection results

Date: 2026-07-18

Decision: **pass — retain validated read-only profile evidence surfaces**

## Implemented boundary

`summarize_render_profile_evidence()` is now a public inference API. It
validates the complete profile before returning a fresh dictionary containing
only the frozen identity and declared-evidence fields. The new
`scripts/inspect_render_profile.py` command loads a profile with repository
asset verification and prints the same summary as deterministic JSON.

No v1 profile or recipe schema, tracked profile, renderer path or render output
was changed.

## Current declared truth

The tracked safe-rich profile reports:

- `data_grade`: `none`;
- `expert_grade`: `none`;
- `method`: `heuristic`;
- `calibrated_reference_allowed`: `false`;
- identity interpretation: `look_approximation`;
- no stock ID and no latent mode ID.

The display surface therefore makes the current evidence ceiling more visible;
it does not upgrade it.

## Evidence

- contract config SHA-256:
  `a048906c94948dadebf2ac17ce6f24f2393a6b2ac4ea0d3a4cb442ad96c6f750`;
- implementation commit: `34134275bc5ccc483a32c9a631679e1e8548d011`;
- render-profile schema remains
  `3ff0fdaa59f4c0b2f9787707b6556c16391a25c9d515f7d7487e757953c19780`;
- render-recipe schema remains
  `e3bfda5d6f9a5f9bf8f96e20f5e732fcfb79d2ebc0d71f91151d4a71ff4803d5`;
- safe-rich profile remains
  `72a9948e6fc76e230c4593b847728712bc034840cd339e6b34f079ac7d424c79`;
- two CLI runs are byte-identical;
- evidence escalation and asset-hash tampering fail non-zero;
- returned-summary mutation does not mutate the source profile;
- 9 focused and 656 complete CPU tests pass;
- compile and diff checks pass.

## Claim ceiling

This closes U2.6's declared-evidence visibility requirement for the current v1
profile contract. It does not create a measured, paired, held-out, stock-backed
or calibrated profile; it does not validate the truth of a declaration beyond
the existing schema and asset gates; and it is not a profile editor.

