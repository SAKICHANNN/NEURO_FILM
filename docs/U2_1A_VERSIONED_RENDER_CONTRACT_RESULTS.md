# U2.1A Versioned Render Contract Results

**Date:** 2026-07-17  
**Node:** `ULT > U2.1 > U2.1A`  
**Decision:** **pass** — promote the v1 identity and replay envelope; broader U2.1 remains active.

## Delivered contract

- strict Draft 2020-12 profile and recipe schemas;
- dependency-free fail-closed validators and a local-file replay verifier;
- an exact, programmatically checked migration of all eight legacy safe-rich styles;
- immutable hashes for the legacy YAML, target statistics and guardrails;
- an opt-in `--write-recipe` path that writes only after successful encoding;
- input, output, profile, asset and full software-commit identity in every recipe.

The implementation lives in `src/inference/render_contract.py`. It does not add a parallel renderer and the default CLI remains recipe-free and byte-compatible.

## Evidence and claim ceiling

The migrated profile records:

- `film_stock_id: null`;
- `latent_mode_id: null`;
- `method: heuristic`;
- `data_evidence: none`;
- `expert_evidence: none`;
- `interpretation: look_approximation`;
- `calibrated_reference: false`.

No current style name is promoted to stock, mode, paired or calibrated truth. This leaf opens no fitting, training or LSM action.

## Reproducibility repair

Cross-process PNG bytes initially differed because Pillow generated a fresh ICC creation timestamp. The output boundary now freezes a valid `2000-01-01 00:00:00` sRGB header and clears the optional profile ID. Colour tags and the normalized semantic fingerprint are unchanged, and LittleCMS reopens the result successfully.

## Verification evidence

- focused contract/output/ingress suite: **30 passed**;
- complete CPU suite: **248 passed**;
- plain and opt-in-recipe PNGs are byte-identical;
- output SHA-256: `8fee12da0052357427fa3876752d2eee3a753d7597a0c0ac254a55cc1278b968`;
- recipe SHA-256: `96bf93d01ee2f681ef7600f564e559c98a430a2d86d2a043896bb6651340901c`;
- profile SHA-256: `72a9948e6fc76e230c4593b847728712bc034840cd339e6b34f079ac7d424c79`;
- recorded software commit: `d0197f2ad34d1feb57ec2ab211f4c7ec06f90dfc`;
- `verify_render_recipe_files` passes the profile, all immutable assets, input and output hashes.

The committed smoke used quarantined `data/film_domain/velvia_50/fl_76ef574fbe793892.jpg` only to test mechanics. It supplies no label, rights, stock-style or preference evidence. Its ignored outputs remain under `outputs/u2_1a_committed_smoke`.

## Branch decision

U2.1A closes as a verified product-foundation pass. U2.1 remains broader than this leaf: stable public API/evolution rules, additional bounded operator profiles, signing/distribution policy and any calibrated evidence route remain pending. Ultimate therefore continues to another legal ready leaf.
