# U2.5B profile-driven safe-Lab adapter results

Date: 2026-07-18

Decision: **pass — retain explicit current-profile compatibility adapter**

## Implemented boundary

`render_film.py` now accepts the explicit `--use-render-profile` flag. When
selected, the renderer validates the complete profile, referenced assets and
requested style before pixel decoding, then supplies the profile's existing
colour parameters to the unchanged safe-Lab kernel. Effects and export remain
on the established path.

Profile validation also moves before rendering whenever `--write-recipe` is
requested. Invalid provenance can no longer leave an image before recipe
construction fails. Metrics record `profile_driven_adapter` as a boolean. The
default remains the legacy YAML-backed path.

## Parity and failure evidence

- all eight current safe-rich styles produce byte-identical sRGB8 PNGs between
  legacy and profile-driven paths;
- Ektar100, HP5 and Velvia50 produce sample-identical sRGB16 PNGs;
- repeated profile-driven renders are byte-identical;
- a profile-driven recipe verifies against the selected profile and assets;
- invalid evidence escalation, asset hashes and missing styles create zero
  image, metrics, recipe or layer outputs;
- invalid recipe-only profiles also fail before image output;
- default metrics explicitly report the adapter as disabled;
- U2.5A B&W and all existing output/replay tests remain green.

The complete suite initially exposed a historical-test defect: G4H compared
all future renderer changes to its old pre-contract head. Commit `a8545b3`
now replays that frozen gate at its actual evaluation commit while separately
checking that the staged-density research adapter still has zero current
production imports. The G4H config and historical result were not changed.

## Reproducibility

- contract config SHA-256:
  `5c62c5484d3cfd5b0731d9cf18cd6e0ab78ffb213f661355114d4dc434c1e766`;
- historical-test repair commit:
  `a8545b3`;
- adapter implementation commit:
  `7390d080881c9d6f005dd6750a72cac024a42fb0`;
- profile schema remains
  `3ff0fdaa59f4c0b2f9787707b6556c16391a25c9d515f7d7487e757953c19780`;
- recipe schema remains
  `e3bfda5d6f9a5f9bf8f96e20f5e732fcfb79d2ebc0d71f91151d4a71ff4803d5`;
- safe-rich profile remains
  `72a9948e6fc76e230c4593b847728712bc034840cd339e6b34f079ac7d424c79`;
- 45 focused/adjacent and 675 complete CPU tests pass;
- compile and diff checks pass.

## Claim ceiling

This closes U2.5 for the current safe-rich v1 compatibility scope only. It
does not promote the opt-in to default, validate arbitrary future profiles,
change image quality, add an operator, upgrade evidence or establish stock
authenticity/calibration.

