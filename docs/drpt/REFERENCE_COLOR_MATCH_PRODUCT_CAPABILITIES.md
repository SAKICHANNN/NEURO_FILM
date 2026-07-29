# Reference Match Product Capabilities V1

Status: P170 implements a read-only discovery contract for local UI and IPC
clients. It does not inspect files, authorize a render, promote an algorithm or
expand the relative-SDR media boundary.

## Contract

`reference_match_product_capabilities_payload()` composes existing public
contracts rather than defining a second renderer:

- ordered batch range: 1 through 64 sources;
- operations: fit-and-render, input inspection and recipe replay;
- accepted decoded rails: display-linear sRGB and display-linear Rec.2020;
- exact existing output capability v1 payload;
- exact existing output metadata-minimization v1 payload;
- current recipe identity and input-inspection identities;
- current delivery truth: the algorithm is not promoted, default delivery is
  identity fallback, and the research override is explicit and non-product.

The strict immutable schema is
`configs/schemas/reference_match_product_capabilities_v1.schema.json`.
Changing any nested capability, claim or default requires a new schema
version. The older `--capabilities` output remains unchanged.

Frozen SHA-256 identities:

- schema file: `7ccf46959e5b5f920981cf3fa0eef9dea85bc321db92c252f0cee539e8a4f467`;
- compact sorted UTF-8 payload:
  `8d4341f4c1aa36e0422112864bed6fc594dd4a2bbaaecf286b825ffd517ede33`.

## CLI

```powershell
python scripts/match_reference_color.py --product-capabilities
```

The command performs no file probe and emits only the canonical JSON discovery
payload. Render arguments are rejected in this mode.

Verification closes with eight dedicated tests, 50 focused CLI/report/metadata
tests, 1,010 non-manifest color/reference-match tests with five platform/data
skips, and nine immutable v40/v41 manifest tests.

The first committed schema used an equivalent root-level `const`. P171
integration review correctly rejected that representation because contract
inventory requires an explicit root `additionalProperties: false`. The schema
was rewritten without changing the accepted payload or its payload hash.

## Boundary

This is local consumer-shell discovery. It does not advertise RAW, HDR,
gain-map, video, arbitrary profiles, a promoted D-PCT candidate, target-device
runtime or main-project integration.
