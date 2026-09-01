# U7.19B — Multi-vendor RAW product-ingress matrix

## Question

Can the public generic RAW/WorkingImage boundary and unchanged Look
Approximation CLI/recipe/replay chain consume the exact already-source-locked
Nikon NEF, Fujifilm RAF, Sigma X3F, Leaf MOS and Hasselblad 3FR/FFF files that
correspond to extensions the product already declares?

This is one consolidated compatibility matrix. It is not six adjacent feature
leaves, a private stored-code callable intake, or evidence of vendor-exact
colour, calibrated stock response, physical-film reproduction or arbitrary
files from any format.

## Predecode facts and ownership

- U7.19B has decoded zero source pixels and produced zero media before this
  contract and its machine-readable config are committed.
- The twelve sources are existing exact P315/P316/P317/P319/P320 identities.
  They remain in the producer repository through its project-owned P-backed
  `data`/`outputs` junctions; U7.19B does not copy them.
- `.nef`, `.raf`, `.x3f`, `.mos`, `.3fr` and `.fff` are already present in the
  public `RAW_SUFFIXES`; no dispatcher or loader change is expected.
- Other tasks own none of the U7.19B paths. `.codex/` and `tmp/` are foreign
  existing untracked roots and remain untouched.

## Frozen strata

Each extension is an independent stratum. All members in a stratum must pass.

| Extension | Exact members | Product representative |
| --- | ---: | --- |
| `.nef` | Nikon D6 12-bit, D3S 14-bit | D6 12-bit |
| `.raf` | Fujifilm S2Pro layout 0, S7000 layout 1 | S2Pro |
| `.x3f` | Sigma SD10, DP1s | DP1s |
| `.mos` | Leaf Aptus22 compression 99, compression 1 | compression 99 |
| `.3fr` | Hasselblad X2D A and B | A |
| `.fff` | Hasselblad X2D A and B | A |

No post-result source replacement, member selection, stratum splitting,
private-callable rescue or format-specific parameter change is allowed.

## Stage A — public WorkingImage preflight

Each source runs in a fresh worker process through `inspect_input` and
`load_working_image`. Required gates:

1. exact source bytes and SHA-256 before and after;
2. public dispatch selects RAW and reports the source extension;
3. HWC3 owned, writable, C-contiguous float32 pixels;
4. finite unit interval with nonconstant content;
5. `linear_srgb` / `scene_linear`, orientation applied, alpha absent;
6. exact generic-RAW and missing-scene-tone-map warning boundary;
7. forward/reverse pixel hashes identical;
8. worker exit zero, wall time at most 600 seconds and peak process-tree RSS at
   most 16 GiB per source;
9. zero network and zero persistent media output.

A failing stratum does not block unrelated passing strata.

## Stage B — unchanged product chain

Only the frozen representative of each passing stratum may run:

- public `scripts/render_film.py`;
- explicit `ektar_100`, full amount `1.0`,
  `safe-rich-product-v1`, PNG8;
- strict recipe validation and replay to a fresh path;
- output and replay bytes exact;
- claim fields remain `Style-safe`, `film-inspired`,
  `look-approximation`, with calibrated reference forbidden;
- source immutable, network zero and owned scratch residue zero;
- wall time at most 1,800 seconds and peak process-tree RSS at most 16 GiB per
  render/replay worker.

All media and recipes are temporary under the repository-relative P-backed
`tmp` root and are removed after identities are recorded. Formal reports live
under repository-relative P-backed `outputs`.

## Stop rule and claim ceiling

Any member or product-chain failure closes that extension without rescue. If
every extension fails, U7.19B is a formal negative and makes no product change.
Passing extensions establish only exact-file generic public-ingress
compatibility. They do not establish generic format or camera support,
vendor/Adobe colour, demosaic quality, pixel-shift fusion, scene-to-display
tone mapping, calibrated film-stock response, physical-film reproduction,
multi-stock completion, public release or product-value evidence.

