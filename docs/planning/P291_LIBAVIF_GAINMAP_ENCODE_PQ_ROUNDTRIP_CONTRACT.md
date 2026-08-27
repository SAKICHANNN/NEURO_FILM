# P291 libavif gain-map encode-to-PQ integration contract

## Position and question

`ULT > U1 > HDR media > P291`

Can the exact private P290 create-only encoder output, produced from the exact
P289 HDR/SDR endpoints, pass unchanged through the already-frozen official
libavif decode, P283/P87 absolute-Rec.2020 ingress and P284/P89 PQ RGB16 PNG
publication rail without material code drift or identity ambiguity?

This is the one consumer integration question opened by P289. It is not a new
encoder profile, decoder wrapper, tone policy, codec-quality experiment or
product admission.

## Frozen parents and source

- P289 encoder evidence and configuration are exact parents.
- P290 create-only API, configuration and evidence are exact parents.
- P283/P284 evidence and the existing libavif ingress/PQ publication sources
  are hash-bound and unchanged.
- The only source is P289's exact selected official libavif fixture
  `seine_hdr_gainmap_small_srgb.avif`. P289 reconstructs the exact frozen HDR
  and SDR endpoint PNGs before P290 creates the candidate media.
- Runtime is the exact official libavif v1.4.2 Windows release already retained
  by P282/P289. No download, rebuild, alternate backend or media replacement is
  allowed.

## Frozen execution

Two fresh committed-head processes execute forward and reverse validation
order. Each process must:

1. verify every parent, source, runtime and implementation identity before
   endpoint construction;
2. reconstruct the exact P289 HDR and SDR endpoint PNGs;
3. invoke `encode_gainmap_avif_create_only_v1` once into an absent temporary
   destination and require the exact P289/P290 media and receipt identities;
4. use official `avifdec` to decode the candidate base/HDR rendition and use
   official `avifgainmaputil tonemap` at headroom zero to decode its SDR
   rendition;
5. require both decoded endpoint hashes to equal the frozen P290 identities;
6. pass the decoded HDR RGB16 samples through the unchanged
   `prepare_libavif_gainmap_match_view_v1`, using source primaries 1, transfer
   16, full range and higher role `base`;
7. publish that owned absolute-Rec.2020 array twice with the unchanged
   `save_absolute_rec2020_cdm2_to_pq_rgb16_png`, requiring repeat-exact bytes,
   sample readback and create-only rejection;
8. compare the published RGB16 samples against both the decoded candidate HDR
   samples and the direct frozen P289 HDR endpoint samples; and
9. delete all owned media/PNG scratch and verify sources/runtimes unchanged.

The alternate SDR endpoint is identity-checked but is not transformed into an
HDR MatchView. P291 does not invent an SDR-to-HDR tone policy.

## Frozen gates

- all parent/config/source/runtime/source-code identities exact;
- candidate media, receipt and both decoded endpoints exact P289/P290;
- decoded HDR and direct HDR endpoint differ by at most 17 uint16 codes per
  component, with p95 at most 16 codes (the parent one-code 12-bit envelope);
- P87/P89 PQ publication roundtrip against the decoded HDR is median/p95/max
  at most `0/0/1` uint16 code;
- end-to-end published PQ versus the direct P289 HDR endpoint is median at most
  1, p95 at most 16 and maximum at most 18 uint16 codes;
- CICP/profile/geometry/range/ownership/immutability exact;
- repeat PNG bytes and decoded sample hashes exact;
- foreign destination, wrong media hash, wrong role and nonfinite publication
  controls reject before replacing or publishing an output;
- forward/reverse reports byte-exact; and
- zero persistent candidate media, PNG or stage residue.

## Stop rule and claim ceiling

Any failed gate closes this exact P289/P290-to-P283/P284 integration without
changing codec parameters, matrices, transfer functions, quantization rules,
thresholds, fixture or backend. No clipping, metadata rewriting, tolerance
rescue or second fixture is permitted.

A pass proves only one private Windows end-to-end arithmetic/container
integration for one exact official fixture. It does not establish an SDR tone
policy, complete ISO 21496-1 conformance, arbitrary AVIF/HEIF or HDR quality,
display validation, public dependency/API/package/schema/capability, default
loader, product admission, film-stock evidence or candidate-3 consumption.
