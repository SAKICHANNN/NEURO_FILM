# P301 — Mono-HDR-3D real multi-exposure source readiness

## Question

Does the exact official ICML 2025 Mono-HDR-3D release expose a rights-clear,
hash-bound and group-identifiable real-camera multi-view/multi-exposure source
that can support a new capture-time observation leaf?

## Frozen source and access

- Official repository commit `80ca495897343cbcbd4feb55bfb5b887901c8aff`
  and tree `a9a65c1bdf414282dd68c18386adada9a2a01dff`.
- Exact `README.md` and `LICENSE` Git blobs only.
- The README-declared Google Drive folder may receive one metadata-level HEAD
  request. Folder/file bodies, images, poses, archives and pixels are forbidden.

## Frozen gates

1. Commit/tree/README/LICENSE identities are exact.
2. The official README explicitly declares four real scenes, 35 poses per
   scene and five exposures per pose.
3. The release explicitly states that the external dataset files are covered
   by rights suitable for this private research lane.
4. The release provides an exact asset manifest with file paths, sizes and
   cryptographic checksums.
5. Scene/group identities are explicit enough to freeze disjoint roles before
   any payload request.

Failure of rights, manifest or grouping closes the source before data access
and is not a scientific result. A pass would open only a separately frozen,
bounded one-scene source acquisition; it would not consume candidate 3.

## Claim ceiling

Zero-pixel official-source readiness only. No HDR quality, radiometric truth,
model reproduction, package/schema/capability, product or stock claim.
