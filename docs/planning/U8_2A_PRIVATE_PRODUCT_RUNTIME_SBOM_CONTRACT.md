# U8.2A — Private product-runtime SBOM contract

## Purpose

Generate two deterministic software bills of materials for the exact installed
private Look Approximation runtime:

- CycloneDX 1.7 JSON;
- SPDX 2.3 JSON.

This leaf inventories an already installed private runtime. It does not choose
a project licence, authorize public distribution, complete U8.2, or change any
renderer, look, recipe, launcher, or media path.

## Frozen input

- receipt: `outputs/private-product-runtime-u1-5h-73e720c/product-runtime.json`;
- receipt schema: `kmcfm.private-product-runtime-receipt.v3`;
- receipt SHA-256:
  `aa3a97686f9172de139f07edf0be784a23d86d9da85042d0cdda9bbbdb21f9c9`;
- installed source commit:
  `73e720c56ffe3cbe7d54868d16c1c7a8f33097a3`;
- installed requirements SHA-256:
  `78b2ec79ffd4053c42c90a5db765baf715fed261efa39a5e762e013145718517`.

The receipt and its runtime Python must resolve beneath the same installation
root. Runtime discovery must execute that exact Python with isolated startup;
the host interpreter is not an inventory source.

## Required representation

1. The application, CPython runtime, and every installed Python distribution
   are represented exactly once.
2. Package identity is normalized from installed Core Metadata and includes
   version, PURL, metadata version, `METADATA` SHA-256, and available licence
   file hashes.
3. Active installed dependency edges are evaluated using the installed
   environment. Extras that are not active are excluded.
4. A valid installed `License-Expression` may be carried forward. A legacy
   free-text `License` field, classifiers, or missing metadata is retained as
   evidence but never promoted to an inferred SPDX expression.
5. K-MCFM and CPython use `NOASSERTION` in SPDX unless an independently bound
   licence source is added by a future leaf. K-MCFM additionally records
   `public_release=false`, `film-inspired`, and `look-approximation`.
6. Both JSON files use canonical sorted-key UTF-8 plus one LF. No current time,
   random UUID, host path, user name, or environment variable may enter them.

## Gates

- exact receipt hash/schema/source/requirements binding;
- runtime Python contained by the installation root;
- receipt distributions equal their installed versions;
- installed inventory complete and duplicate-normalized names rejected;
- metadata and licence-file hashes stable across two fresh probes;
- CycloneDX 1.7 and SPDX 2.3 structural validation;
- cross-format component identities and dependency edges exact;
- project licence remains unresolved / `NOASSERTION`;
- private Look Approximation claim ceiling exact;
- forward/reverse generation byte exact;
- create-only publication preserves existing and late foreign destinations;
- owned stage/output residue zero after rejected publication.

## Stop rule

Any identity, completeness, determinism, licence, containment, or publication
gate failure closes this leaf. Do not delete foreign files, infer a licence,
drop an installed component, relax equality, regenerate the product runtime,
or claim public-release readiness as a rescue.

## Claim ceiling

`PASS_PRIVATE_RUNTIME_SBOM_READINESS` means only that the exact private Windows
runtime has deterministic dependency inventories. It is not legal clearance,
vulnerability analysis, signature, installer certification, calibrated stock
evidence, physical-film reproduction, or public release readiness.
