# P233 — local R1CZ payload recovery capsule

Date: 2026-08-26

Status: frozen before create-only capsule materialization

## Question

P230 proves that this repository's private consumer can verify and apply the
exact 4,376-byte R1CZ payload, but formal execution still reads that artifact
from the producer checkout. Can the consumer repository retain the same bytes
in a strict local capsule and reproduce P230 without any producer checkout
read?

R1DB independently proves installed-wheel apply-only behavior in the producer
package. It does not persist a consumer-owned copy here, so P233 is an
artifact-recovery boundary rather than a new algorithm or quality experiment.

## Frozen capsule

The create-only capsule contains exactly:

- schema `kmcfm.r1cz-payload-capsule.v1`;
- encoding `hex-lower-v1`;
- the expected 4,376-byte payload SHA-256;
- the exact bundle ID;
- lowercase hexadecimal payload bytes.

Hex is used because the producer artifact is canonical JSON with no trailing
newline. The capsule's own JSON formatting is irrelevant after strict
canonical validation; decoding must reproduce the original payload bytes
exactly, without stripping or normalizing anything.

## Formal gates

- P230 evidence and local consumer source identities are exact;
- capsule keys, encoding, hex alphabet, decoded length/hash and bundle ID are
  exact;
- the decoded payload passes the unchanged P230 loader and envelope;
- the frozen 729-row stress fixture and output hashes equal P230 exactly;
- payload-hex tamper, wrong encoding and wrong bundle ID reject before output;
- forward/reverse fresh-process reports are byte exact;
- formal producer-checkout, paired-build, network and media reads are zero.

## Stop rule and claim ceiling

Any failed gate closes local materialization. Do not rebuild the paired bundle,
read the producer checkout during formal execution, normalize payload JSON,
change encoding, or alter consumer arithmetic.

A PASS is one private checkout-independent recovery artifact for one exact
self-authored paired/capture-time payload. It is not a public package/schema/
capability, natural HDR quality result, after-only inference, or product
admission.
