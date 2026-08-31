# U7.8F Windows-safe batch job ID contract

## Question

Can the existing U7.8A/U7.8B multi-input product transaction reject job IDs
that are distinct JSON strings but not distinct safe Windows/exFAT directory
components, before any input file is decoded, hashed or rendered and before any
workspace, stage or destination is created?

## Parent state and observed defect

U7.8A and U7.8B use `job_id` as a directory name. Their frozen v1 syntax
`^[a-z0-9][a-z0-9._-]{0,63}$` admits both `a` and `a.` and also admits Windows
device basenames such as `con`, `aux` and `com1`. On the target Windows
filesystem, creating `a.` after `a` addresses the existing `a` directory.
Semantic string uniqueness therefore does not prove filesystem identity.

U7.8F is the sole prospective correction. It does not rewrite the historical
U7.8A/U7.8B evidence and does not rerun the closed U7.8C/U7.8E real-scale
cohorts.

## Frozen validation

The existing ASCII syntax, length and semantic uniqueness rules remain. In
addition, every `job_id` must satisfy both rules below:

1. it must not end in a full stop (`.`);
2. the substring before its first full stop, compared case-insensitively, must
   not be one of `con`, `prn`, `aux`, `nul`, `com1` through `com9`, or `lpt1`
   through `lpt9`.

The basename rule intentionally rejects forms such as `con.txt`; Windows treats
the device basename as reserved even when an extension is present. The syntax
already excludes spaces, slashes, colons, non-ASCII characters, leading dots
and mixed case.

Validation occurs while parsing the complete manifest. Invalid IDs reject
before input-path resolution is used for filesystem reads, before input hashing
or image decode, and before any stage, resumable workspace or destination is
created. U7.8B inherits the exact shared U7.8A parser; it must not introduce a
second normalization policy.

## Formal fixture and controls

- one valid two-job manifest, enumerated forward and reverse, must produce the
  same canonical parsed jobs;
- every frozen trailing-dot and reserved-device control must reject through
  both U7.8A and U7.8B entry points;
- read/hash/render/software-commit hooks are forbidden for every invalid row;
- destination, stage and resumable workspace remain absent;
- valid dotted identifiers such as `scene.001` and existing production IDs
  remain accepted unchanged;
- one Windows filesystem witness must show the original `a`/`a.` alias fact
  without leaving residue;
- parent U7.8A/U7.8B/U7.8D focused regressions must remain green.

## Stop rules

- Do not normalize, trim or silently rewrite a caller's identifier.
- Do not create hash-derived replacement directory names in this leaf.
- Do not change manifest, receipt, recipe or checkpoint schemas.
- Do not alter pixels, colour arithmetic, stock parameters, output formats,
  transaction provenance or recovery semantics.
- Do not reopen U7.8C/U7.8E or expand into packaging, installer, source search,
  RAW/HDR/OpenEXR or adjacent transaction wrappers.
- Any formal gate failure closes U7.8F without weakening the rule or replacing
  the filesystem witness.

## Claim ceiling

Private Windows/Python path-component safety for existing deterministic
three-look product batch and resume mechanics. No calibrated film-stock
response, image-quality, product-value, public API, package, installer or
release claim.
