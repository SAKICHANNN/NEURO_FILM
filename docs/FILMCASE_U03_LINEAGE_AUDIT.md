# FilmCase U0.3 Lineage Audit

**Status:** complete as a fail-closed audit; current reference-derived FilmCase lane is blocked.

## Scope and containment

This audit introduced the isolated `src/filmcase/` package and
`scripts/audit_filmcase_manifest.py`.  It reads the legacy ignored manifest and
local caption sidecars, writes only new ignored artifacts below
`outputs/filmcase/u03_lineage_audit/`, and never changes the legacy manifest,
renderer, existing training code, or source images.

The audit output is manifest schema v2.  Every row carries a lineage class,
allowed-use state, source-group state, FilmCase split, and eligibility result.
Uncertain rows fail closed to `quarantine`; no missing source, uploader, roll,
scanner, or license record is inferred from a filename or caption.

## 2026-07-11 local evidence

Command:

```powershell
.\.venv\Scripts\python.exe scripts\audit_filmcase_manifest.py --write
```

| Finding | Result |
|---|---:|
| Legacy manifest rows seen/audited | 4,210 / 4,210 |
| Caption sidecars matched | 2,320 |
| `caption_only` lineage | 2,320 |
| `unresolved` lineage | 1,890 |
| `resolved_group` lineage | 0 |
| FilmCase-eligible rows | 0 |
| Quarantined rows | 4,210 |
| Exact duplicate groups from legacy SHA-256 | 0 |
| dHash near-duplicate pairs at distance <= 4 | 140 |
| Cross-FilmCase-split duplicates | 0, because every row is quarantined |

The zero cross-split count is not evidence that the historical random split was
safe: there is no eligible group split to test.  The audit found no usable
source/uploader/roll/scanner grouping and retained the legacy
`unknown-flickr-user-content` licensing boundary.  Therefore the only allowed
FilmCase lane is `anchor_only`; reference-derived identifiability, case-memory
construction, router supervision, and training remain blocked.

One unusually large raster crossed Pillow's decompression-bomb warning limit.
The final implementation records such a row as `perceptual_hash_failed` and
excludes it from perceptual duplicate evidence rather than silently overriding
Pillow's safety guard.

## Recovery gate

Reference-derived FilmCase work can reopen only after source records provide,
per row, a durable source URL or ID, uploader or scan/roll group, license
snapshot with allowed-use, and content/perceptual hash provenance.  A new
group-aware audit must then show zero cross-split exact and near-duplicate
conflicts before any identifiability or routing experiment.

This is an evidence and rights gate, not a request for the owner to supply
data.  It is safe to continue the independent frozen-anchor and deterministic
renderer work while the reference lane remains closed.
