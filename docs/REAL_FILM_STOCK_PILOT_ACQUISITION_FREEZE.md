# Real-film stock pilot acquisition freeze

Date: 2026-07-15

Node: `ULT > RF0.4 > SF0.1`

Decision: exact bounded download approved by the active autonomous data scope

## Frozen evidence

At software commit `70fe95bb129ffb278391fd385db6cf5ed12946fd`, two
complete executions produced byte-identical evidence:

- source: `ttgroup/blueneg-release` at
  `b038a1ae68f42067ff12b5e79ddbe62919b7af23`;
- registry/metadata cross-check: 13 film strings, 53 rolls and 491 frames, zero
  mismatches;
- acquisition: 189 exact preview/proxy objects / 227,287,697 bytes;
- manifest SHA-256:
  `1ac2dfce6060d0094d221e8776f4318ebdd144a86194c534daca01ae3749324e`;
- report SHA-256:
  `9ba954cffd5f47839fcb14fe7b025bb8be27ca48256baf7b579dce4c21e756d3`;
- 17 official-test rolls are sealed globally and no selected acquisition row
  belongs to one of them;
- every row has an exact remote path, size and LFS SHA-256.

All four physical inputs are frozen in the report: registry, frames, rolls and
remote inventory. The rerun is byte-identical because it records the committed
software identity rather than a dirty worktree identity.

## Pilot result before pixel download

| Stock ID | Unsealed rolls | Preview frames | Public display proxies / rolls | Pre-download decision |
|---|---:|---:|---:|---|
| `kodak_gold_100_gen5` | 7 | 50 | 47 / 6 | only current display-operator candidate; prior physical-roll result remains nuisance-negative |
| `konica_super_xg_100` | 7 | 22 | 1 / 1 | density/metadata identifiability only |
| `fujifilm_nph_400` | 4 | 53 | 0 / 0 | density/metadata identifiability only |
| `kodak_ga_100_5095` | 3 | 16 | 0 / 0 | minimum density/metadata identifiability pilot |

`kodak_gold_400_gen5` was rejected before download: sealing its two
official-test rolls leaves only one roll and one frame. Its nominal 3-roll / 38
frame count was therefore scientifically misleading for this split.

## Download contract

`configs/real_film_stock_pilot_acquisition_decision.json` authorises only the
frozen 189 objects into the isolated ignored root
`data/raw/blueneg_stock_pilots_v1`. The existing 101-file BlueNeg pilot remains
untouched. The downloader rejects path traversal, wrong source/revision,
manifest/report hash drift, wrong counts/bytes, missing LFS hashes, corrupted
payloads and manifest-external lane files. It does not decode pixels and cannot
download the 16-bit archive.

Passing download verification permits only SF0 pixel integrity/content audits.
It does not establish stock signal, a display-colour expert, `S2` transfer,
calibration, authenticity or release clearance.
