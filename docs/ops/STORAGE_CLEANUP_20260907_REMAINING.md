# Storage cleanup recovery point, 2026-09-07

Scope: project-owned neuro_film and zhuise storage only. This is not authority
to remove other P/D directories, models, scientific evidence or arbitrary old data.

## Completed

Commit e7b5a7239 records ten exact stale runtime/acquisition scratch roots removed.
The filewise run removed 63,272 files / 3,476,267,829 logical bytes; initial
partial removal is excluded. P free was approximately 89.19 GB afterwards.
Canonical data/output/checkpoint/loras junctions are unchanged.

## Verified transport residue, platform-blocked

- `P:\zhuise_storage\data\mvsec_r0zb`: retain the complete
  `mvsec_outdoor_day_1_20Hz.tar` (12,515,891,200 bytes). Live SHA256 matches
  `docs/research/R0ZB_OFFICIAL_MVSEC20_ERAFT_SOURCE_BYTE_LOCK.json` in zhuise:
  `53e0d7065d913ed7cd044aebdd4909cfd9127c5b4519610b70192358be8b49d5`.
  Candidates are exactly `mvsec20.seg00` through `seg07`, `cont00` through
  `cont07`, `tail00` through `tail07`, plus
  `mvsec_outdoor_day_1_20Hz.initial.partial`: 25 files / 12,932,058,112 bytes.
  These are transport residue, not the retained checkpoint `mvsec_20.tar` or
  selected samples/manifests. Directory is not a reparse point. No live compute
  reference found after the read-only hash worker finished. Native PowerShell
  removal was rejected before execution: zero deleted, no alternate-tool retry.
- P293 EyefulTower transport residues were separately blocked before execution
  in the prior batch. Preserve the completed EXR and source manifest. Its
  manifest records eight segments and one aborted prefix, 11,103,249 bytes.
  This turn did not reattempt their deletion.

## Retain pending classification

- zhuise unified-HFR `input_15fps` transport pieces: no complete-copy byte
  validation yet. Filenames or an adjacent tar alone are not deletion evidence.
- Organized legacy/fallback archives: no per-file duplicate proof this run.
- D neuro_film fallback has research source and scratch; D zhuise fallback
  retains r1ed cargo/rustup toolchains. They are not automatically garbage.
- Remaining formal checkouts, wheelhouses and SDK scratch require dependency
  classification. Retired research is not by itself deletion authorization.

The whole-directory request is partially completed, not certified all-clean.
Blocked removal requires a permitted execution path or manual action, not a
shell/API workaround. No disk repair, external communication, training or new
download was performed. Existing interrupted creative and parallel work remains.
