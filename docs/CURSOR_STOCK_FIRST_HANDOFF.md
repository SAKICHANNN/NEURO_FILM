# Cursor stock-first handoff

Date/time: 2026-07-16 (local Asia/Shanghai session)

Branch: `research/fivek-auto-optimize-cache`

HEAD at handoff write time: see final status block below after commit/push.

Remote: `origin` → `https://github.com/SAKICHANNN/NEURO_FILM.git`

## Completed nodes in this Cursor session

1. Pending SF0.1→SF0.2 dirty worktree reviewed, tested (124→126 tests), committed and pushed:
   - `8733764` `real-film: approve bounded stock pilot download`
2. `SF0.2` download executed and verified (byte-identical rerun):
   - `5990a04` `real-film: verify bounded stock pilot download`
3. `SF0.3` integrity/support audit implemented, executed twice byte-identically, documented:
   - `c952309` `real-film: add stock pilot integrity audit`
   - plus the follow-up results/propagation commit recorded below

## Active / ready

- Active parent: `ULT > RF stock-first real-film mainline`
- Completed through: `SF0.3`
- Ready leaf: `RF1.4` per-stock leave-one-roll-out identifiability
- Do not start RF2.S/RF3/GPU until a stock's RF1.4 passes

## Data directories

| Path | Files / bytes | Notes |
|---|---|---|
| `data/raw/blueneg_stock_pilots_v1` | 189 lane files / 227,287,697 bytes | new SF0.2 root; gitignored |
| `data/raw/blueneg` | 213 local files counted | old pilot untouched |
| FSA/OWI partial | 258 derivatives / 81,016,399 bytes | sealed auxiliary; untouched |
| `outputs/real_film/stock_pilots_v1/integrity_audit/` | report + 8 contact sheets | gitignored outputs |

## Important SHA-256 values

| Artifact | SHA-256 |
|---|---|
| acquisition manifest | `1ac2dfce6060d0094d221e8776f4318ebdd144a86194c534daca01ae3749324e` |
| metadata report | `9ba954cffd5f47839fcb14fe7b025bb8be27ca48256baf7b579dce4c21e756d3` |
| download report | `de286950dd78970281b592ffd1975f26db9b2a56565d92d84043bf3eafb348b3` |
| integrity report | `0e09c90106824425ceaa57dead73155229c35997ae933db55f88420c70e93f0a` |
| integrity software commit | `c952309bbee4da51f3849e155c49ddca2706dac3` |

BlueNeg revision remains `b038a1ae68f42067ff12b5e79ddbe62919b7af23`.
Required credit: `Copyrighted by Tien-Tsin Wong`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Observed in this session: `126 passed` after the integrity module landed.

## Vision adjudication

- Completed automatically: decode, size/hash, exact dup, dHash triage, support matrices, eligibility gates
- Pending Codex vision adjudication: 8 contact sheets under
  `outputs/real_film/stock_pilots_v1/integrity_audit/contact_sheets/`
- Border/interior masks: `not_implemented` (do not invent)

## Closed / negative hypotheses retained

- Physical-roll-only Roll2Film remains closed (BlueNeg nuisance retrieval result)
- FILM-R family learning remains structurally stopped
- Gold 400-5 remains rejected for first pilots after whole-test-roll sealing
- No stock has reached `S2`

## Unresolved risks

- GA100 and Konica have weak structural content diversity for RF1.4
- Only Gold has a meaningful display-proxy lane
- All 189 files lack ICC profiles
- Root `LICENSE` still absent (U0.2 blocked)
- No Cursor visual pass/fail was claimed

## Running processes

None intentionally left running. No download/train jobs active at handoff.

## Resume commands

```powershell
cd C:\Users\hhvrf\Documents\neuro_film
.\.venv\Scripts\python.exe scripts/download_real_film_stock_pilots.py --workers 8
.\.venv\Scripts\python.exe scripts/audit_real_film_stock_pilots.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Dirty / untracked policy

After the final handoff commit/push, the worktree should be clean except for
gitignored data/outputs. Do not delete ignored pilot roots.

## New user authorization needed for

- paid Xi Film / other licensed purchases
- 290GB BlueNeg archive
- paid GPU
- public release / root license decision
- film/lab/scanner capture

Necessary free bounded downloads already authorized and completed for the 189
objects.

## Recommended first three Codex steps

1. Vision-adjudicate the eight stock-pilot contact sheets; record border/base/caption notes without claiming stock signal.
2. Freeze the RF1.4 evaluator contract before any operator fitting: LOO-roll, pooled/wrong-stock/retrieval/historical/shuffled/simple controls, separate density vs display domains, claim ceiling.
3. Implement and run RF1.4 for Gold display and the three density candidates; stop/downgrade stocks that fail structural or nuisance gates rather than adding capacity.
