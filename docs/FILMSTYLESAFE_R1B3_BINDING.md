# U5.R1B3 A0 inventory binding

Date: 2026-07-17

Node: `ULT > U5.R1 > U5.R1B3`

Decision: **five A0 members bound to verified local artifacts; two cards remain unbound**

## Bound members

| member_id | parent | exact artifact |
|---|---|---|
| `a0-source-u41-01` | `u41-01` | U4.1 gold input 01 |
| `a0-strength-53/55/56` | `u41-01` | U4.2 normalized anchors on sample 01 |
| `a0-id11-regression` | `u41-11` | U4.2 anchor56@11 (siblings 09/53/55@11 recorded) |

`exact_hash` values were re-hashed from local files and match U4.1/U4.2 manifests.
`perceptual_hash` is dHash64 of the **parent input**, left-padded to 64 hex so
53/55/56 share the parent leakage key without inventing a new hash family.

## Still unbound

- `a0-synth-speckle-proto-001` — card-only synthetic operator
- `a0-hardneg-halation` — legitimate-local hard-negative placeholder

## Verification

```text
.\.venv\Scripts\python.exe scripts\audit_filmstylesafe_r1b3_inventory.py
.\.venv\Scripts\python.exe -m pytest -q
```

Result: 5 bound / 2 unbound; leakage pass; 387 CPU tests pass.

## Claim ceiling

This leaf binds existing A0-only evidence into the FilmStyleSafe membership
schema. It does not generate pixels, open hidden splits, recruit humans, train
SCIS, or claim population preference / severe-risk guarantees. Visual
adjudication of ID11 remains historical autonomous evidence already recorded
elsewhere; this leaf only freezes file identity.

## Next

`U5.R1B4`: implement the first explicit synthetic failure operator under the
frozen card, or bind RF2.C0 Ektar/fixed-e0 as an additional A0 control member.
Parallel U1 colour-state product leaves remain legal.
