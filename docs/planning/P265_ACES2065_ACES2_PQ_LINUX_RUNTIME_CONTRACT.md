# P265 — ACES2065 / ACES 2 PQ Linux runtime contract

Status: dependencies source-locked; frozen for committed-head formal execution.

## Question

Can WSL2 Ubuntu 22.04 x86_64 with exact CPython 3.12 wheels execute the
unchanged strict P251/P252 ACES2065-1/AP0 to official ACES 2 canonical
Rec.2100-PQ RGB16 PNG path and reproduce the complete Windows P252 bytes?

## Frozen inputs and runtime

- exact P249 synthetic AP0/D60 OpenEXR: 1,144 bytes, SHA-256
  `1a25181406270142c98f2169897d3a4c468ec50c979fff9077d37f956c888726`;
- exact Windows P252 output: 469 bytes, SHA-256
  `533cad62273a55a5863782021d39b477b82ea9230a8578462f6996caa259a2c4`;
- exact RGB16 sample SHA-256
  `5194e283ea14c90cf9f79e09041d684db1e2fe53f6276f8d5cf08d453dc7aa6d`;
- WSL2 Ubuntu 22.04 x86_64, `/usr/bin/python3.12`;
- exact producer-retained NumPy 2.4.4 and OpenEXR 3.4.15 Linux wheels;
- official PyPI PyOpenColorIO 2.5.2 CPython 3.12 manylinux 2.27/2.28 x86_64
  wheel, expected SHA-256
  `3d0ab1c398d560513c651c7b9c676e68f264a63b97e29fec21722d4270f865aa`.

The Linux child imports the exact consumer source modules from the committed
checkout while bypassing the broad package initializer; it does not copy or
rewrite the renderer. All wheel contents and output media live under the
repo-relative P-backed project tree and are hash-bound.

The PNG module's import-only SDR ICC dependency is replaced inside the isolated
child by a sentinel that raises if called. The frozen Rec.2100-PQ route must
leave its call count at zero; this avoids adding unexercised Pillow/tifffile
dependencies while making accidental SDR-branch execution a formal failure.

## Frozen gates

1. All parent source/evidence/config identities and three wheels are exact.
2. Linux runtime, OpenEXR and OpenColorIO versions and OCIO config cache ID are
   exact.
3. Strict AP0/D60 metadata, decoded AP1 pixels and invalid controls match P252.
4. Linux RGB16 samples and complete canonical PNG are byte-exact with Windows.
5. Forward/reverse Linux partition schedules and two outer processes are exact.
6. Source is immutable; publication is create-only and failure-atomic.
7. Network reads during formal execution and owned temporary residue are zero.

Any source, wheel, runtime, metadata, pixel, sample, file, atomicity, replay or
cleanup mismatch closes P265. No wheel/version/config/source/output/tolerance
substitution or rescue is allowed.

## Claim ceiling

Private WSL2 Ubuntu 22.04 x86_64 target-runtime parity on one synthetic P249
fixture only. No physical Linux, arbitrary EXR/HDR, real-image quality,
public dependency/API/package/schema/capability, product admission, film-stock
evidence, single-reference matching or candidate 3.
