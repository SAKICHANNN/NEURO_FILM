# U5.R2AB0 — CAVE DoRF named-film response source audit

## Result

**Limited internal-research source pass.** The official Columbia CAVE DoRF
archive is small, reproducible and structurally useful for one synthetic
explicit-response experiment. It is not a complete film colour operator and
does not open real-image rendering, fitting, training or product integration.

The retained archive is 1,734,345 bytes at SHA-256
`d030d1ee3cd3243b412dd817b7479873f3680aaea33baaa9c127752880eacd63`.
Its sole member, `dorfCurves.txt`, is 6,595,757 bytes at SHA-256
`6dfe71f44c84c3fce9dafa96ce2ebd08854580aeaa3b695cf4ce3ea904ddbf3d`.
The archive is ignored data and is restored with:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py dorf
```

## Observed inventory

- 201 six-line response records;
- 1,024 normalized irradiance and brightness samples per record;
- every irradiance sequence is strictly increasing;
- every brightness sequence is finite, nondecreasing and exactly spans 0–1;
- 104 `graph-log-log-neg`, 50 `graph-log-log-pos`, 26 `mlab-lin-lin`,
  13 `gamma-lin-lin`, five `graph-lin-lin` and three
  `mlab-lin-lin-sparse` records;
- 46 strict source-name RGB triplets using exact `Red/Green/Blue` suffixes;
- three duplicated source names (`FP2900ZR/ZG/ZB`, twice each);
- several exact curve duplicates, including channels inside named triplets;
- source spelling leaves `Kodachrome-25` Red/Green and `Kodachrome-25CD` Blue
  as separate incomplete groups. The audit does not silently repair this.

The [official DoRF page](https://www.cs.columbia.edu/CAVE/software/softlib/dorf.php)
describes 201 curves with 1,000 points, while the exact downloaded member
contains 1,024 samples per curve. Runtime evidence is bound to 1,024 and the
contradiction is retained. The associated
[Grossberg–Nayar PAMI paper](https://cave.cs.columbia.edu/old/publications/pdfs/Grossberg_PAMI04.pdf)
describes the response-space model and database construction.

## Rights and claim boundary

No explicit licence was located on the official page or inside the archive.
The source is restricted to internal research audit:

- do not redistribute the archive or extracted curves;
- do not use it for commercial/product integration;
- do not infer emulsion generation, process or scanner state from a name;
- do not call normalized per-channel curves digital/film pairs;
- do not claim spectral sensitivity, cross-talk, absolute exposure,
  development, print, scan or viewing calibration.

The highest valid claim is **historical normalized named-film
channel-response prior**.

## Branch decision

AB0 opens exactly one next leaf: preregister a synthetic explicit per-channel
response diversity and safety pilot. It may test whether strict RGB triplets
retain distinguishable non-basic behaviour relative to shared/gamma controls
and whether neutral-axis/range/Jacobian properties are acceptable. It must
preserve a no-result branch and may not inspect real images before automatic
gates.

Current-pixel fitting, model training, LSM, stock authenticity, calibrated
response and production integration remain closed.
