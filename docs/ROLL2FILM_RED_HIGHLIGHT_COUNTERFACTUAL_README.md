# ID 11 Red-Highlight Counterfactual

Files in the packaged directory:

- `00_source.jpg`: neutral source image;
- `01_anchor09.png`: conservative source-gamut/margin-4 anchor;
- `02_anchor56.png`: stronger source-gamut anchor with the known red-metal highlight speckle;
- `03_anchor56_chroma_margin4.png`: chroma-gamut/margin-4 challenger.

Observed diagnostic facts:

- anchor56: 4,517 candidate pixels (0.3137%), 464 small islands, largest 335 pixels;
- challenger: 3,206 candidate pixels (0.2226%), 608 small islands, largest 178 pixels;
- challenger output range `[4,251]`, zero new hard clipping in the controlled check;
- mean chroma fell from about 30.93 to 21.46.

The coverage reduction and non-monotonic island count are diagnostic only. The
researcher must inspect the full-resolution files and must not treat these
numbers as a pass, preference score, or proof that the challenger preserves
enough style.
