# RF0.3 FSA/OWI 64-image pilot results

**Decision:** `pilot_eligible_with_grouping_and_border_constraints`.
The source survives decoding and first visual review, but Phase C and training
remain closed until shooting-assignment leakage and border handling are frozen.

## Automated evidence

The deterministic creator-balanced sample used ten groups (nine curated
photographers plus unknown), with one to eight images per group. All 64 files
downloaded as RGB JPEG, decoded successfully and stayed within the 64 MiB cap:

- 20,570,056 bytes total;
- zero identical payloads;
- zero dHash pairs at Hamming distance four or below;
- four files with embedded ICC data and sixty without;
- two complete executions produced identical manifest and report bytes.

| Artifact | SHA-256 |
|---|---|
| pilot manifest | `d2da35974a9af509f0969424396da6892b42ada059feaf2610fd37f02c05fc98` |
| pilot report | `3eac975d4012d3d13cd4953a64930b2ef69e2ccd2441763146e25a70018aaa15` |
| contact sheet 1 | `3503abdf89f8bd8da19f563752a7679e0a4e16e7c16ab1becb0f17923ad1501b` |
| contact sheet 2 | `410bac7f9813168b55cf7dc08b30e5214682e610111eb5ceb9aca36f4baab648` |

## Visual adjudication

All 64 images were reviewed together with LOC identifiers and creator groups.
No severe decode glitch, broken geometry, obvious modern colourization or
caption/image mismatch was confirmed. The set is visibly real archival film:
it has strong and heterogeneous colour character rather than a uniform
saturation boost.

That heterogeneity is also the main scientific risk:

- indices 9, 17, 21, 35, 41 and 50 show conspicuous colour-cast, ageing or scan
  variation;
- slide/mount borders are common and sometimes themselves strongly coloured;
- related shooting sequences are obvious at least at 6/15/23/31/39, 7/40 and
  47/55;
- shops, farms, factories, landscapes and portraits are diverse overall, but
  creator, location and assignment remain entangled.

Black or coloured borders would corrupt global colour statistics and make a
model learn the archive mount rather than the photographed scene. All later
feature extraction therefore needs a deterministic interior/border mask, with
masked and unmasked ablations reported. Adjacent shooting sequences must stay
inside one split group even when their titles differ.

## Scientific meaning

This pilot is stronger evidence than FILM-R for content diversity and source
provenance, and unlike FilmSet it consists of real scanned film. It still does
not provide a neutral digital counterpart, physical-roll identifiers, process
sessions or scanner settings. It can test whether archive-derived global,
hierarchical or retrieval experts generalize across creators/assignments; it
cannot by itself identify pure Kodachrome or support `calibrated-reference`.

## Next ready leaf

Recover assignment/LOT information where available; otherwise create a
conservative group from creator, date/location text and adjacent LOC scan IDs.
Freeze creator-out and assignment-out splits, then audit whether the 558-record
canonical metadata has enough independent groups. Phase C pixel expansion is
allowed only after that gate passes.
