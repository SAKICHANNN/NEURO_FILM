# SF0.8B YFCC Velvia 50 pixel results

Date: 2026-07-16  
Decision: **pass as the third provisional unpaired `S0` pixel stock; learning remains closed pending SF1 identifiability**.

## Acquisition and rights

The frozen YFCC15M exact-text pool contained 51 CC-BY-2.0 Velvia 50 rows across 26 Flickr UIDs. Prospective cross-process, E-6-as-C-41, HDR/tonemap, multiple-exposure and B&W-developer exclusions removed six rows. A stable UID round robin with at most four rows per UID made 41 candidates reachable.

Each retained file required a currently reachable Flickr page containing the CC BY 2.0 licence URL, a live HTTPS static-image response, at least 512 pixels on the short side and a successful decode. The pilot retained 25 files / 4,575,435 bytes across 14 UIDs; the largest UID share is 16%. Thirteen candidates failed the live licence-page check and three failed minimum dimension. They were rejected even though the old YFCC snapshot contained a permissive licence.

Download manifest SHA-256: `6a7d84978bfb70433c03558a5abe0959b71959dc5ff4d12ca34d7f0874ca2011`.

## Integrity and visual review

The audit was run twice and produced the identical SHA-256 `21059438f6fb335173e912d8cf1673d5bd82a3f345a87a6b5f62502075054ec5`. There are zero exact duplicate groups and zero dHash pairs at Hamming distance four or below.

Two contact sheets and the two highest clipping-risk full-resolution files were reviewed with Codex vision. No severe color block, banding, tearing, scan corruption, repeated texture, geometry damage or red-speckle/posterization failure was confirmed. The high-white cases are a real white matte/signature and a high-key wedding exposure with film border. Strong blue sky, green/red saturation, reversal-film contrast, real clipping, borders and grain are source characteristics rather than glitches.

The retained set covers landscape, architecture, people, a wedding, night scenes, plants, animals and underwater photography. It is materially more diverse than the failed Kodachrome bridge/stone clusters.

## Boundary and next gate

This is an unpaired, community-labelled and scanner/process-bearing source. It establishes only a third provisional `S0` pixel pool alongside Commons Ektar100 and UltraMax400. It does not establish Velvia response, authenticity, calibration, transfer to digital images or learnable stock signal.

SF1.0 must freeze a three-stock support matrix and test stock prediction against source, content, grayscale, low-frequency color, saturation/contrast and shuffled-label controls under author-group holdout. If the correct stock signal does not survive those controls, fitting a colour operator remains forbidden.
