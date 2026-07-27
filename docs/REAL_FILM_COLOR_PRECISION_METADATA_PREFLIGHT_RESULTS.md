# SF2.10R Color Precision Metadata Preflight Results

Date: 2026-07-27

Decision: **metadata topology promising; rights, replication and verified
pairing DoR fail before pixels**

## Reproducibility and access boundary

- software commit:
  `c0021ec9c7934560a6ca730f7c4d55bcf1ea93f8`;
- config SHA-256:
  `9af77d24b9f3223d35b2ba68585542606ced4ba79675c7248dbb96a13e756bc1`;
- both formal reports are byte-identical at
  `2bac9009f5741398c2c5d0a23f42724a4bac82407e1ec5741bb6727c41d03529`;
- both stderr logs are empty;
- eight previously retained HTML responses were read;
- zero embedded image URL requests and zero image payload bytes.

The comparison and terms responses repeat byte-for-byte. Product and scanner
pages vary only in the observed per-request `shopify-y` UUID; after removing
only that token, all four page pairs repeat exactly.

## Public metadata topology

The comparison HTML embeds 927 unique S3 image URLs:

| Dimension | Count |
|---|---:|
| stock-looking normalized labels | 20 |
| collection / scene / stock metadata cells | 95 |
| filename-implied condition keys | 471 |
| complete Frontier/Noritsu filename pairs | 456 |
| incomplete or typo-separated pair candidates | 15 |
| Frontier / Noritsu URLs | 464 / 463 |
| PNG / JPEG URLs | 502 / 425 |
| filenames containing `pushed` | 124 |
| filenames containing `cross-processed` | 34 |
| filenames containing `tungsten` | 20 |

The 20 normalized labels cover nine Kodak-looking and eleven Fuji-looking
families. This is unusually rich potential scanner-nuisance topology.

Every item above is filename-derived. It does not verify the physical stock,
exposure, illuminant, process, push, roll or scanner settings. The 15 unmatched
keys include spelling and condition typos, which is why the audit does not
silently fuzzy-match them.

## Confounders and pairing ceiling

The comparison page states that:

- the assets are 6K scans of 35 mm film;
- minor exposure adjustments were applied after scanning;
- scanners automatically adjust colour, white balance and contrast per frame.

The scanner article says selected Frontier/Noritsu examples came from the same
negative. This supports the source concept, but filenames alone do not verify
pixel registration or prove that all 456 candidates share one negative.

A separate commercial product page says its emulations were built by matching
digital and film captures under the same lens and lighting. The comparison
page exposes no digital counterpart manifest. The commercial statement cannot
be reassigned to these 927 comparison assets.

Physical roll ID, process session and scanner settings/profile remain unknown,
and independent roll/process replication is not established.

## Rights and project decision

The terms page was updated July 15, 2026. It grants a personal,
non-transferable licence for the app and purchased products and forbids
redistributing, reverse-engineering or extracting purchased emulation
profiles. It does not publish a research/ML reuse grant for comparison-image
pixels, and the comparison footer says all rights reserved.

The recorded status is
`no_public_reuse_grant_found/rights_unknown`. This is a conservative evidence
gap, not legal advice and not an assertion that permission could never be
obtained.

The branch is
`metadata_topology_promising_rights_and_replication_blocked`. Preserve the
source lead. Do not request images, purchased products, profiles, LUTs or the
advertised 9.4 GB PDF bundle. Reopen only with explicit pixel permission or a
licensed export, credible independent roll/process support, a verified pairing
manifest and a newly frozen bounded acquisition contract.

No pixel study, scanner operator, stock operator, digital-to-film operator,
training, LSM or production integration opens.
