# SF2.10R Color Precision Metadata-Only Preflight

Date: 2026-07-27

Status: **implementation and evidence contract frozen; formal repeat pending**

## Parent and purpose

`SF2.10R` is a stock-first data-feasibility child after current community
pixels failed source/content identifiability and direct IT8 measurements lacked
a common uncalibrated recorder input. Color Precision publicly exposes a film
comparison page with many scanner-labelled asset URLs. Its scanner article
also states that selected examples compare one negative on Frontier and
Noritsu scanners.

The useful research question is narrow:

> Does the public metadata expose a sufficiently rich nuisance-control
> topology to justify a future, separately authorized pixel preflight?

This leaf does not ask whether the images look like a stock and does not
download them.

## Frozen local evidence

Eight HTML responses were retained before this contract:

- comparison page and exact repeat;
- product page and repeat;
- scanner article and repeat;
- terms page and exact repeat.

Every exact byte count and raw/normalized SHA-256 is frozen in
`configs/sf2_10r_color_precision_metadata_v1.json`. Normalization removes only
the observed per-request `shopify-y` UUID. It does not normalize visible
content, image URLs, stock strings or legal text.

The formal runner is offline. It is forbidden to request any embedded image
URL or other network resource.

## Evidence semantics

All names parsed from image URLs are weak source metadata:

- a stock-looking filename is not verified `film_stock_id`;
- `pushed`, exposure and `tungsten` strings are not physical truth;
- Frontier/Noritsu suffixes create filename-implied pair candidates, not
  pixel-verified same-negative registration;
- physical roll, process session and scanner settings/profile remain unknown.

The separate product page says its commercial emulations use matched
digital/film captures under the same lens and light. The comparison page does
not expose a common digital counterpart manifest, so that statement cannot
turn these comparison images into reusable digital/film pairs.

## Gates

A topology can be called promising only if the frozen parser reproduces:

- at least 100 filename-implied Frontier/Noritsu pairs;
- at least ten stock-looking labels;
- repeat-stable normalized HTML;
- explicit disclosure of scanner auto-adjustment and minor exposure edits.

A pixel preflight additionally requires all of:

- an explicit research/reuse grant covering comparison image pixels;
- independent roll/process replication or a defensible grouping substitute;
- a manifest proving any common digital/film inputs claimed for the study;
- a new bounded acquisition contract.

The current terms grant personal use for the app and purchased products and
forbid extracting/reverse-engineering those products. They do not state a
public research/ML reuse grant for the comparison images, whose footer says
all rights reserved. This is recorded as
`no_public_reuse_grant_found/rights_unknown`, not as an invented legal ruling.

## Branches

- `metadata_topology_promising_rights_and_replication_blocked`: preserve the
  topology as a future lead, but stop before pixels.
- `source_contract_or_topology_not_reproduced`: preserve the discrepancy and
  close this source.
- `future_separately_preregistered_pixel_preflight_if_all_DoR_pass`: may open
  only after rights, replication and pairing evidence are independently
  supplied.

No result opens image access, fitting, training, stock claims, LSM or
production integration.
