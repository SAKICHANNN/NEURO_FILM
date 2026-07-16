# SF1.3A YFCC shared-author bounded pixel contract

Date: 2026-07-16

Node: `ULT > RF0.4 > SF1.3A`

Status: frozen before image access

## Question

Can the eight SF1.2-passing UIDs yield a small, rights-current, decodable and
group-connected Ektar100/Velvia50 pixel pilot without duplicate, source or
content failures that immediately close the design?

This is an acquisition/integrity question only. It does not fit an operator or
test latent modes.

## Scope

- use only the eight exact SF1.2-passing UIDs;
- use the prospective candidate order already frozen by SF1.2;
- at most four rows per UID and stock, 38 candidates total;
- reverify the live CC BY 2.0 page before every image request;
- prefer Flickr's bounded large derivative and never request an original;
- at most 32 MiB/file, 512 MiB total, sequential requests with 1.5s spacing;
- require a decoded short side of at least 512 pixels;
- preserve failed attempts and exact source/derivative URLs, licence timestamp,
  bytes, SHA-256, dimensions and lineage.

## Gates

The pilot passes acquisition only if:

1. each stock retains at least eight files;
2. at least five UIDs retain at least one valid pixel for each stock;
3. every retained payload passes hash, decode, dimension and exact manifest
   checks;
4. exact and dHash<=4 duplicate/sibling groups are reported before any split;
5. contact sheets and full-resolution risk cases receive autonomous severe-
   artifact review;
6. content/source support is reported by UID and stock without imputing
   exposure, process, scanner or physical roll.

Failure closes or narrows the shared-author pixel design; thresholds are not
weakened and no substitute author is added after access.

## DoD and next branch

Produce a hash-bound download manifest, integrity/source/content report,
contact sheets, visual verdict, config/software identities, full CPU tests and
an explicit decision. A pass may open only a newly preregistered shared-author
stock-identifiability diagnostic. Training, operator fitting and LSM remain
false until that later diagnostic independently passes nuisance controls.
