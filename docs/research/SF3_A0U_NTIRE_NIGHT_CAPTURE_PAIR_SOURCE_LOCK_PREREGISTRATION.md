# SF3.A0U NTIRE night capture-pair source-lock preregistration

Date: 2026-08-21

## Direction correction

SF3.A0T candidate 1 showed that capture white balance alone did not identify a
materially better explicit ISP than an equal-capacity camera-static baseline.
SF3.A0U therefore does not vary that model or reuse the RGB2RAW cohort. It asks
whether a materially stronger observation is actually available: synchronous
beam-splitter RAW/professional-camera pairs with explicit sensor, noise,
orientation and pixel-alignment metadata.

The official CVPRW 2025 report states that Huawei Mate 40 Pro RAW and Sony
ZV-E10 JPEG images were captured synchronously through a non-polarizing beam
splitter on static scenes. It also lists black/white level, noise profile, CFA,
orientation, white-balance correction and target-crop bounds. Zenodo record
15213321 declares open access under CC BY 4.0.

## Frozen source audit

Before any member payload or pixel read, parse only bounded HTTP-Range tails of
the exact 47,288,204,304-byte RAW ZIP and 379,600,345-byte Sony ZIP. Bind the
Zenodo record, archive sizes/checksums, ZIP64/central-directory identities,
safe canonical member names, scene/group structure, pairing graph and the
presence of explicit metadata/crop members. Total archive-range traffic is
limited to 16 MiB and complete archive download is forbidden.

Passing requires at least 64 complete independently grouped pairs with every
listed physical/alignment metadata class structurally present. Two complete
reports must be byte exact. Missing or ambiguous license, pairing, grouping or
metadata closes the source before member extraction. A pass opens only a
separately frozen small member/pixel registration preflight; it does not become
candidate 2 and the bounded counter remains `1/3`.

## Claim boundary

This audit cannot establish target quality, camera rendering preference,
operator reuse, film/stock identity, arbitrary night photography, package,
schema, capability or product readiness. The Sony image is a synchronized
professional-camera reference, not neutral scene truth or a calibrated stock
render.
