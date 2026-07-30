# U5.R2BD0 Color Precision Chart Source Results

**Decision:** retain as an internal scanner-nuisance and source-connectivity
dataset; keep exposure-resolved analysis, fitting, training and redistribution
closed.

The publicly offered [Charts download](https://www.colorprecision.com/pages/film-comparison-tool)
was frozen before payload access at a 2.5 GB cap. The exact
`Charts-PDFs.zip` is 2,410,038,342 bytes with SHA-256
`6f1ccdb2...cd91a`. ZIP paths, size limits and CRC pass. Its two primary PDFs
are bound independently:

- Frontier: 1,422,900,860 bytes, `e0752ad0...10d8`;
- Noritsu: 986,816,453 bytes, `0bfdf2b5...e7ef`.

Text and visual inspection establish one controlled chart/tabletop setup with
24 pages, nine named Kodak stocks, 12 stock/process variants, two scanner
interpretations and 168 advertised EV labels. Every variant appears under both
scanner views. This is much better connectivity than community scene pools,
but it is still authored scan output, not digital/film pairing or stock
response truth.

## Exposure identity stop

The PDF object inventory contains 192 full-photo placements but 191 unique
objects. Every page advertises seven EV values, yet only one of 24 pages has an
equal unique-object count; one page reuses an image object. The extra or missing
exposure cannot be inferred from brightness. Consequently:

- page-level stock/process/scanner nuisance analysis may continue internally;
- individual embedded images must use unassigned position/object identities;
- exposure-resolved fitting and supervision are forbidden.

## Rights and claims

The site footer is All Rights Reserved, and the current
[terms](https://www.colorprecision.com/pages/terms) do not grant training,
redistribution or profile-extraction rights for the comparison archive.
Therefore this source cannot train a model, fit a product operator, enter LSM,
or support released/commercial weights. U5.R2BD1 may only ask whether
page-level stock/process variation survives paired scanner nuisance controls.

