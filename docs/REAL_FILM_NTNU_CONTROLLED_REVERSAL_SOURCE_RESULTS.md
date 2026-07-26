# SF2.8R NTNU controlled-reversal source reconnaissance

Date: 2026-07-26

Node: `ULT > RF0.4 > SF2.8R`

Decision: **promising controlled design; public data unavailable**

Claim ceiling: publication/thesis method evidence only

## Question

Does the newly published NTNU controlled reversal-film study expose a
rights-clear, lineaged corpus that can open a stock-specific explicit-operator
fitting leaf?

This was bounded, read-only source reconnaissance. It did not fit an operator,
train a model, infer a data licence, contact an author or treat publication
figures as training pixels.

## Authoritative sources

- Balica, Ciortan and Trumpy, *Colour correction of archival photographs for
  change detection and digital restoration of paintings*,
  <https://doi.org/10.1016/j.daach.2026.e00550>, CC BY 4.0;
- Balica, *The Use of Analogue Photographic Documentation to Infer the Past
  Appearance of Paintings*, NTNU master's thesis,
  <https://hdl.handle.net/11250/3221327>;
- exact NVA metadata record
  `0199c398b026-a2da028d-a09e-4961-9651-6eb58addea0f`.

The article's licence applies to the publication. The two NTNU repository files
carry NVA's general Copyright Act terms; this audit does not promote either
surface into a dataset or commercial-training licence.

## What the experiment really contains

The thesis documents a materially stronger capture design than current
community-photo pools:

- fresh Kodak Ektachrome E100 and Fujifilm Fujichrome Velvia 50, both
  120-format reversal film;
- one Rolleiflex twin-lens reflex camera at f/8;
- two mock-up paintings, halogen and LED illumination, and metered plus
  higher/lower shutter-speed captures;
- a Calibrite Passport ColorChecker in every scene;
- HySpex VNIR-1800 hyperspectral captures under matched illumination geometry;
- a ten-band custom multispectral transmission scan of the developed film;
- pre/post accelerated-ageing hyperspectral references.

This is controlled film-to-reference evidence, not a digital-camera/film pair.
Inverting its correction mapping would still be a condition-specific
film-inspired simulation hypothesis.

## Availability audit

The exact public NVA record contains only two files:

| File | Bytes | SHA-256 | Contents |
|---|---:|---|---|
| thesis PDF | 472,364,357 | `e231da44fc8163559d8ae6d9b4f41cf825b4106a0316aeeb078efecdb43e8a68` | 109-page thesis |
| submission ZIP | 2,225,944 | `9b150ca59274f8d0c9a01492f1d34347282d4b065ff5ea7500cc1b60aef970b7` | draft cover PDF plus AI declaration |

No raw hyperspectral/MSI cube, TIFF, patch measurement table, alignment
manifest, source code or dataset licence is exposed by the official article,
thesis record or attachment archive.

## Identifiability limits

The acquisition campaign reportedly contains many frames, but the thesis
selects six for detailed analysis:

- Yoda: Velvia/halogen normal and overexposed; Ektachrome/LED normal;
- Cactus: Ektachrome/halogen normal; Velvia/LED normal and underexposed.

The thesis explicitly says the analysed photographs do not contain a
same-illumination cross-stock comparison. Its displayed cross-stock comparison
therefore mixes film stock with halogen versus LED. Independent roll count and
process session are not reported as structured evidence. Stock effect is not
identified by the published subset, even before the missing-data problem.

The study's own results also reinforce the Ultimate simplicity and artifact
rules: regularised root-polynomial correction outperformed the tested shallow
and deeper MLPs, while performance dropped for strongly under/overexposed
slides. That is method precedent, not proof that the inverse polynomial is safe
or stylistically sufficient.

## Binding branch

Decision code:
`controlled_source_promising_but_public_data_unavailable`.

- Do not extract publication figures as fitting data.
- Do not infer reusable pixel rights from article CC BY or repository access.
- Do not fit, train, cluster latent modes or claim a stock response.
- Preserve this design as a priority acquisition lead and as evidence that
  exposure, illuminant and scan-chain controls must remain explicit.
- Reopen only if an official release supplies immutable raw film MSI data,
  hyperspectral references, condition metadata, alignment lineage and explicit
  reusable rights.

The next legal work remains read-only source discovery or data-independent
explicit-operator research. This close does not stop the Ultimate Goal.
