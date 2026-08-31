# SF3.A3X autonomous real-film data authorization request packet

Date: 2026-08-31

Node: `ULT > SF3 > SF3.A3X`

Status: ready for user-authorized external outreach; no message sent

## 1. Purpose and claim boundary

The user cannot supply physical film photographs. Public-source discovery has
found several controlled physical-film observations whose public records are
scientifically promising but whose data payload, grouping or reuse rights are
not sufficient for fitting. This packet converts the four strongest closed
source routes into precise authorization requests.

An affirmative reply is not scientific evidence and does not itself authorize
pixel use. Before any download or fit, the project must separately freeze the
exact offered inventory, sizes and hashes; stock, roll, process, scanner,
source and scene identities; development/validation/sealed-confirmation roles;
allowed fitting and derived-output uses; and leakage/severe-artifact gates.
Silence, refusal, a LUT-only offer, publication figures or permission to view
without fitting rights all leave the corresponding route closed.

No email, form submission, purchase, account creation or file request is
authorized by this document. Sending any request requires the user's explicit
external-message approval.

## 2. Ranked request queue

| Priority | Holder and official contact | Target contribution | Minimum useful export | Current closed evidence |
|---:|---|---|---|---|
| 1 | FilmMatch / Gianmarco Della Calce; `filmmatch.info@gmail.com`; [official contact](https://www.film-match.com/contact) | Portra 400 controlled paired colour | Original digital chart captures plus linear film scans for the described five rolls, three illuminants, -5..+5 EV cells and later validation roll; exact roll/session/scanner map and manifest | `SF3.A3R`: method is strong, but no Portra observation payload, fitting rights, exact manifest or group lock is public |
| 2 | PARVEC / Pablo Maraver Cardenas; `parvec.film@gmail.com`; [official project](https://www.parvec.es/35-mm-color-science-project) | Portra 400 and Ektar 100 under one controlled workflow | Unedited digital references, film scans or patch measurements for both stocks; per-stock roll/process/scanner/session identities; at least one held group per stock; exact manifest | `SF3.A3N` and `R1HE`: controlled measurements are described, but raw numerical data, documentation, group identities and fitting rights are absent |
| 3 | FILM2PAINT data holders: Irina-Mihaela Ciortan (`irina-mihaela.ciortan@ntnu.no`), Giorgio Trumpy (`giorgio.trumpy@ntnu.no`), and Zentralbibliothek Zurich/ZB-Lab (`zb@zb.uzh.ch`); [official paper](https://library.imaging.org/archiving/articles/21/1/4) | Velvia 50 controlled target observations | Original ColorChecker Digital SG and IT8 film scans or extracted patch measurements, target references, stock/format/exposure identifiers, scan metadata and an exact asset manifest; independent group identities if known | `SF3.A3P`: physical Velvia 50 target topology is documented, but the public repositories contain no dataset payload or dataset-specific fitting licence |
| 4 | NTNU Colourlab: Irina-Mihaela Ciortan and Giorgio Trumpy at the institutional addresses above; [Colourlab people](https://www.ntnu.edu/idi/people/colourlab) | Velvia 50 controlled multispectral observation; E100 is useful context but does not replace Ektar 100 | Original developed-film multispectral scans, matching hyperspectral references, condition/stock/exposure map, capture and processing metadata, and exact manifest/checksums | `SF2.8R`, `R1HC`: controlled Velvia 50/E100 geometry is described, but public attachments do not establish a licensed, grouped dataset |

FilmMatch and PARVEC are complementary rather than substitutes: the first is
the strongest Portra-only controlled source; the second is the best current
same-workflow Portra/Ektar bridge. FILM2PAINT and the NTNU study are the two
strongest Velvia 50 controlled-measurement holders. Ektachrome E100 must never
be counted as Ektar 100.

## 3. Common requested rights

The request should ask for a short written, non-exclusive authorization that
explicitly states whether the project may:

1. receive and store the exact files privately for internal commercial R&D;
2. decode, measure and fit bounded deterministic colour operators from them;
3. retain and use derived numerical parameters, evaluation summaries and
   non-reconstructive model weights in a commercial product;
4. publish aggregate research metrics and method descriptions with attribution;
5. share only hashes, manifests and small non-reconstructive diagnostics unless
   separate redistribution permission is granted.

Raw-image redistribution is not required. The holder may require an NDA,
attribution, access controls or a no-redistribution clause. A research-only or
non-commercial grant is useful for private scientific comparison but cannot
authorize commercial product fitting; that distinction must remain explicit.

## 4. Requested technical package

Ask each holder for the smallest export that preserves the experiment:

- original or losslessly exported digital reference and physical-film scan or
  measurement files, not screenshots, article figures or graded examples;
- a CSV/JSON manifest with relative path, byte size and SHA-256 where possible;
- exact stock formulation, format and roll/charge identity;
- exposure, illuminant, chart/scene and capture-session identity;
- process/lab/developer and scanner/camera/software identity, or explicit
  `unknown` values rather than inferred metadata;
- a frozen split proposal with development, validation and untouched
  confirmation groups; if no split exists, enough independent groups for the
  project to freeze one before pixels are read;
- known crop, registration, tone, colour-management and post-processing steps.

An export lacking same-scene or target correspondences can still support
stock-identifiability research, but it cannot identify a digital-to-film
operator. An export with only one roll/source/scanner cannot establish
transferability and remains below `S2`.

## 5. Message template

Subject: Request for a licensed research export of controlled photographic-film data

> Hello [name],
>
> I am working on K-MCFM, a deterministic colour-rendering research project.
> Your [project/paper] appears to contain unusually valuable controlled
> observations of [stock(s)]. The public material describes [specific capture
> structure], but I could not locate a reusable data export with exact group
> identities and fitting rights.
>
> Would you be willing to provide, or discuss licensing, the smallest private
> export that preserves the experiment: original digital references and film
> scans or patch measurements, a path/size/hash manifest, and the available
> roll, exposure, process, scanner and session identifiers? Raw redistribution
> is not required. The desired authorization is private internal commercial
> R&D, bounded deterministic-operator fitting, retention of non-reconstructive
> derived parameters, and publication of aggregate metrics with attribution.
> NDA, access-control and no-raw-redistribution terms are acceptable.
>
> We will not describe the result as a calibrated stock response unless it
> passes independent roll/source/scanner holdouts and severe-artifact gates.
> If the files cannot be shared, a patch table or other non-image measurement
> export with the same group metadata may still be useful.
>
> Thank you for considering this request. I can provide a concise inventory and
> rights checklist tailored to your archive before any transfer.

## 6. Holder-specific additions

- **FilmMatch:** ask whether the five training rolls and later validation roll
  remain separable in the archive, whether the Reflecta outputs are linear and
  ungraded, and whether Sony FX3 chart references are available per condition.
- **PARVEC:** ask for the exact Portra 400/Ektar 100 rows first, without waiting
  for the full stock catalogue, and request one untouched roll/session per
  stock if the archive supports it.
- **FILM2PAINT/ZB:** ask whether the 2009 target dataset can be exported as
  patch measurements if scan rights prevent image sharing, and who can grant
  rights for the physical target records.
- **NTNU 2026:** request the condition table before any large hyperspectral
  transfer so that Velvia 50 and E100 observations are not confounded by
  illuminant, exposure or painting identity.

## 7. Response classification and stop rules

| Response | Action |
|---|---|
| Explicit rights plus exact grouped inventory | Freeze a new source-intake contract before any body or pixel read |
| Rights acceptable but group/manifest incomplete | Request only the missing metadata; do not download pixels |
| Data available under non-commercial/research-only terms | Admit only a separately labelled private research lane; no product fitting |
| LUT/profile/product only | Keep as a control; do not treat derived output as observation truth |
| Viewing permission without fitting/derived-output rights | Source remains closed |
| Silence or refusal | Record outreach status only; do not infer scientific failure or retry through alternate identities |

Follow up at most once after 14 calendar days. Do not mass-mail, scrape private
assets, accept click-through terms on the user's behalf or purchase anything
without separate approval.
