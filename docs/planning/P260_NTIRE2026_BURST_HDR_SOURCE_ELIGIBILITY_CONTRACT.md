# P260 NTIRE 2026 burst-HDR source eligibility contract

## Question

Does the official NTIRE 2026 Efficient Burst HDR and Restoration release provide
a genuinely new multi-exposure RAW capture observation whose code, data rights,
public payload and scene grouping are sufficient for a separate prospective
consumer experiment?

## Scope and frozen sources

- Official challenge repository: `https://github.com/Eve-ctr/RawFusion`, frozen
  at `1ac0212b73b3dbdabb262bde36b51aad7abee971`.
- Official CVF paper: *NTIRE 2026 Challenge on Efficient Burst HDR and
  Restoration: Datasets, Methods, and Results*.
- The discovery pass saw a highly relevant structure (nine heterogeneous-
  exposure RAW frames plus one RGB ground truth per scene), but also repository
  text restricting training-data sharing and other-purpose use. Discovery is
  not admission evidence; P260 re-reads only the frozen official sources.
- No dataset file, image, thumbnail, archive, checkpoint, model, competition
  account/session, third-party mirror or executable code may be requested.

## Frozen execution

1. In two fresh processes, request only bounded official GitHub API metadata,
   the commit-pinned README, commit-pinned recursive tree and the official CVF
   paper page/PDF metadata needed to bind identity and public claims.
2. Record response URL/status/bytes/SHA-256 and normalize only explicit facts:
   scene/frame/ground-truth counts, exposure grouping, public payload locator,
   dataset use terms, repository licence, exact inventory/checksums and scene
   split/group identifiers.
3. Absence of a licence is not permission. Challenge participation, paper open
   access, public repository visibility or dataset-download instructions do not
   establish commercial-compatible code/data rights.
4. A restriction against sharing or other-purpose use fails the rights gate
   even if participants could previously download the challenge payload.

## Frozen eligibility gates

All gates must pass simultaneously:

1. **official identity:** commit-pinned repository and CVF paper identify the
   same NTIRE 2026 challenge;
2. **new physical observation:** official sources explicitly define scenes with
   nine noisy/misaligned heterogeneous-exposure RAW inputs and an aligned RGB
   ground truth;
3. **anonymous public payload:** the current official surface exposes payload
   files without account, competition enrollment, request or contact;
4. **commercial-compatible data rights:** explicit terms permit project use of
   every input and ground-truth payload outside challenge participation;
5. **commercial-compatible code rights:** an explicit repository licence
   covers the executable starting kit/baseline;
6. **exact inventory:** public file names plus checksums or an equivalently exact
   manifest bind the available payload before download;
7. **group isolation:** public scene/split identities support file-disjoint,
   scene-disjoint development and confirmation roles;
8. **exact replay:** both metadata-only reports are byte-identical.

Failure of any gate yields
`NOT_READY_NTIRE2026_BURST_HDR_RIGHTS_OR_PAYLOAD_GAP_NOT_SCIENTIFIC_RESULT`.
It does not consume candidate 3 and authorizes no dataset/model download,
training, inference, pixel read, mirror use, licence inference or product map.

## Claim ceiling

P260 is a current official-source eligibility audit only. A complete pass could
open only a separately preregistered payload-integrity and physical-observation
preflight. It cannot establish HDR quality, arbitrary RAW support, a shared
operator, A1/A4/A5, package/schema/capability or product admission.
