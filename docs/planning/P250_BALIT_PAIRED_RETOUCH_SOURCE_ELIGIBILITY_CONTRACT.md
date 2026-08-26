# P250 BALit paired-retouch source eligibility contract

## Question

Does the official 2026 BALit record provide a genuinely new paired RAW to
expert-retouched observation with the public payload, commercial-use rights,
and scene/file grouping facts required to request a new bounded
single-reference scientific cycle?

## Scope and frozen sources

- Official DOI: `https://doi.org/10.21227/cdqg-q549`.
- Official IEEE DataPort record:
  `https://ieee-dataport.org/documents/balit-dataset-backlit-low-light-image-enhancement`.
- Discovery facts are not acceptance evidence: the public record describes
  1,000 Canon CR2 inputs and 1,000 expert-retouched references, approximately
  21.5GB and 26.84GB respectively, but also presents a login-to-access
  boundary. P250 re-reads only official metadata after this contract is
  committed.
- No dataset file, thumbnail, image URL, archive, checksum-listed payload,
  account session, cookie, paper PDF, model, code or third-party mirror may be
  requested.

## Frozen execution

1. Perform two fresh-process unauthenticated reads of the DOI and official
   record only, with bounded response size and no cookies or credentials.
2. Record final URLs, response size/SHA-256 and deterministic normalized facts.
3. Search only the returned official text for pair counts/sizes, explicit
   licence terms, anonymous file access, file-level manifest/checksums, and
   scene/group/split identifiers.
4. Do not infer commercial rights from `Open Access`, research intent,
   citation instructions, an IEEE account requirement, or absence of an
   explicit prohibition.
5. Do not infer scene-group isolation from filenames being pairable or from a
   generic instruction not to mix train/validation/test rows.

## Frozen eligibility gates

All gates must pass simultaneously:

1. **official identity:** DOI and DataPort record resolve to the same named
   BALit dataset and publisher record;
2. **paired observation:** the official record explicitly binds 1,000 original
   Canon RAW inputs to 1,000 expert-retouched references;
3. **public payload:** dataset files are anonymously enumerable and retrievable
   without accepting terms, logging in, subscription, request or contact;
4. **commercial rights:** the official dataset record carries an explicit
   commercial-compatible licence covering both RAW inputs and retouched
   references;
5. **reproducible inventory:** file-level names, sizes and checksums or an
   equivalently exact manifest are public before payload access;
6. **group isolation:** explicit scene/file group identifiers and fixed or
   derivable group-disjoint development/confirmation roles are public;
7. **exact replay:** both metadata-only reports are byte-identical.

Failure of any one gate yields
`NOT_READY_BALIT_SOURCE_RIGHTS_ACCESS_OR_GROUPING_GAP_NOT_SCIENTIFIC_RESULT`.
It does not consume candidate 3 and does not authorize account creation,
download, mirror use, rights assumption, model fitting or image scoring.

## Claim ceiling

P250 is a metadata-only source eligibility audit. A pass could open only a
separate preregistered member-integrity and paired-observation preflight. It
cannot establish editing quality, arbitrary-reference identifiability,
A1/A4/A5, a shared operator, data redistribution, package/schema/capability or
product admission.
