# U6.P6ZK bounded scanner-chain profile-file contract

## Question

Can the exact U6.P6ZI canonical generic scanner profile be carried as one
small, deterministic, create-only research file and loaded into the unchanged
U6.P6ZJ bundle executor without normalizing bytes, trusting file metadata or
executing pixels before identity validation?

## Frozen parent and artifact

- Parent profile: exact U6.P6ZI canonical profile, 1,006 bytes, SHA-256
  `3ebdee37366871460b1f5feaedbe0362c59e593efd3485652a892183bba6b891`.
- Parent execution: unchanged U6.P6ZJ caller-verified bundle executor and
  evidence.
- File schema: `neuro_film.generic_scanner_chain_profile_file.v1` with exactly
  `schema`, `profile_sha256`, and `profile` fields. The nested profile is the
  strict P6ZI JSON object, not base64 or a path reference.
- The complete canonical ASCII file is frozen at 1,162 bytes and SHA-256
  `38ed5d2ea154eefdd3f1a264ff86db313f963e3f1ff556c57729fe7ce9c78350`.
- Maximum accepted file size is 4,096 bytes. Newline, BOM, duplicate keys,
  nonfinite constants, unknown fields, noncanonical ordering/whitespace,
  symlinks and non-regular files are forbidden.
- The file contains no image, output, machine path, calibrated scanner claim,
  stock/process identity or product authority.

## Frozen execution and gates

Two fresh processes, in forward and reverse control order, must pass:

1. all P6ZI/P6ZJ parent file hashes and evidence identities remain exact;
2. encoded file bytes, byte count, file SHA and nested profile SHA are exact;
3. create-only publication preserves an existing foreign destination and an
   injected late destination, removes only its owned sibling stage, and leaves
   no residue;
4. strict load requires caller-held expected file SHA and profile SHA before
   returning canonical profile bytes;
5. wrong file/profile identity, tamper, noncanonical JSON, duplicate key,
   oversize, symlink and malformed input reject before scanner pixel execution;
6. direct P6ZJ bundle execution and file-loaded execution are bit-exact and
   repeat-exact on the frozen procedural float64 transmittance probe;
7. no network, external image, target, calibration or product data is read;
8. forward and reverse reports are byte-identical.

Failure closes the file package and retains P6ZI/P6ZJ in-memory mechanics.
Pass retains only a private research artifact transport and closes adjacent
installer, discovery, signing, archive, runtime and product wrappers.

## Claim ceiling

This leaf can establish deterministic bounded local file transport for one
generic physical-inspired experimental scanner profile. It cannot establish
scanner calibration, photographic quality, film-stock response, arbitrary
profiles, public schema/package/capability, installer/discovery, native
runtime, signing or product readiness.
