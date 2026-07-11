# Reproducibility Baseline

`configs/reproducibility_baseline.json` is the tracked local-safe contract for
the test suite and existing benchmark registry.  It is deliberately narrower
than a product-quality benchmark: the historical six regression fixtures are
not the U4 severe-artifact gold set, and this file makes no quality or stock
authenticity claim.

Run the verifier without touching image assets:

```powershell
.\.venv\Scripts\python.exe scripts\verify_reproducibility_baseline.py
```

To capture the machine/runtime facts locally, write only to an ignored output:

```powershell
.\.venv\Scripts\python.exe scripts\verify_reproducibility_baseline.py `
  --write-environment outputs\reproducibility\environment.json --json
```

The verifier checks hashes for the benchmark/fixture registries and color
guardrails/profiles, checks the required test files, then records Python,
platform, selected package versions and current commit.  Deliberate config
changes must update their checksum in the same reviewed commit.  It never
downloads dependencies, reads private images, renders output, or writes a
source manifest.

The committed GitHub workflow executes the same verifier and pytest with a
small CPU-only dependency set. It becomes active only if a repository owner
pushes this branch; no remote run is started by creating the workflow.
