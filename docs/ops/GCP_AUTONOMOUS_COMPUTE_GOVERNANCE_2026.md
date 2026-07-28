# GCP Autonomous Compute Governance

Status: **authorized but not yet used**

Owner scope: Goal thread `019f4b76-e70a-75c0-b7ea-b473ab38c200`

Hard budget: **USD 2,500**; warning at **USD 2,000**; no new starts at
**USD 2,250**

## Authority

The user authorizes this Goal thread to autonomously create, run, stop and
clean up its own GCP compute, training, storage and job resources within the
USD 2,500 hard budget.

This authorization overrides the earlier default prohibition on paid
cloud/GPU use for this Goal. It does not relax any evidence, security,
licensing, model-output or product-promotion gate.

All other project-scoped reversible research and engineering choices are also
autonomous unless a platform-enforced action cannot be preauthorized.

## Frozen ownership namespace

Every resource created by this thread must use:

- gcloud configuration: `nf-019f4b76-agent`;
- prefix: `nf-019f4b76-`;
- name pattern: `nf-019f4b76-<purpose>-<unique-suffix>`;
- labels, when the resource supports them:
  - `owner=codex-thread`
  - `thread=019f4b76e70a75c0b7eab473ab38c200`
  - `project=neuro-film`
  - `goal=ultimate-stock-first`

An existing, unknown or differently prefixed resource is foreign. Discovery
may read enough metadata to avoid collision and estimate capacity, but this
thread must never modify, stop, delete, rename, reconfigure or occupy it.

The local ownership ledger is authoritative. A resource may be stopped or
deleted only when its exact full name is recorded with
`created_by_this_thread=yes`.

## Before the first cloud mutation

1. Read current Google Cloud official documentation for the exact service,
   command/API, pricing, quotas, lifecycle and cleanup behavior.
2. Verify the active `gcloud` account/configuration/project read-only without
   printing tokens or raw credentials.
3. Reuse existing OAuth. Do not run `gcloud auth login` unless a read-only
   check proves the credential is invalid; report that state before login.
4. Record the intended resource full name, region/zone, purpose, shape,
   maximum runtime, idle timeout, worst-case cost and cleanup command in the
   ledger as `planned`.
5. Confirm the worst-case cumulative cost remains below the USD 2,250
   new-start threshold and the USD 2,500 hard ceiling.
6. Confirm labels/prefix, private access, least privilege, input/output paths,
   checkpoint/resume and failure cleanup.

## Runtime discipline

Every paid job must have:

- bounded maximum runtime and cost;
- liveness/progress evidence;
- idle timeout and no idle paid accelerator;
- checkpoint/resume where useful;
- completion, failure, cancellation and thread-interruption cleanup;
- periodic thread-cost reconciliation;
- a retained job/resource ID and concise operation log.

When a job remains `PENDING`, do not wait indefinitely or resubmit blindly.
Inspect, in order:

1. region/zone;
2. quota and actual accelerator capacity;
3. machine/accelerator shape compatibility;
4. container/image availability;
5. service-account permission and API enablement;
6. request schema and command arguments;
7. input paths, data format and job content.

Correct the diagnosed cause before a replacement submission, and first ensure
the previous submission cannot start and bill concurrently.

## Cost and cleanup

- Hard cumulative ceiling: USD 2,500.
- Warning threshold: USD 2,000; no new paid start at or above USD 2,250.
- A preflight estimate is mandatory.
- A job must not be submitted if its worst-case cost would cross the ceiling.
- Stop paid machines immediately when they are not needed.
- Delete only this thread's disposable resources when their evidence has been
  retained.
- Record actual or best-available cost, final state and cleanup outcome in the
  ledger.
- Never use this authorization to claim ownership of a shared-project
  resource.

## Security

- No public service or public bucket.
- No IAM change beyond the minimum required for a thread-owned resource; if a
  platform gate requires separate human confirmation, stop that specific
  action and continue other work.
- Never echo, persist, log or commit credentials, tokens, keys or raw auth
  headers.
- The DeepSeek key must never be written to disk or logs.
- Cloud outputs remain subject to the same rights, lineage, leakage and claim
  ceilings as local outputs.

## Current state

No GCP mutation has occurred under this governance leaf. Thread-owned resource
count and accounted spend are both zero. Local RTX 5070 Ti Laptop work remains
preferred when it is scientifically and operationally sufficient.
