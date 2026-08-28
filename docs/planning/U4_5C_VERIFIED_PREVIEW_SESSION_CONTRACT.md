# U4.5C — Verified three-stock preview session

## Question

Can the private three-stock desktop workflow meet the existing 300 ms warm
preview target without weakening U7.3H's complete-file validation?

## Frozen mechanism

One session admission must validate the complete input, render profile and all
three U7.3G preview files through the unchanged U7.3H cache index.  It must then
read each preview into owned immutable bytes and independently re-hash those
bytes against the cache row.  Only this admitted snapshot may serve warm
lookups.

A warm lookup performs no filesystem read, pixel decode, render or write.  It
returns the three ordered immutable preview byte strings and their frozen
identities.  Disk mutation after admission cannot change the admitted snapshot;
a new admission after the same mutation must fail closed.

## Gates

- four fresh session admissions, each followed by eight warm lookups;
- every admission performs complete input/profile/output hash validation;
- every admitted preview byte string re-hashes to its frozen output identity;
- all warm lookups return the exact Velvia 50, Portra 400 and Ektar 100 bytes in
  canonical order;
- maximum individual warm lookup wall time is 0.300 seconds;
- warm lookup filesystem reads, pixel decodes, renders and writes are zero;
- post-admission disk mutation leaves the current snapshot unchanged and makes
  a new admission fail closed;
- cross-process scientific payloads are exact after excluding timing.

## Stop rules and claim ceiling

Do not skip admission hashing, accept a different file identity, copy mutable
buffers, change the three stock operators or reinterpret U7.3H's failed
fresh-process timing.  Failure closes this session-resident mechanism without
threshold rescue.

Passing establishes only a private Windows/Python session-resident warm lookup
for three non-calibrated Look Approximation previews.  It is not render
acceleration, stock calibration or distinguishability, installer/release,
cross-platform support, or completion of the multi-stock scientific goal.
