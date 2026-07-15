# RF0.3 FSA/OWI grouping contract

The collection does not expose a physical-roll field and the Commons mirror
does not provide an exact LOT/assignment identifier for every canonical scan.
This contract therefore freezes a conservative leakage guard, not a recovered
ground-truth assignment.

The LOC finding aid says that an assignment was usually given a block of
sequential negatives. Records are grouped only when they share a curated
creator category and consecutive LOC numeric identifiers are separated by at
most five. The deliberately wide gap merges nearby sequences to reduce the
chance that related frames cross folds. It may merge distinct assignments;
that lowers statistical power but is safer than optimistic leakage.

Unknown-creator records are stress-only. The primary evaluation is
leave-one-creator-out; the secondary evaluation holds out an entire
`sequence_guard_group`. Random image splits are forbidden. These groups must
never be described as physical rolls, development batches, scanner sessions or
exact LOTs.

The gate and thresholds are in
`configs/real_film_fsa_owi_grouping.json`. Passing permits a bounded Phase C
download and RF1 identifiability work. It does not establish film signal.
