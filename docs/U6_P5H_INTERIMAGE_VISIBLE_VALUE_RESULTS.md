# U6.P5H interimage visible value

Decision: **automatic metric pass; blind visible-value failure; runtime route
closed**.

Two reports and all three blind sheets are byte-identical. Exact metrics show:

- median `2.086%` changed pixels, with a maximum one-code sRGB8 change;
- positive strong-edge opponent-gradient gain on 18/18 images, median
  `0.949%`;
- median strong-edge/flat changed-pixel concentration `2.167`.

Before reading the key, all 27 unamplified one-to-one A/B pairs were recorded
as ties. After reveal, candidate preference is therefore `0/9` in each of
three rounds, below the frozen `6/9` gate. There are zero severe failures.

P5F remains useful research evidence that a bounded cross-layer mechanism can
be exact and artifact-safe. It does not add reliable visible value at the
frozen strength. Do not retune it on this cohort or spend runtime-integration
work on the unchanged candidate; retain P5C as the simpler authority.
