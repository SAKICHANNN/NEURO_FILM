# U5.R2AQ2D source-support diagnosis

AQ2D confirms that the rejected AQ2 operator test is source-support limited.
It cannot promote or rescue an operator.

Two runs reproduce report SHA
`23e05fc5b094908bbc415761bef2b93dc82d4b8b605f3f1ae9c8aad61592365f`
at software commit `1f5e3a57642db53238e18b0b78c01605a8611bb4`.

For each held 288-patch recorder grid, the other four complete grids form the
development support:

| Held slide | held/dev p95 NN ratio | outside development convex hull |
|---:|---:|---:|
| 1 | 1.639 | 0% |
| 2 | 1.267 | 0% |
| 3 | 1.951 | 0% |
| 4 | 1.036 | 38.54% |
| 5 | 0.686 | 11.46% |

Slides 1 and 3 exceed the frozen 1.5x local-distance threshold; slides 4 and
5 exceed the 5% convex-hull threshold. Exact RGB overlap is only 0.35--1.04%.
Slide 4 is therefore locally close in some regions while occupying a large
unsupported hull extension, matching its cross-set AQ2 failure.

The five grids do not provide sufficient redundant coverage to distinguish a
transferable nonlinear recorder-to-slide mapping from grid-specific
interpolation/extrapolation. The branch remains closed. Reopening requires
additional rights-cleared controlled source grids or a distinct physical
evidence source, not more capacity on the same rows.
