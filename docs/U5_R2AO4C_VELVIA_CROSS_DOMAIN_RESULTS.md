# U5.R2AO4C Velvia cross-domain result

Two formal reports are byte-identical at `f909fb9e...0913`. The same bounded
one-matrix family transfers in both directions between the 24-patch Velvia
ColorChecker proxy and the 47-point Velvia stable-pigment proxy:

| Fit → confirm | Identity RMSE | Full affine | One matrix | Gain vs identity / affine |
|---|---:|---:|---:|---:|
| chart → pigment | .10075 | .06411 | **.03812** | 62.16% / 40.53% |
| pigment → chart | .16224 | .11126 | **.05874** | 63.79% / 47.20% |

Both one-matrix predictions remain inside the RGB cube. The affine controls
produce substantial out-of-cube values.

The wrong-stock diagnostic is directionally encouraging but not confirmatory.
The Velvia chart operator scores `.07352` on the Ektachrome palette, slightly
worse than its chart-fit full affine (`.07126`), while it scores `.03812` on
the Velvia palette. An Ektachrome-palette operator scores `.04955` on that
Velvia palette, 23.06% worse than the correct Velvia chart operator. Because
the two film stocks use different paintings, content remains confounded and no
stock-specificity claim opens.

This is stronger than chart self-consistency: a fixed real-film-derived
operator generalizes to independent pigment colours. It remains one paper,
one rendering pipeline, author-rendered sRGB, and non-same-time film/HSI
evidence. The next bounded challenge fits the 71 exact Velvia pairs jointly
before returning to the full-photo style/artifact frontier.
