# U5.R2BL14 FilmMatch hard-abstention result

BL14 tests the only follow-up authorized by BL0: a hard choice between the
unchanged bounded adaptive operator and the unchanged global fallback.  The
single routing fact is source median luma.  For every held illuminant, its
threshold is selected only from leave-one-development-illuminant-out
predictions; there is no blend and no outer-target tuning.

Two complete reports are byte-identical at `d5c91849...7171c` with stable
evidence ID `b668b141...d7683`.  Hard fallback extracts real conditional value:
it improves RGB RMSE by `9.50%` over global and `2.71%` over BL0 dense routing,
activates 15/33 groups, wins 11/15 active groups, improves all three held
illuminants and lowers aggregate P95 to `.8397x` global.  Both possible output
operators remain structurally safe: zero out-of-cube output and minimum cube
Jacobian `.2844`.

The frozen stability gates fail.  The development threshold for held 3200K
admits `EV+0/+1`; `3200K|ev=+0` regresses to `1.18075x` global, above the
`1.05x` maximum.  Improvement over the independently fitted shuffled-hard
control is `4.8185%`, just below the frozen `5%` floor.  These are deterministic
failures, not numerical noise.

Close source-conditioned hard routing and retain the global fallback.  Do not
retune quantiles, add another score on this consumed session, or call source
brightness physical exposure.  Future algorithm work must use a
mechanism-distinct explicit/physical representation or genuinely independent,
content-diverse paired evidence.
