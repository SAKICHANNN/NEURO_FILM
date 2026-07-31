# U5.R2BL0 FilmMatch source-conditioned operator result

BL0 tests the missing adaptive algorithm directly: source-only chart
statistics predict the twelve parameters of an explicit monotone-curve plus
positive-matrix operator. Each outer fold holds out one complete illuminant;
normalization, ridge selection and fitting use development groups only. Two
formal reports are byte-identical at `abe7b1e7...2ed`.

There is real conditional value. A per-group evaluator Oracle improves RGB
RMSE over the global operator by `17.86%`. The predicted operator improves
aggregate RGB RMSE by `6.98%`, lowers aggregate P95 to `.8166x` global, and
beats the label-shuffled predictor by `11.48%`. It is also structurally safe:
zero out-of-cube output and minimum cube Jacobian `.2844`.

The required group stability fails. Adaptive prediction wins only `13/33`
groups (`39.39%`, below the frozen `66.67%` gate). It loses all 15 negative-EV
groups, while winning all nine `+3..+5 EV` groups. This is a selective-regime
signal, not a reliable dense router.

Close BL0 as `oracle_only` and keep the global fallback. Do not hide the
instability behind aggregate gains or reinterpret source statistics as
physical exposure/illuminant truth. A next algorithm may preregister a hard
abstaining policy, but the consumed validation scene cannot serve as fresh
confirmation and this session cannot support stock calibration or product
promotion.
