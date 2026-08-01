# U6.P3P positive-spread backing-return result

P3P tests a mechanism-distinct compiler after P3N/P3O: retain only the
positive exterior lobe `max(blur(E)-E, 0)` of the unchanged P3D return profile,
then apply the existing shared-RGB sensitometry response bound. This is a
generic compiled approximation, not linear optical transport or a measured
emulsion.

Two formal reports are byte-identical at `c9a890f1...c0037` (stable ID
`21febb41...522a5`). The mechanism is clean: constants and impulse centres are
exact identities; the edge adds only on the dark side; impulse halo energy is
`.35377`; far red/blue response is `9.222x`; and no exposure leaves its domain.
The FFT implementation matches a direct symmetric-convolution oracle while
removing the original large-kernel runtime bottleneck.

The sole failed gate is decisive. Maximum scan-transmittance change is only
`.004091`, below the frozen `.01` visible-response floor. The configured `.02`
safety cap is inactive, so increasing it cannot help. Close P3P without
amplifying P3D fractions or lowering visibility. Retain this as negative
mechanism evidence and move to a different colour/physical algorithm.
