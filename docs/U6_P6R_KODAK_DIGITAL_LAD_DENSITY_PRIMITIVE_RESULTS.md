# U6.P6R Kodak Digital LAD density-primitive results

The H-387 guide tables compile into an exact typed neutral primitive. It keeps
10-bit Cineon code, negative/interpositive recording mode, raw printing
density, nonnegative physical density, Status M above D-min, D-min and total
Status M as distinct quantities.

Two reports are byte-identical at `f5ec8fde...55510`; stable evidence ID
`d9b5d734...c816c`. Across all 1,024 codes, formula/table error is
`4.44e-16` and raw density-to-code roundtrip error is `1.14e-13`. Negative is
strictly increasing, interpositive strictly decreasing, the raw IP tail at
code 1023 remains `-0.116` while the explicitly separate physical value is
zero, and all four 445-code LAD aims preserve `above-D-min + D-min = total`.
Serialization and partition execution are exact.

No RGB image transform, fit, render or product integration occurred. P6S may
now test compatibility with the existing density/print interpretation types;
it must not use this neutral calibration primitive as a film-look operator or
as measured local stock/process/scanner evidence.
