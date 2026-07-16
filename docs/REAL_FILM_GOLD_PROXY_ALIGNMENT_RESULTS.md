# RF1.4B0 Gold100 display-proxy alignment results

Date: 2026-07-16

Decision: pass alignment/support eligibility; colour fitting remains forbidden

At commit `929131c4cd5f87be6b3962ea3f44ca4588f4cc4c`, two complete
runs were byte-identical. Report SHA-256:
`f562141ac0b836efa0f361ca730d2d69d8ec197a1c09be366a7543b8fc27ff4c`.

The audit checked all evidence hashes before restricted pickle loading or image
decode. The pickle opcode/global allowlist accepted only NumPy ndarray
reconstruction; persistent IDs, extensions and object constructors were
forbidden.

## Result

- 47/47 Gold100 preview/proxy siblings have official transformation records;
- six independent physical rolls remain, with 2/13/6/12/2/12 pairs;
- every 3x3 matrix is finite and nonsingular;
- every bbox lies inside the preview;
- every bbox width and height exactly equal its proxy dimensions;
- no resize-to-fit, inferred homography, pair dropping or colour fit occurred.

This passes only the data/alignment prerequisite for RF1.4B1. The pseudo-ground
truth represents an archive print/scan display chain. It is not isolated Kodak
Gold stock response and does not establish calibration, authenticity or S2.

Machine decision:
`configs/real_film_gold_proxy_alignment_decision_v1.json`.
