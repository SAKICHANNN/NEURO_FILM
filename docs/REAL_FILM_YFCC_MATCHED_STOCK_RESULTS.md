# SF1.0A YFCC same-source stock support

Date: 2026-07-16  
Decision: **retain YFCC Ektar100 as a prospective source bridge; stop YFCC UltraMax400**.

The already frozen 7.35M-row YFCC15M metadata was rescanned with preregistered exact `Ektar 100` and `UltraMax/Ultra Max 400` expressions. Only CC BY 2.0 static-photo rows were eligible. The audit ran twice with the identical SHA-256 `966032057c1b490760060f14e3dbe8ba5e7e1702a38184e83434e5ce9f9e10b5`.

| Stock | Rows | Flickr UIDs | Largest UID share | Gate |
|---|---:|---:|---:|---|
| Kodak Ektar 100 | 26 | 10 | 23.08% | pass |
| Kodak UltraMax 400 | 16 | 5 | 68.75% | fail dominance |

UltraMax receives no YFCC pixel pilot; its 11-row dominant UID is an obvious source/content shortcut. The Ektar pool contains no matches to the prospectively reused cross-process, HDR, multiple-exposure or B&W-developer exclusions.

If YFCC Ektar pixels pass live rights and visual gates, the experiment graph becomes connected: YFCC compares Ektar to Velvia50, Commons compares Ektar to UltraMax400, and Ektar is the bridge stock. This does not guarantee identifiability, but it makes source-versus-stock effects testable in a way that the original disconnected three-stock pool did not.

SF1.0A2 may retain at most 20 YFCC Ektar files with at most four per UID. No classifier or colour operator is fitted before that pixel bridge passes.
