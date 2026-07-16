# SF0.6B third-stock metadata contract

Audit exact Kodachrome 25, Kodachrome 64, Ektachrome Elite 100 5045 EB,
Ektachrome Elite 200, Vision3 50D and Vision3 250D Commons categories. These
are separate stock/process interpretations; they are not pooled into generic
Kodachrome, Ektachrome or Vision3 families.

The SF0.6A gates remain unchanged: each passing stock needs >=20 files, >=5
uploaders and normalized authors, <=60% largest uploader/author, >=8 explicit
licence-complete derivative rows whose URL differs from the original, >=95%
minimum dimension 512 and complete free-licence/source/SHA1 metadata.

This is metadata-only. One further pass is sufficient to reach three
provisional Commons source candidates (with Ektar and UltraMax) and open a
separately frozen bounded pixel audit. A pass does not open training directly.

Licence enumeration correction: one Kodachrome64 row uses `GFDL 1.2`, which
the FSF publishes as a free copyleft documentation licence. It is accepted for
the all-row metadata sanity check but remains excluded from strict pixel
candidates; the latter stay limited to the frozen CC0/CC BY/public-domain set.

Machine contract: `configs/real_film_commons_third_stock_audit_v1.json`.
