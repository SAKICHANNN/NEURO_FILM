from src.eval.filmmatch_domain_balanced_capacity import _fit


def test_residual_degree_variant_overrides_default() -> None:
    config = {
        "candidate": {
            "fit": {
                "residual_degree": 3,
                "residual_identity_ridge": 0.01,
            }
        }
    }
    variant = {"residual_degree": 4, "residual_identity_ridge": 0.1}
    assert int(variant.get("residual_degree", config["candidate"]["fit"]["residual_degree"])) == 4
    assert float(
        variant.get(
            "residual_identity_ridge",
            config["candidate"]["fit"]["residual_identity_ridge"],
        )
    ) == 0.1
    assert callable(_fit)
