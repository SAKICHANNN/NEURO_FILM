from __future__ import annotations

from src.eval.physical_native_ordered_chain_conformance import (
    NativeAdjacencyProfileV1 as EvalAdjacencyProfile,
    adjacency_profile_struct as eval_adjacency_profile_struct,
)
from src.eval.physical_native_print_conformance import (
    NativePrintProfileV1 as EvalPrintProfile,
    profile_struct_from_payload as eval_print_profile_struct,
)
from src.eval.physical_native_spatial_conformance import (
    NativeGaussianProfileV1 as EvalGaussianProfile,
    gaussian_profile_struct as eval_gaussian_profile_struct,
)
from src.film_physics.native_abi_layouts import (
    NativeAdjacencyProfileV1,
    NativeGaussianProfileV1,
    NativePrintProfileV1,
    native_adjacency_profile_struct,
    native_gaussian_profile_struct,
    native_print_profile_struct,
)


def test_p8ba_eval_uses_single_production_abi_layout_source() -> None:
    assert EvalPrintProfile is NativePrintProfileV1
    assert EvalGaussianProfile is NativeGaussianProfileV1
    assert EvalAdjacencyProfile is NativeAdjacencyProfileV1
    assert eval_print_profile_struct is native_print_profile_struct
    assert eval_gaussian_profile_struct is native_gaussian_profile_struct
    assert (
        eval_adjacency_profile_struct
        is native_adjacency_profile_struct
    )
