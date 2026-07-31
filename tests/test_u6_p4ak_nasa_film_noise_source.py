from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.eval.nasa_film_noise_source import NasaFilmNoiseSourceError, analyze_machine_text, load_contract
ROOT=Path(__file__).resolve().parents[1]; CONTRACT=ROOT/"configs/u6_p4ak_nasa_film_noise_mechanism_source_v1.json"
def test_contract_mutation_fails_closed(tmp_path: Path) -> None:
    value=json.loads(CONTRACT.read_text(encoding="utf-8")); value["source"]["maximum_pdf_bytes"]=50000000; path=tmp_path/"contract.json"; path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(NasaFilmNoiseSourceError,match="contract drift"): load_contract(path)
def test_machine_text_requires_anchors_and_conservative_numeric_table() -> None:
    anchors=["Film granularity noise","film response"]
    text="Film granularity noise is separate. Film response now becomes a function of the background exposure.\nExposure granularity\n0.1 2\n0.2 3\n"
    result=analyze_machine_text(text,anchors); assert all(result["anchor_results"].values()); assert not result["observations"]["reusable_numeric_granularity_table"]
