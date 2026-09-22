import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_chroma_null_domain_v3 import null_candidates, null_split
from src.models.semantic_film_adversarial_v1 import balanced_schedule


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {**json.loads((ROOT / "configs/semantic_film_adversarial_v1.json").read_text()), **json.loads((ROOT / "configs/chroma_constrained_film_v2.json").read_text()), **json.loads((ROOT / "configs/chroma_null_domain_v3.json").read_text())}


class ChromaNullDomainTests(unittest.TestCase):
    def test_frozen_sha_rule_counts_and_disjoint_split(self):
        rows = [{"id": f"casebank_{i:04d}_x{i * 7}"} for i in range(381)]
        sources, references, collections = null_split(rows, 38)
        expected = sorted(range(381), key=lambda i: hashlib.sha256(rows[i]["id"].encode()).hexdigest())[:38]
        self.assertEqual(references, sorted(expected))
        self.assertEqual(len(sources), 343)
        self.assertFalse(set(sources) & set(references))
        self.assertEqual(sorted(sources + references), list(range(381)))
        self.assertEqual(np.bincount(collections[references], minlength=3).tolist(), [13, 13, 12])
        self.assertTrue((collections[sources] == -1).all())
        self.assertEqual(int(collections[expected[0]]), 0)
        self.assertEqual(int(collections[expected[4]]), 1)
        again = null_split(list(reversed(rows)), 38)[1]
        self.assertEqual(sorted(380 - i for i in again), references)

    def test_candidates_keep_existing_rows_and_schedule_never_leaks(self):
        rows = [{"id": f"img{i}"} for i in range(30)]
        sources, references, collections = null_split(rows, 9)
        candidates = np.asarray([[i, 3, 3, 49, 49, (i + k) % 3] for i in range(30) for k in range(3)])
        source_pool, reference_pool = null_candidates(candidates, sources, references)
        self.assertEqual(len(source_pool) + len(reference_pool), len(candidates))
        np.testing.assert_array_equal(np.concatenate((source_pool, reference_pool))[np.lexsort(np.concatenate((source_pool, reference_pool)).T[::-1])], candidates[np.lexsort(candidates.T[::-1])])
        config = {**CONFIG, "steps": 30, "batch_size": 4, "clusters": 3}
        schedule, report = balanced_schedule(source_pool, reference_pool, collections, config, 220922)
        self.assertTrue(np.isin(schedule["reference"][..., 0], references).all())
        self.assertFalse(np.isin(schedule["source"][..., 0], references).any())
        np.testing.assert_array_equal(collections[schedule["reference"][..., 0]], schedule["collections"])
        self.assertEqual(report["collection_counts"], [40, 40, 40])
        np.testing.assert_array_equal(schedule["source"][..., -1], schedule["reference"][..., -1])

    def test_config_changes_only_reference_distribution(self):
        v2 = json.loads((ROOT / "configs/chroma_constrained_film_v2.json").read_text())
        self.assertEqual(CONFIG["arm"], "A")
        self.assertEqual(CONFIG["field_bounds"]["A"], v2["field_bounds"]["A"])
        self.assertEqual(CONFIG["seeds"], v2["seeds"])
        self.assertEqual((CONFIG["steps"], CONFIG["batch_size"]), (v2["steps"], v2["batch_size"]))
        self.assertEqual(len(set(CONFIG["blind_labels"])), 8)
        self.assertFalse(set(CONFIG["blind_labels"]) & set("PQRSTUVW"))


if __name__ == "__main__":
    unittest.main()
