import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


class ResultsSchemaTests(unittest.TestCase):
    def test_phase1_uses_token_rank_payload(self):
        for slug in ('gpt2', 'pythia-70m', 'smollm2-135m', 'qwen2.5-0.5b'):
            path = ROOT / 'phase1-norms-and-structure' / 'results' / slug / 'results.json'
            with open(path) as f:
                data = json.load(f)
            self.assertIn('token_rank', data)
            self.assertNotIn('frequency_proxy', data)

    def test_comparison_schema_drops_shared_vocab_and_outliers(self):
        path = ROOT / 'cross-model-comparison' / 'results' / 'comparison.json'
        with open(path) as f:
            data = json.load(f)

        self.assertIn('concept_inventory_size', data)
        self.assertIn('neighbor_probe_count', data)
        self.assertIn('neighborhood_jaccard', data)
        self.assertIn('ghost_universality', data)
        self.assertIn('analogy_scorecard', data)
        self.assertNotIn('shared_vocab_size', data)
        self.assertNotIn('outlier_migration', data)


if __name__ == '__main__':
    unittest.main()
