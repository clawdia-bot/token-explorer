import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
PHASE3 = ROOT / 'phase3-positional-embeddings' / 'results'


class Phase3ResultsSchemaTests(unittest.TestCase):
    def test_gpt2_phase3_schema_contains_structured_metrics(self):
        with open(PHASE3 / 'gpt2' / 'results.json') as f:
            data = json.load(f)

        self.assertTrue(data['compatible'])
        self.assertEqual(data['model']['position_type'], 'learned_absolute')
        self.assertIn('position_norms', data)
        self.assertIn('position_similarity', data)
        self.assertIn('pca', data)
        self.assertIn('periodicity', data)
        self.assertIn('token_position_subspace', data)
        self.assertIn('token_position_interaction', data)

    def test_rope_models_are_explicit_skips(self):
        for slug in ('pythia-70m', 'smollm2-135m', 'qwen2.5-0.5b'):
            with open(PHASE3 / slug / 'results.json') as f:
                data = json.load(f)

            self.assertFalse(data['compatible'])
            self.assertEqual(data['model']['position_type'], 'rope')
            self.assertIn('skip_reason', data)
            self.assertNotIn('position_norms', data)


if __name__ == '__main__':
    unittest.main()
