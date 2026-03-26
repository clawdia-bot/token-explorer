import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
PHASE3B = ROOT / 'phase3-positional-embeddings' / 'rope-results'


class Phase3BRopeSchemaTests(unittest.TestCase):
    def test_gpt2_is_explicit_non_rope_skip(self):
        with open(PHASE3B / 'gpt2' / 'results.json') as f:
            data = json.load(f)

        self.assertFalse(data['compatible'])
        self.assertEqual(data['model']['position_type'], 'learned_absolute')
        self.assertIn('skip_reason', data)

    def test_rope_models_have_operator_and_qk_sections(self):
        for slug in ('pythia-70m', 'smollm2-135m', 'qwen2.5-0.5b'):
            with open(PHASE3B / slug / 'results.json') as f:
                data = json.load(f)

            self.assertTrue(data['compatible'])
            self.assertEqual(data['model']['position_type'], 'rope')
            self.assertIn('rope', data)
            self.assertIn('relative_kernel', data)
            self.assertIn('qk_rotation', data)
            self.assertIn('mean_query_drift_by_position', data['qk_rotation'])
            self.assertIn('score_spearman_by_position', data['qk_rotation'])


if __name__ == '__main__':
    unittest.main()
