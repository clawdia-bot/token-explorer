import unittest

from common.models import load_model
from common.probes import ALL_REQUIRED_CONCEPTS, validate_model_probes


class ProbeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {
            slug: load_model(slug)
            for slug in ('gpt2', 'pythia-70m', 'smollm2-135m', 'qwen2.5-0.5b')
        }

    def test_all_required_concepts_resolve_exactly(self):
        for slug, model in self.models.items():
            resolved = validate_model_probes(model)
            self.assertEqual(set(resolved), set(ALL_REQUIRED_CONCEPTS))
            self.assertTrue(all(isinstance(idx, int) for idx in resolved.values()))


if __name__ == '__main__':
    unittest.main()
