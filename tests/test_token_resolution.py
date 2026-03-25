import unittest

from common.models import load_model, resolve_token_exact, resolve_token_loose


class TokenResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model('gpt2')

    def test_strict_requires_exact_surface_form(self):
        self.assertIsNone(resolve_token_exact(self.model, 'Tokyo'))
        self.assertIsNotNone(resolve_token_exact(self.model, ' Tokyo'))

    def test_loose_keeps_space_prefixed_fallback(self):
        strict_idx = resolve_token_exact(self.model, ' Tokyo')
        loose_idx = resolve_token_loose(self.model, 'Tokyo')
        self.assertEqual(loose_idx, strict_idx)

    def test_loose_keeps_case_insensitive_behavior(self):
        self.assertIsNone(resolve_token_exact(self.model, 'tOkYo'))
        self.assertEqual(
            resolve_token_loose(self.model, 'tOkYo'),
            resolve_token_exact(self.model, ' Tokyo'),
        )


if __name__ == '__main__':
    unittest.main()
