import unittest

from common.models import load_model


class ModelPositionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpt2 = load_model('gpt2')
        cls.pythia = load_model('pythia-70m')

    def test_gpt2_exposes_learned_absolute_position_embeddings(self):
        self.assertTrue(self.gpt2.has_learned_positions)
        self.assertEqual(self.gpt2.position_type, 'learned_absolute')
        self.assertIsNotNone(self.gpt2.pos_emb)
        self.assertEqual(self.gpt2.pos_emb.shape, (1024, self.gpt2.hidden_dim))
        self.assertEqual(self.gpt2.max_positions, 1024)

    def test_pythia_reports_rope_without_position_matrix(self):
        self.assertFalse(self.pythia.has_learned_positions)
        self.assertEqual(self.pythia.position_type, 'rope')
        self.assertIsNone(self.pythia.pos_emb)
        self.assertEqual(self.pythia.max_positions, 2048)


if __name__ == '__main__':
    unittest.main()
