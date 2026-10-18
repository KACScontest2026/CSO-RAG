import sys,types,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cso_rag.retrieval import pooling_modes

class Pooling(unittest.TestCase):
    def test_modern_lasttoken(self):
        self.assertEqual(pooling_modes([object(),types.SimpleNamespace(pooling_mode='lasttoken'),object()]),['lasttoken'])
    def test_legacy_lasttoken(self):
        self.assertEqual(pooling_modes([types.SimpleNamespace(pooling_mode_lasttoken=True,pooling_mode_mean_tokens=False)]),['lasttoken'])
    def test_mean_and_combined_are_not_lasttoken_only(self):
        self.assertNotEqual(pooling_modes([types.SimpleNamespace(pooling_mode='mean')]),['lasttoken'])
        self.assertNotEqual(pooling_modes([types.SimpleNamespace(pooling_mode=['mean','lasttoken'])]),['lasttoken'])

if __name__=='__main__':unittest.main()
