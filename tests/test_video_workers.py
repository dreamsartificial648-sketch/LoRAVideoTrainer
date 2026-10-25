"""Run inside an installed video runtime; exercises real PEFT adapters without model downloads."""
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8)
        self.to_k = torch.nn.Linear(8, 8)
        self.to_v = torch.nn.Linear(8, 8)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(8, 8))

    def forward(self, x):
        return self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x))


@unittest.skipUnless(importlib.util.find_spec("peft"), "Run worker tests in an installed video runtime.")
class WorkerAdapters(unittest.TestCase):
    def test_train_save_reload_and_reject_other_architecture(self):
        for backend, other in (('modelscope_video', 'ltx_video'), ('ltx_video', 'modelscope_video')):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                worker = importlib.import_module(f'tools.backends.{backend}')
                wrong = importlib.import_module(f'tools.backends.{other}_contract')
                torch.manual_seed(42)
                base = Attention()
                base_weights = {k: v.clone() for k, v in base.state_dict().items()}
                base.requires_grad_(False)
                model = worker.inject_lora(base, 2, 2)
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable, lr=0.01)
                x = torch.randn(1, 4, 8)
                before = model(x).detach().clone()
                model(x).square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable))
                optimizer.step()
                after = model(x).detach()
                self.assertFalse(torch.equal(before, after))
                path = Path(directory) / 'adapter.safetensors'
                worker.save_adapter(model, path, {'rank': 2, 'alpha': 2}, 1, 1)
                self.assertEqual(worker.contract.checkpoint_info(path)['rank'], '2')
                with self.assertRaises(ValueError):
                    wrong.checkpoint_info(path)
                restored = Attention()
                restored.load_state_dict(base_weights)
                restored.requires_grad_(False)
                restored = worker.load_lora(restored, path)
                torch.testing.assert_close(restored(x), after)
                zero = Attention()
                zero.load_state_dict(base_weights)
                expected = zero(x).detach()
                zero.requires_grad_(False)
                zero = worker.load_lora(zero, path, strength=0)
                torch.testing.assert_close(zero(x), expected)


if __name__ == '__main__':
    unittest.main()
