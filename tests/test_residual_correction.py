import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hydropfn.models.residual_correction import ResidualCorrection


class CorrectionTests(unittest.TestCase):
    def test_exact_parity_learning_and_no_baseline_gradient(self):
        torch.manual_seed(4)
        for latent_dim in (0,8):
            head=ResidualCorrection(3,2,latent_dim)
            base=torch.randn(4,12,requires_grad=True)
            forcing=torch.randn(4,12,3); attrs=torch.randn(4,2)
            z=torch.randn(4,12,8,requires_grad=True) if latent_dim else None
            original=base.detach().clone()
            pred=head(base,forcing,attrs,z)
            torch.testing.assert_close(pred,base,rtol=0,atol=0)
            opt=torch.optim.SGD(head.parameters(),lr=.01)
            (pred-original-1).square().mean().backward()
            self.assertIsNone(base.grad)
            if z is not None: self.assertIsNone(z.grad)
            self.assertGreater(head.output.weight.grad.abs().sum().item(),0)
            opt.step(); head.eval()
            self.assertFalse(torch.equal(head(base,forcing,attrs,z),original))
            torch.testing.assert_close(base,original,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
