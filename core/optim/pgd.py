import torch


class PGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, names=names)
        super(PGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                perturbed_gradient = p.grad + torch.randn_like(p.grad) * group["sigma"]
                p.data.add_(perturbed_gradient, alpha=-group["lr"])
