import torch
from torch.nn import functional as F
eps = 1e-4
class FirstOrderUPGDv1AntiCorrNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv1AntiCorrNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.tanh(F.normalize(avg_utility / bias_correction, dim=-1) / group["temp"])
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(p.grad.data + noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(p.grad.data + noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv2AntiCorrNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv2AntiCorrNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.tanh(F.normalize(avg_utility / bias_correction, dim=-1) / group["temp"])
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_((p.grad.data + noise) * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_((p.grad.data + noise) * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv1NormalNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv1NormalNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.tanh(F.normalize(avg_utility / bias_correction, dim=-1) / group["temp"])
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(p.grad.data + noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(p.grad.data + noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv2NormalNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv2NormalNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.tanh(F.normalize(avg_utility / bias_correction, dim=-1) / group["temp"])
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_((p.grad.data + noise) * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_((p.grad.data + noise) * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv1AntiCorrMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv1AntiCorrMax, self).__init__(params, defaults)

    def step(self, loss):
        gate_utility = None
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    current_util_max = avg_utility.max()
                    if current_util_max > global_max_util:
                        global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if 'gate' in name:
                    gate_utility = torch.tanh_((state["avg_utility"] / bias_correction) / group["temp"] / global_max_util) / torch.tanh_(torch.tensor(1.0))
                    continue
                if gate_utility is not None:
                    if group["noise_damping"]:
                        new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(p.grad.data + noise * (1-gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(p.grad.data + noise * (1-gate_utility.T), alpha=-group["lr"])
                        gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv2AntiCorrMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv2AntiCorrMax, self).__init__(params, defaults)

    def step(self, loss):
        gate_utility = None
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    current_util_max = avg_utility.max()
                    if current_util_max > global_max_util:
                        global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if 'gate' in name:
                    gate_utility = torch.tanh_((state["avg_utility"] / bias_correction) / group["temp"] / global_max_util) / torch.tanh_(torch.tensor(1.0))
                    continue
                if gate_utility is not None:
                    if group["noise_damping"]:
                        new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_((p.grad.data + noise) * (1-gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_((p.grad.data + noise) * (1-gate_utility.T), alpha=-group["lr"])
                        gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv1NormalMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv1NormalMax, self).__init__(params, defaults)

    def step(self, loss):
        gate_utility = None
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    current_util_max = avg_utility.max()
                    if current_util_max > global_max_util:
                        global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if 'gate' in name:
                    gate_utility = torch.tanh_((state["avg_utility"] / bias_correction) / group["temp"] / global_max_util) / torch.tanh_(torch.tensor(1.0))
                    continue
                if gate_utility is not None:
                    if group["noise_damping"]:
                        noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(p.grad.data + noise * (1-gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(p.grad.data + noise * (1-gate_utility.T), alpha=-group["lr"])
                        gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderUPGDv2NormalMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderUPGDv2NormalMax, self).__init__(params, defaults)

    def step(self, loss):
        gate_utility = None
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    current_util_max = avg_utility.max()
                    if current_util_max > global_max_util:
                        global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if 'gate' in name:
                    gate_utility = torch.tanh_((state["avg_utility"] / bias_correction) / group["temp"] / global_max_util) / torch.tanh_(torch.tensor(1.0))
                    continue
                if gate_utility is not None:
                    if group["noise_damping"]:
                        noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_((p.grad.data + noise) * (1-gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_((p.grad.data + noise) * (1-gate_utility.T), alpha=-group["lr"])
                        gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])