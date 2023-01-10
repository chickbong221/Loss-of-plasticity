import torch

class FeatureNvidiaUtilitySO:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'nvidia_so_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for name, p in  self.network.named_parameters():
                if 'gate' in name:
                    fo_utility = (-p.data * p.grad + 0.5 * (p.data ** 2) * p.hesscale) ** 2
                    fo_utility_net.append(fo_utility)
        return fo_utility_net  