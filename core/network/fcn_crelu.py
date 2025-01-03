import torch.nn as nn
from .gate import GateLayer
import torch, math
from functools import partial
import collections

class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.cat((nn.functional.relu(x, inplace=self.inplace), 
                          nn.functional.relu(-x, inplace=self.inplace)), dim=1)

class FullyConnectedCReLUGates(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FullyConnectedCReLUGates, self).__init__()
        self.name = "fully_connected_crelu_gates"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", CReLU())
        self.add_module("gate_1", GateLayer(n_hidden_units*2))
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units*2, out_features=n_hidden_units // 2))
        self.add_module("act_2", CReLU())
        self.add_module("gate_2", GateLayer(n_hidden_units))
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
    def zero_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def const_init(self, const=0.1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, const)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, const)

# class FullyConnectedCReLUWithHooks(nn.Sequential):
#     def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
#         super().__init__()
#         self.name = "fully_connected_crelu_with_hooks"
#         self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
#         self.add_module("act_1", CReLU())
#         self.add_module("linear_2", nn.Linear(in_features=n_hidden_units*2, out_features=n_hidden_units // 2))
#         self.add_module("act_2", CReLU())
#         self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
#         self.activations = collections.defaultdict(list)
#         self.n_units = n_hidden_units + n_hidden_units // 2

#         for name, layer in self.named_modules():
#             if isinstance(layer, nn.Linear):
#                 layer.reset_parameters()
#             if isinstance(layer, CReLU):
#                 layer.register_forward_hook(partial(self.activation_hook, name))

#     def __str__(self):
#         return self.name

#     def activation_hook(self, name, module, inp, out):
#         self.activations[name] = torch.sum(out == 0.0).item()

class FullyConnectedCReLU(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FullyConnectedCReLU, self).__init__()
        self.name = "fully_connected_crelu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", CReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units*2, out_features=n_hidden_units // 2))
        self.add_module("act_2", CReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
    def zero_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def const_init(self, const=0.1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, const)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, const)

class SmallFullyConnectedCReLUGates(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=50):
        super(SmallFullyConnectedCReLUGates, self).__init__()
        self.name = "small_fully_connected_crelu_gates"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", CReLU())
        self.add_module("gate_1", GateLayer(n_hidden_units*2))
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units*2, out_features=n_hidden_units))
        self.add_module("act_2", CReLU())
        self.add_module("gate_2", GateLayer(n_hidden_units*2))
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units*2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name

class SmallFullyConnectedCReLU(nn.Sequential):
    def __init__(self, n_obs=4, n_outputs=1, n_hidden_units=50):
        super(SmallFullyConnectedCReLU, self).__init__()
        self.name = "small_fully_connected_crelu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", CReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units*2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name

class ConvolutionalNetworkCReLU(nn.Sequential):
    def __init__(self, n_obs=4, n_outputs=10):
        super(ConvolutionalNetworkCReLU, self).__init__()
        self.name = "convolutional_network_crelu"
        self.add_module("conv_1", nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5))
        self.add_module("pool_1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("conv_2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5))
        self.add_module("pool_2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("flatten", nn.Flatten())
        self.add_module("linear_1", nn.Linear(in_features=16 * 5 * 5, out_features=120))
        self.add_module("act_1", CReLU())
        self.add_module("linear_2", nn.Linear(in_features=120*2, out_features=84))
        self.add_module("act_2", CReLU())
        self.add_module("linear_3", nn.Linear(in_features=84*2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name


# class ConvolutionalNetworkCReLUWithHooks(nn.Sequential):
#     def __init__(self, n_obs=4, n_outputs=10):
#         super(ConvolutionalNetworkCReLUWithHooks, self).__init__()
#         self.name = "convolutional_network_crelu_with_hooks"
#         self.add_module("conv_1", nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5))
#         self.add_module("pool_1", nn.MaxPool2d(kernel_size=2, stride=2))
#         self.add_module("conv_2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5))
#         self.add_module("pool_2", nn.MaxPool2d(kernel_size=2, stride=2))
#         self.add_module("flatten", nn.Flatten())
#         self.add_module("linear_1", nn.Linear(in_features=16 * 5 * 5, out_features=120))
#         self.add_module("act_1", CReLU())
#         self.add_module("linear_2", nn.Linear(in_features=120, out_features=84))
#         self.add_module("act_2", CReLU())
#         self.add_module("linear_3", nn.Linear(in_features=84, out_features=n_outputs))
#         self.activations = collections.defaultdict(list)
#         self.n_units = 120 + 84
#         for name, layer in self.named_modules():
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#                 layer.reset_parameters()
#             if isinstance(layer, CReLU):
#                 layer.register_forward_hook(partial(self.activation_hook, name))

#     def __str__(self):
#         return self.name

#     def activation_hook(self, name, module, inp, out):
#         self.activations[name] = torch.sum(out == 0.0).item()

#     def __str__(self):
#         return self.name

if __name__ == "__main__":
    # example of dummy input to network
    net = ConvolutionalNetworkCReLU()
    inputs = torch.randn(42, 3, 32, 32)
    output = net(inputs)
    
    print(output.shape)
