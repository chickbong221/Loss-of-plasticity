from core.network.fcn_sigmoid import FullyConnectedSigmoid
from core.learner.sgd import SGDLearner

# Khởi tạo mạng và learner
network = FullyConnectedSigmoid()
learner = SGDLearner(network)

# Thử truyền một batch dữ liệu
import torch
input = torch.randn(32, 10)  # 32 mẫu, 10 đầu vào
output = network(input)
print(output)
