import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()


x = torch.rand(4, 3)
print(x)
x.requires_grad_()
x_linear = nn.Linear(3, 1)

y = x_linear(x)
print(y)
grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))
print(grad)

print(help(torch.float32))

