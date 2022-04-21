import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


def l2normalize(v: Tensor, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module: nn.Module, name='weight', power_iterations=1):
        """
        Reference:
        SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/pdf/1802.05957.pdf
        """
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        """
        TODO: Implement Spectral Normalization
        Hint: 1: Use getattr to first extract u, v, w.
              2: Apply power iteration.
              3: Calculate w with the spectral norm.
              4: Use setattr to update w in the module.
        """
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        w_l = w.view(u.size(0), -1)

        # Apply power iteration
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(torch.t(w_l), u))
            u = l2normalize(torch.matmul(w_l, v))

        # Calculate w with spectral norm
        sigma = torch.matmul(torch.t(u), torch.matmul(w_l, v))
        w = w / sigma

        # Update w
        setattr(self.module, self.name, w)


    def _make_params(self):
        """
        No need to change. Initialize parameters.
        v: Initialize v with a random vector (sampled from isotropic distrition).
        u: Initialize u with a random vector (sampled from isotropic distrition).
        w: Weight of the current layer.
        """
        w: Tensor = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        # print(height, width)
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        """
        No need to change. Update weights using spectral normalization.
        """
        self._update_u_v()
        return self.module.forward(*args)

