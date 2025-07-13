import torch
from torch import Tensor


class Primitives:
    def __init__(self, device: str):
        self.device = device

    def torch_rand_uniform(self, a: float, b: float) -> Tensor:
        return torch.rand(1, device=self.device) * (b - a) + a

    def add_array(self, a: Tensor | float, b: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, device=self.device)
        return torch.add(a, b)

    def sub_array(self, a: Tensor | float, b: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, device=self.device)
        return torch.sub(a, b)

    def mul_array(self, a: Tensor | float, b: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, device=self.device)
        return torch.mul(a, b)

    def div_array(self, a: Tensor | float, b: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, device=self.device)
        # protect against division by zero
        b = torch.where(b == 0, torch.tensor(1e-10, device=self.device), b)
        return torch.div(a, b)

    def neg_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.neg(a)

    def square_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.pow(a, 2)

    def abs_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.abs(a)

    def exp_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.exp(a)

    def inv_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        # protect against division by zero
        a = torch.where(a == 0, torch.tensor(1e-10, device=self.device), a)
        return torch.reciprocal(a)

    def cos_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.cos(a)

    def sin_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.sin(a)

    def relu_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.relu(a)

    def kill_array(self, a: Tensor | float) -> Tensor:
        if not isinstance(a, Tensor):
            a = torch.tensor(a, device=self.device)
        return torch.zeros_like(a)

    def norm_array(self, a: Tensor | float, b: Tensor) -> Tensor:
        c = self.sub_array(a, b)
        return torch.linalg.norm(c, dim=-1)
