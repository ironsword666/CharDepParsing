# -*- encoding: utf-8 -*-

import torch
from supar.utils.common import MIN


class Semiring(object):

    pass


class LogSemiring(Semiring):

    zero = MIN
    one = 0

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)


class MaxSemiring(LogSemiring):

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        return torch.max(x, dim=dim, keepdim=keepdim)[0]
