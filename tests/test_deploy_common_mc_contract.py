from __future__ import annotations

import torch
import torch.nn as nn

from fall_detection.deploy.common import predict_mu_sigma


class _BNDropoutTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(p=0.5)
        self.head = nn.Linear(4, 1)
        self.bn_training_seen: list[bool] = []
        self.drop_training_seen: list[bool] = []
        self.bn.register_forward_hook(self._record_bn_state)
        self.drop.register_forward_hook(self._record_drop_state)

    def _record_bn_state(self, module, _inputs, _output) -> None:
        self.bn_training_seen.append(bool(module.training))

    def _record_drop_state(self, module, _inputs, _output) -> None:
        self.drop_training_seen.append(bool(module.training))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = self.drop(x)
        x = x.mean(dim=-1)
        return self.head(x).squeeze(-1)


def test_predict_mu_sigma_keeps_batchnorm_in_eval_but_enables_dropout() -> None:
    model = _BNDropoutTCN().eval()
    x = torch.ones((8, 4), dtype=torch.float32).numpy()

    mu, sigma = predict_mu_sigma(
        model=model,
        arch="tcn",
        X=x,
        device=torch.device("cpu"),
        two_stream=False,
        M=6,
    )

    assert 0.0 <= mu <= 1.0
    assert sigma >= 0.0
    assert model.training is False
    assert model.bn.training is False
    assert model.drop.training is False
    assert model.bn_training_seen
    assert model.drop_training_seen
    assert all(state is False for state in model.bn_training_seen)
    assert any(state is True for state in model.drop_training_seen)
