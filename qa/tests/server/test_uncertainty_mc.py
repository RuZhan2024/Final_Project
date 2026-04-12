from __future__ import annotations

import importlib
import sys
import types

import numpy as np


class _Tensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    def reshape(self, *shape):
        return _Tensor(self.arr.reshape(*shape))

    def squeeze(self, axis=None):
        return _Tensor(np.squeeze(self.arr, axis=axis))

    def unsqueeze(self, axis):
        return _Tensor(np.expand_dims(self.arr, axis))

    def clone(self):
        return _Tensor(self.arr.copy())

    def new_empty(self, shape):
        return _Tensor(np.empty(shape, dtype=self.arr.dtype))

    def mean(self, dim=0):
        return _Tensor(np.mean(self.arr, axis=dim))

    def std(self, dim=0, unbiased=False):
        ddof = 1 if unbiased else 0
        return _Tensor(np.std(self.arr, axis=dim, ddof=ddof))

    def __getitem__(self, idx):
        return _Tensor(self.arr[idx])

    def __setitem__(self, idx, value):
        self.arr[idx] = value.arr if isinstance(value, _Tensor) else value

    def __add__(self, other):
        other = other.arr if isinstance(other, _Tensor) else other
        return _Tensor(self.arr + other)

    def __sub__(self, other):
        other = other.arr if isinstance(other, _Tensor) else other
        return _Tensor(self.arr - other)

    def __mul__(self, other):
        other = other.arr if isinstance(other, _Tensor) else other
        return _Tensor(self.arr * other)

    def __truediv__(self, other):
        other = other.arr if isinstance(other, _Tensor) else other
        return _Tensor(self.arr / other)

    def __le__(self, other):
        other = other.arr if isinstance(other, _Tensor) else other
        return _Tensor(self.arr <= other)


class _Dropout:
    def __init__(self):
        self.training = False

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class _Model:
    def __init__(self):
        self.training = False
        self.dp = _Dropout()

    def modules(self):
        return [self, self.dp]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


def _install_fake_torch(monkeypatch):
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_torch.inference_mode = lambda: _Ctx()
    fake_torch.zeros_like = lambda x: _Tensor(np.zeros_like(x.arr))
    fake_torch.sqrt = lambda x: _Tensor(np.sqrt(x.arr))
    fake_torch.clamp = lambda x, min=0.0: _Tensor(np.clip(x.arr, min, None))
    fake_torch.all = lambda x: bool(np.all(x.arr))
    fake_nn.Dropout = _Dropout
    fake_nn.Dropout2d = _Dropout
    fake_nn.Dropout3d = _Dropout
    fake_nn.AlphaDropout = _Dropout
    fake_torch.nn = fake_nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)


def _import_uncertainty(monkeypatch):
    _install_fake_torch(monkeypatch)
    sys.modules.pop("core.uncertainty", None)
    return importlib.import_module("core.uncertainty")


def test_mc_predict_returns_n_used_with_early_stop(monkeypatch):
    unc = _import_uncertainty(monkeypatch)
    model = _Model()
    calls = {"n": 0}

    def _fwd():
        calls["n"] += 1
        return _Tensor([0.8])  # constant samples => sigma=0 => early stop

    mu, sigma, n_used = unc.mc_predict_mu_sigma(
        model,
        forward_fn=_fwd,
        M=40,
        max_sigma_for_early_stop=0.05,
        min_M_for_early_stop=4,
        return_n_used=True,
    )

    assert mu.shape == sigma.shape == (1,)
    assert isinstance(n_used, int)
    assert n_used == calls["n"]
    assert 4 <= n_used < 40


def test_mc_predict_return_samples_honors_n_used(monkeypatch):
    unc = _import_uncertainty(monkeypatch)
    model = _Model()
    calls = {"n": 0}

    def _fwd():
        calls["n"] += 1
        return _Tensor([1.0])  # constant samples => sigma=0 => early stop

    mu, sigma, samples, n_used = unc.mc_predict_mu_sigma(
        model,
        forward_fn=_fwd,
        M=30,
        max_sigma_for_early_stop=0.05,
        min_M_for_early_stop=3,
        return_samples=True,
        return_n_used=True,
    )

    assert samples.ndim == 2
    assert samples.shape[1] == 1
    assert samples.shape[0] == n_used == calls["n"]
    assert 3 <= n_used < 30
    assert mu.shape == sigma.shape == (1,)


def test_mc_predict_supports_se_based_early_stop(monkeypatch):
    unc = _import_uncertainty(monkeypatch)
    model = _Model()
    calls = {"n": 0}

    def _fwd():
        calls["n"] += 1
        return _Tensor([0.3])  # constant samples => standard error quickly -> 0

    _mu, _sigma, n_used = unc.mc_predict_mu_sigma(
        model,
        forward_fn=_fwd,
        M=25,
        max_se_for_early_stop=0.02,
        min_M_for_early_stop=5,
        return_n_used=True,
    )

    assert n_used == calls["n"]
    assert 5 <= n_used < 25


def test_mc_predict_return_samples_supports_se_based_early_stop(monkeypatch):
    unc = _import_uncertainty(monkeypatch)
    model = _Model()
    calls = {"n": 0}

    def _fwd():
        calls["n"] += 1
        return _Tensor([0.3])  # constant samples => SE quickly reaches 0

    _mu, _sigma, samples, n_used = unc.mc_predict_mu_sigma(
        model,
        forward_fn=_fwd,
        M=30,
        max_se_for_early_stop=0.01,
        min_M_for_early_stop=4,
        return_samples=True,
        return_n_used=True,
    )

    assert samples.shape[0] == n_used == calls["n"]
    assert 4 <= n_used < 30
