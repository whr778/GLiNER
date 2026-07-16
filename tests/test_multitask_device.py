"""Device resolution for the multitask pipelines (GLiNERClassifier,
GLiNERRelationExtractor, GLiNERSummarizer, and their shared base class).

Covers cuda/mps/cpu selection without requiring any of those backends to
actually be present on the machine running the tests -- availability checks
are monkeypatched so all branches are exercised deterministically.
"""

import warnings

import pytest

from gliner.multitask.base import resolve_device


def _patch_availability(monkeypatch, *, cuda: bool, mps: bool):
    monkeypatch.setattr("gliner.multitask.base.torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr("gliner.multitask.base.is_torch_mps_available", lambda: mps)


class TestResolveDeviceAuto:
    def test_prefers_cuda_when_available(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=True, mps=True)
        assert resolve_device("auto") == "cuda:0"

    def test_falls_back_to_mps_when_no_cuda(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=False, mps=True)
        assert resolve_device("auto") == "mps"

    def test_falls_back_to_cpu_when_nothing_available(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=False, mps=False)
        assert resolve_device("auto") == "cpu"


class TestResolveDeviceExplicit:
    def test_explicit_cpu_is_unchanged(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=False, mps=False)
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda_passes_through_when_available(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=True, mps=False)
        assert resolve_device("cuda:0") == "cuda:0"

    def test_explicit_mps_passes_through_when_available(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=False, mps=True)
        assert resolve_device("mps") == "mps"

    def test_explicit_cuda_falls_back_to_cpu_with_warning_when_unavailable(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=False, mps=True)
        with pytest.warns(UserWarning, match="not available"):
            result = resolve_device("cuda:0")
        assert result == "cpu"

    def test_explicit_mps_falls_back_to_cpu_with_warning_when_unavailable(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=False, mps=False)
        with pytest.warns(UserWarning, match="not available"):
            result = resolve_device("mps")
        assert result == "cpu"

    def test_no_warning_when_explicit_device_is_available(self, monkeypatch):
        _patch_availability(monkeypatch, cuda=True, mps=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_device("cuda:0") == "cuda:0"
