import pytest
import torch


def test_forward_shape_and_finite():
    from kernel_generation.tasks.armt_associate import task

    m = task.Model(*task.get_init_inputs())
    (x,) = task.get_inputs()
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item() is True

    # Regression check: fp16 + cuda should not crash due to W_mem/z staying float32.
    if torch.cuda.is_available():
        m = task.Model(*task.get_init_inputs()).cuda().half()
        (x,) = task.get_inputs()
        x = x.cuda().half()
        with torch.no_grad():
            y = m(x)
        assert y.dtype == torch.float16
        assert torch.isfinite(y).all().item() is True


def test_determinism_with_fixed_seed():
    from kernel_generation.tasks.armt_associate import task

    torch.manual_seed(0)
    m1 = task.Model(*task.get_init_inputs())
    (x1,) = task.get_inputs()
    with torch.no_grad():
        y1 = m1(x1)

    torch.manual_seed(0)
    m2 = task.Model(*task.get_init_inputs())
    (x2,) = task.get_inputs()
    with torch.no_grad():
        y2 = m2(x2)

    assert torch.allclose(x1, x2)
    assert torch.allclose(y1, y2)


def test_use_denom_false_branch(monkeypatch: pytest.MonkeyPatch):
    from kernel_generation.tasks.armt_associate import task

    # Model reads module-level `use_denom` at init time.
    monkeypatch.setattr(task, "use_denom", False)
    m = task.Model(*task.get_init_inputs())
    (x,) = task.get_inputs()
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item() is True


