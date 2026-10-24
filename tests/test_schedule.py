from toksw.schedule import lr_at, p_new


def test_p_new_linear_then_one():
    assert p_new(0, 100, 0.5) == 0.0
    assert p_new(25, 100, 0.5) == 0.5
    assert p_new(50, 100, 0.5) == 1.0
    assert p_new(99, 100, 0.5) == 1.0


def test_abrupt_is_rho_zero():
    assert p_new(0, 100, 0.0) == 1.0


def test_lr_warmup_and_floor():
    assert lr_at(0, 100, 1.0, 10) == 0.1
    assert abs(lr_at(10, 100, 1.0, 10) - 1.0) < 1e-9
    assert abs(lr_at(100, 100, 1.0, 10, 0.1) - 0.1) < 1e-9
