"""Analytical checks; no held-out hardware response is used in test design."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_lab import survival, model_basis, grid_fit, predict


def test_first_order_exact_step():
    t = np.arange(0., 101., 1.)
    q = np.ones_like(t)
    fit = dict(tau_h_s=25., tau_sensor_s=None, ambient_c=20., gain_full_scale_c=40.)
    actual = predict(t, q, 20., fit)
    expected = 20. + 40. * (1. - np.exp(-t / 25.))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_equal_pole_sensor_lag_limit():
    t = np.arange(0., 101., 1.)
    expected = np.exp(-t / 20.) * (1. + t / 20.)
    np.testing.assert_allclose(survival(t, [20.], [20.])[0], expected, atol=1e-12)


def test_future_commands_cannot_change_prior_predictions():
    t = np.arange(0., 101., 1.)
    q1 = np.zeros_like(t)
    q1[10:40] = .5
    q2 = q1.copy()
    q2[70:] = .8
    for sensor in (None, 7.):
        fit = dict(tau_h_s=25., tau_sensor_s=sensor, ambient_c=20., gain_full_scale_c=40.)
        p1, p2 = predict(t, q1, 20., fit), predict(t, q2, 20., fit)
        np.testing.assert_array_equal(p1[:71], p2[:71])


def test_calibration_only_grid_recovers_known_linear_parameters():
    t = np.arange(0., 200., 1.)
    q = np.zeros_like(t)
    q[10:70], q[110:170] = .5, .9
    fit = dict(tau_h_s=45., tau_sensor_s=None, ambient_c=21., gain_full_scale_c=80.)
    measured = predict(t, q, 21., fit)
    ranked = grid_fit(t, q, measured, 21., np.array([20., 45., 90.]), None, (16., 26.), (1., 200.))
    assert ranked[0]["tau_h_s"] == 45.
    np.testing.assert_allclose([ranked[0]["ambient_c"], ranked[0]["gain_full_scale_c"]], [21., 80.], atol=1e-9)
    assert ranked[0]["calibration_sse"] < 1e-15


def test_sensor_lag_initial_conditions_and_long_run_equilibrium():
    t = np.array([0., 10000.])
    fit = dict(tau_h_s=100., tau_sensor_s=15., ambient_c=22., gain_full_scale_c=80.)
    actual = predict(t, np.ones_like(t) * .5, 23., fit)
    np.testing.assert_allclose(actual, [23., 62.], atol=1e-12)
