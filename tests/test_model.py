import numpy as np
import pytest

from model.plant import (ReservoirWindow, _per_channel_evaluation, _select_output_channels,
                         describe_evaluation, resolve_event_context, run_reservoir_plant)
from model.reservoir import EchoStateNetwork, ReservoirConfig
from model.visualize import plot_all

from extreme_event_agent.models import ClinicalEvent


def test_reservoir_config_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="spectral_radius"):
        ReservoirConfig(spectral_radius=0.0)
    with pytest.raises(ValueError, match="leak_rate"):
        ReservoirConfig(leak_rate=0.0)
    with pytest.raises(ValueError, match="n_reservoir"):
        ReservoirConfig(n_reservoir=2)


def test_echo_state_network_is_deterministic_and_shape_correct():
    config = ReservoirConfig(n_reservoir=40, seed=3)
    rng = np.random.default_rng(1)
    U = rng.normal(size=(200, 2))
    esn_a = EchoStateNetwork(n_inputs=2, n_outputs=1, config=config)
    esn_b = EchoStateNetwork(n_inputs=2, n_outputs=1, config=config)
    Xa, Xb = esn_a.run_states(U), esn_b.run_states(U)
    np.testing.assert_array_equal(Xa, Xb)
    assert Xa.shape == (200, 40)
    assert np.all(np.abs(Xa) <= 1.0)
    assert esn_a.achieved_spectral_radius == pytest.approx(config.spectral_radius, abs=1e-6)


def test_fit_readout_reduces_error_versus_untrained_output():
    config = ReservoirConfig(n_reservoir=60, washout=10, seed=5)
    rng = np.random.default_rng(2)
    T = 500
    U = np.sin(np.linspace(0, 40 * np.pi, T))[:, None]
    Y = np.stack([np.roll(U[:, 0], -1)], axis=1) + rng.normal(0, 1e-3, size=(T, 1))
    esn = EchoStateNetwork(n_inputs=1, n_outputs=1, config=config)
    X = esn.run_states(U)
    with pytest.raises(ValueError, match="not fitted"):
        esn.predict(X, U)
    rmse = esn.fit_readout(X, U, Y)
    predicted = esn.predict(X, U)
    naive_rmse = np.sqrt(np.mean((Y[config.washout:, 0] - U[config.washout:, 0]) ** 2))
    assert rmse[0] < naive_rmse
    assert predicted.shape == Y.shape


def test_select_output_channels_falls_back_to_variance_without_a_real_event():
    rng = np.random.default_rng(9)
    names = ["EEG A1", "EEG A2", "EEG A3", "EEG A4"]
    data = rng.normal(0, 1e-6, (4, 4000))
    data[2] *= 50  # one obviously higher-variance channel
    selected, method, process = _select_output_channels(data, names, sfreq=200.0, baseline_seconds=15.0,
                                                        analysis_seconds=5.0, max_output_channels=2)
    assert method == "highest_variance_fallback"
    assert "EEG A3" in selected
    assert len(selected) <= 2


def test_run_reservoir_plant_flags_injected_burst_as_extreme_event(tmp_path):
    sfreq = 200.0
    T = int(90 * sfreq)
    rng = np.random.default_rng(11)
    n_out = 3
    output = rng.normal(0, 1e-5, (T, n_out))
    times = (np.arange(T) / sfreq) - 60.0  # event at index for t=60s into a 90s window -> t=0 here
    burst_mask = (times >= 0) & (times < 3.0)
    output[burst_mask] += 2e-3 * np.sin(2 * np.pi * 6 * times[burst_mask])[:, None]
    mkr = (np.mod(np.arange(T), int(sfreq)) < 2).astype(float)[:, None]

    window = ReservoirWindow(
        times_seconds=times, sfreq=sfreq, input_names=["MKR1+"], input_data=mkr,
        output_names=[f"C{i}" for i in range(n_out)], output_data=output,
        event=ClinicalEvent(60.0, duration_seconds=3.0), baseline_seconds=60.0, analysis_seconds=30.0,
        channel_selection_method="user_specified", process=None)

    evaluation = run_reservoir_plant(window, config=ReservoirConfig(n_reservoir=80, washout=50, seed=4))
    assert evaluation.hidden_states.shape == (T, 80)
    assert evaluation.predicted_output.shape == (T, n_out)
    assert evaluation.detected
    # Centered smoothing (see _moving_average) can pull the detected onset a
    # couple of samples earlier than the true rising edge — allow for that.
    assert -0.1 <= evaluation.onset_time_seconds <= 3.5
    assert evaluation.peak_score >= evaluation.threshold

    text = describe_evaluation(evaluation)
    assert "DETECTED" in text

    figures = plot_all(evaluation, tmp_path, "unit_test")
    assert set(figures) == {"architecture", "connectivity", "spectrum", "hidden_state",
                            "output_prediction", "residual_heatmap", "extreme_event_score"}
    for path in figures.values():
        assert path.stat().st_size > 500


def test_resolve_event_context_prefers_explicit_time(tmp_path):
    context = resolve_event_context(tmp_path / "does-not-need-to-exist.edf", event_time=42.0)
    assert context.time_seconds == 42.0


def test_balanced_selection_ignores_recruitment(monkeypatch):
    # Regression test for the defect where the reservoir's channel choice
    # came from the same recruitment analysis it is meant to independently
    # check: "balanced" must give the identical answer whether or not
    # analyse_brain_process even works, since it must never call it.
    rng = np.random.default_rng(9)
    names = ["EEG PM3", "EEG PM4", "EEG CC8", "EEG CC9",
            "EEG PM'3", "EEG PM'4", "EEG CC'8", "EEG CC'9"]
    data = rng.normal(0, 1.0, (8, 4000))
    data[0] *= 5   # EEG PM3 (right): obviously highest right-side baseline variance
    data[4] *= 5   # EEG PM'3 (left): obviously highest left-side baseline variance

    selected_before, method_before, process_before = _select_output_channels(
        data, names, sfreq=200.0, baseline_seconds=15.0, analysis_seconds=5.0, max_output_channels=4,
        channel_selection="balanced")
    assert method_before == "balanced_hemisphere_variance"
    assert process_before is None  # "balanced" never runs analyse_brain_process at all

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("channel_selection='balanced' must never call analyse_brain_process")
    monkeypatch.setattr("model.plant.analyse_brain_process", _must_not_be_called)

    selected_after, method_after, process_after = _select_output_channels(
        data, names, sfreq=200.0, baseline_seconds=15.0, analysis_seconds=5.0, max_output_channels=4,
        channel_selection="balanced")
    assert selected_after == selected_before
    assert method_after == "balanced_hemisphere_variance"

    right_selected = [name for name in selected_before if name in
                      ("EEG PM3", "EEG PM4", "EEG CC8", "EEG CC9")]
    left_selected = [name for name in selected_before if name in
                     ("EEG PM'3", "EEG PM'4", "EEG CC'8", "EEG CC'9")]
    assert len(right_selected) == 2 and len(left_selected) == 2
    assert "EEG PM3" in right_selected and "EEG PM'3" in left_selected


def test_per_channel_score_normalized_independently():
    # A channel whose residual -- baseline noise AND burst alike -- runs 10x
    # the amplitude of another, but is otherwise the identical relative
    # shape, must not get a proportionally larger z-score: each channel is
    # normalized against its own baseline, not a shared scale.
    sfreq = 100.0
    T = 3000
    times = (np.arange(T) / sfreq) - 10.0  # 10s baseline, then post-event
    rng = np.random.default_rng(3)
    burst_mask = times >= 0
    shape = np.zeros(T)
    shape[burst_mask] = np.sin(2 * np.pi * 3 * times[burst_mask])
    base_with_burst = rng.normal(0, 1e-3, T) + 1.0 * shape
    residual = np.stack([base_with_burst * 10.0, base_with_burst * 1.0], axis=1)  # "big" = 10x "small"

    washout = 20
    baseline_scored = np.flatnonzero(times < 0.0)[washout:]
    per_channel_score, onset, peak, peak_time = _per_channel_evaluation(
        residual, times, baseline_scored, washout, threshold=6.0, smoothing_samples=5,
        output_names=["big", "small"])

    assert per_channel_score.shape == (T, 2)
    assert peak["big"] == pytest.approx(peak["small"], rel=0.2)
    assert onset["big"] is not None and onset["small"] is not None
    assert abs(onset["big"] - onset["small"]) < 1.0
