import numpy as np

from sr_harness.analysis import plots


def _record(curve):
    return dict(harness="h", task_id="t", seed=0, run_id="r", curve=curve, target_var=4.0,
                baseline_mse={"train": 4.0, "train_dist": 5.0, "ood": 9.0},
                scores={"train_dist": None, "ood": None})


def test_run_without_evaluations_gets_the_training_mean_baseline():
    # review bug: a run with no evaluation was scored at the cap (MSE = 1e300) instead of the baseline
    r = _record(curve=[])
    df = plots.step_table([r], "ood_mse", max_step=3)
    assert np.allclose(df["value"], np.log10(9.0))
    assert np.allclose(plots.step_table([r], "ood_r2", max_step=3)["value"], 0.0)
    assert np.isclose(plots.final_log10_mse(r, "ood"), np.log10(9.0))
    assert np.isclose(plots.final_log10_mse(r, "train"), np.log10(4.0))
    tok = plots.tokens_table([r], "ood_mse", grid=[1000, 5000])
    assert len(tok) == 2 and np.allclose(tok["value"], np.log10(9.0))     # the run is not dropped


def test_steps_before_the_first_evaluation_use_the_baseline():
    r = _record(curve=[dict(index=2, tokens=3000, ood_mse=0.01, ood_r2=0.9)])
    df = plots.step_table([r], "ood_mse", max_step=3)
    assert np.allclose(df["value"], [np.log10(9.0), -2.0, -2.0])          # step 1: baseline; then 0.01
    tok = plots.tokens_table([r], "ood_mse", grid=[1000, 5000])
    assert np.allclose(tok["value"], [np.log10(9.0), -2.0])
