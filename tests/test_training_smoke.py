from __future__ import annotations

from multimod.config import make_experiment_config
from multimod.training import run_experiment

from .helpers import write_toy_mosei_pickle


def test_training_smoke_run(tmp_path):
    data_path = write_toy_mosei_pickle(tmp_path / "toy_mosei.pkl")
    output_dir = tmp_path / "outputs"
    config = make_experiment_config(
        experiment_name="xmodal_transformer_robust",
        data_path=str(data_path),
        output_dir=str(output_dir),
    )
    config.data.batch_size = 2
    config.training.max_epochs = 2
    config.training.patience = 1
    result = run_experiment(config=config, seed=13, device_name="cpu")

    assert "summary" in result
    assert "clean_weighted_f1" in result["summary"]
    assert "avg_perturbed_weighted_f1" in result["summary"]


def test_eidmsa_retry_smoke_run(tmp_path):
    data_path = write_toy_mosei_pickle(tmp_path / "toy_mosei.pkl")
    output_dir = tmp_path / "outputs"
    config = make_experiment_config(
        experiment_name="eidmsa_realistic_retry",
        data_path=str(data_path),
        output_dir=str(output_dir),
    )
    config.data.batch_size = 2
    config.training.max_epochs = 2
    config.training.patience = 1
    config.model.evidential_warmup_epochs = 1
    result = run_experiment(config=config, seed=13, device_name="cpu")

    assert "summary" in result
    assert "clean_weighted_f1" in result["summary"]
    assert "avg_perturbed_weighted_f1" in result["summary"]
