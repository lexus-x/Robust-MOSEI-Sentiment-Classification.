from __future__ import annotations

import warnings

from multimod.config import make_experiment_config
from multimod.data import build_dataloaders, describe_dataset
from multimod.models import InputDims, build_model

from .helpers import write_toy_mosei_pickle


def test_model_forward_shapes(tmp_path):
    data_path = write_toy_mosei_pickle(tmp_path / "toy_mosei.pkl")
    stats = describe_dataset(data_path, split="train")
    dims = InputDims(text=stats.text_dim, audio=stats.audio_dim, vision=stats.vision_dim)
    batch = next(iter(build_dataloaders(data_path, batch_size=2, num_workers=0)["train"]))

    for experiment_name in (
        "text_only",
        "early_fusion",
        "xmodal_transformer",
        "xmodal_transformer_robust",
    ):
        config = make_experiment_config(experiment_name, data_path=str(data_path))
        model = build_model(config.model, dims)
        logits, aux = model(batch["text"], batch["audio"], batch["vision"], batch["mask"])
        assert logits.shape == (2, 3)
        if experiment_name == "xmodal_transformer_robust":
            assert "gates" in aux
            assert aux["gates"].shape == (2, 3)


def test_transformer_model_build_does_not_emit_nested_tensor_warning(tmp_path):
    data_path = write_toy_mosei_pickle(tmp_path / "toy_mosei.pkl")
    stats = describe_dataset(data_path, split="train")
    dims = InputDims(text=stats.text_dim, audio=stats.audio_dim, vision=stats.vision_dim)
    config = make_experiment_config("xmodal_transformer", data_path=str(data_path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_model(config.model, dims)

    assert not any("enable_nested_tensor is True" in str(warning.message) for warning in caught)
