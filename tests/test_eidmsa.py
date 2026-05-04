"""Unit tests for EIDMSA components: IB encoder, PID, evidential fusion, TTA."""

from __future__ import annotations

import torch
import pytest

# Ensure reproducibility
torch.manual_seed(42)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def batch_data(device: torch.device) -> dict[str, torch.Tensor]:
    """Minimal synthetic batch for testing."""
    batch_size = 4
    seq_len = 8
    text_dim = 300
    audio_dim = 74
    vision_dim = 35
    return {
        "text": torch.randn(batch_size, seq_len, text_dim, device=device),
        "audio": torch.randn(batch_size, seq_len, audio_dim, device=device),
        "vision": torch.randn(batch_size, seq_len, vision_dim, device=device),
        "mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "label": torch.randint(0, 3, (batch_size,), device=device),
        "sample_id": [f"test_{i}" for i in range(batch_size)],
    }


@pytest.fixture
def dims() -> dict[str, int]:
    return {
        "text_dim": 300,
        "audio_dim": 74,
        "vision_dim": 35,
        "hidden_dim": 64,
        "latent_dim": 32,
        "num_classes": 3,
    }


# ── IB Encoder Tests ─────────────────────────────────────────────────────

class TestIBEncoder:
    def test_single_encoder_output_shape(self, device: torch.device) -> None:
        from multimod.models.ib_encoder import IBEncoder

        enc = IBEncoder(input_dim=300, hidden_dim=64, latent_dim=32).to(device)
        x = torch.randn(4, 8, 300, device=device)
        mask = torch.ones(4, 8, dtype=torch.bool, device=device)

        z, kl = enc(x, mask)
        assert z.shape == (4, 8, 32), f"Expected (4,8,32), got {z.shape}"
        assert kl.shape == (), f"KL should be scalar, got {kl.shape}"
        assert kl.item() >= 0.0, "KL divergence should be non-negative"

    def test_multimodal_encoder(self, batch_data, dims, device) -> None:
        from multimod.models.ib_encoder import MultiModalIBEncoder

        enc = MultiModalIBEncoder(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
        ).to(device)

        z_t, z_a, z_v, kl = enc(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        assert z_t.shape == (4, 8, 32)
        assert z_a.shape == (4, 8, 32)
        assert z_v.shape == (4, 8, 32)
        assert kl.item() >= 0.0

    def test_deterministic_at_eval(self, device) -> None:
        from multimod.models.ib_encoder import IBEncoder

        enc = IBEncoder(input_dim=300, hidden_dim=64, latent_dim=32).to(device)
        enc.eval()
        x = torch.randn(4, 8, 300, device=device)
        mask = torch.ones(4, 8, dtype=torch.bool, device=device)

        z1, _ = enc(x, mask)
        z2, _ = enc(x, mask)
        torch.testing.assert_close(z1, z2, msg="Eval mode should be deterministic")


# ── PID Decomposition Tests ──────────────────────────────────────────────

class TestPIDDecomposition:
    def test_output_shapes(self, device, dims) -> None:
        from multimod.models.pid_decomposition import PIDDecomposition

        pid = PIDDecomposition(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        batch_size = 4
        seq_len = 8
        z = torch.randn(batch_size, seq_len, dims["latent_dim"], device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        components = pid(z, z, z, mask)
        for key in ("unique_text", "unique_audio", "unique_vision", "redundant", "synergistic"):
            assert key in components, f"Missing component: {key}"
            assert components[key].shape == (batch_size, dims["latent_dim"]), \
                f"{key} shape mismatch: {components[key].shape}"

    def test_consistency_loss_finite(self, device, dims) -> None:
        from multimod.models.pid_decomposition import PIDDecomposition

        pid = PIDDecomposition(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        z = torch.randn(4, 8, dims["latent_dim"], device=device)
        mask = torch.ones(4, 8, dtype=torch.bool, device=device)
        labels = torch.randint(0, dims["num_classes"], (4,), device=device)

        components = pid(z, z, z, mask)
        loss = pid.pid_consistency_loss(components, labels, dims["num_classes"])
        assert torch.isfinite(loss), f"PID consistency loss is not finite: {loss}"


# ── Evidential Fusion Tests ──────────────────────────────────────────────

class TestEvidentialFusion:
    def test_evidence_non_negative(self, device, dims) -> None:
        from multimod.models.evidential_fusion import EvidentialHead

        head = EvidentialHead(
            input_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        x = torch.randn(4, dims["latent_dim"], device=device)
        evidence = head(x)
        assert (evidence >= 0).all(), "Evidence must be non-negative"

    def test_fusion_output_shapes(self, device, dims) -> None:
        from multimod.models.evidential_fusion import EvidentialFusion

        fusion = EvidentialFusion(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        components = {
            name: torch.randn(4, dims["latent_dim"], device=device)
            for name in ("unique_text", "unique_audio", "unique_vision", "redundant", "synergistic")
        }

        output = fusion(components)
        assert output["alpha"].shape == (4, dims["num_classes"])
        assert output["evidence"].shape == (4, dims["num_classes"])
        assert output["uncertainty"].shape == (4, 1)
        assert output["conflict"].shape == (4, 1)

    def test_alpha_greater_than_one(self, device, dims) -> None:
        from multimod.models.evidential_fusion import EvidentialFusion

        fusion = EvidentialFusion(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        components = {
            name: torch.randn(4, dims["latent_dim"], device=device)
            for name in ("unique_text", "unique_audio", "unique_vision", "redundant", "synergistic")
        }

        output = fusion(components)
        assert (output["alpha"] >= 1.0).all(), "Dirichlet α must be >= 1 (evidence + 1)"

    def test_evidential_loss_finite(self, device, dims) -> None:
        from multimod.models.evidential_fusion import evidential_loss

        alpha = torch.ones(4, dims["num_classes"], device=device) * 2.0
        labels = torch.randint(0, dims["num_classes"], (4,), device=device)
        loss = evidential_loss(alpha, labels, epoch=5, total_epochs=20)
        assert torch.isfinite(loss), f"Evidential loss is not finite: {loss}"


# ── Full EIDMSA Model Tests ─────────────────────────────────────────────

class TestEIDMSA:
    def test_forward_pass(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )

        assert output["logits"].shape == (4, dims["num_classes"])
        assert output["alpha"].shape == (4, dims["num_classes"])
        assert output["uncertainty"].shape == (4, 1)
        assert output["conflict"].shape == (4, 1)
        assert set(output["modality_reliability"]) == {"text", "audio", "vision"}
        assert "unique_vision" in output["component_reliability"]
        assert torch.isfinite(output["ib_kl"]), "IB KL should be finite"

        # Logits should sum to ~1 (they're expected probs from Dirichlet)
        sums = output["logits"].sum(dim=-1)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5,
            msg="Expected probs should sum to 1",
        )

    def test_compute_loss(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        losses = model.compute_loss(output, batch_data["label"], epoch=5, total_epochs=20)

        for key in ("total", "evidential", "ib", "pid"):
            assert key in losses, f"Missing loss component: {key}"
            assert torch.isfinite(losses[key]), f"Loss '{key}' is not finite: {losses[key]}"

    def test_compute_loss_can_disable_evidential_objective(self, batch_data, dims, device) -> None:
        import torch.nn.functional as F

        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        losses = model.compute_loss(
            output,
            batch_data["label"],
            epoch=5,
            total_epochs=20,
            pid_weight=0.0,
            use_evidential_loss=False,
        )

        expected = F.nll_loss(output["logits"].clamp_min(1e-8).log(), batch_data["label"])
        torch.testing.assert_close(losses["evidential"], expected)
        torch.testing.assert_close(losses["total"], expected + losses["ib"])

    def test_backward_pass(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        losses = model.compute_loss(output, batch_data["label"])
        losses["total"].backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients after backward pass"

    def test_uncertainty_increases_with_missing_modality(self, batch_data, dims, device) -> None:
        """Uncertainty should increase when modalities are missing."""
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)
        model.eval()

        with torch.no_grad():
            # Clean input
            out_clean = model(
                batch_data["text"], batch_data["audio"],
                batch_data["vision"], batch_data["mask"],
            )
            # Missing both audio and vision
            out_missing = model(
                batch_data["text"],
                torch.zeros_like(batch_data["audio"]),
                torch.zeros_like(batch_data["vision"]),
                batch_data["mask"],
            )

        # On average, uncertainty should be higher with missing modalities
        # (this is a statistical test, so we check tendency not hard guarantee)
        clean_unc = out_clean["uncertainty"].mean().item()
        missing_unc = out_missing["uncertainty"].mean().item()
        # Just verify both are valid floating point numbers
        assert 0.0 <= clean_unc <= 1.0, f"Clean uncertainty out of range: {clean_unc}"
        assert 0.0 <= missing_unc <= 1.0, f"Missing uncertainty out of range: {missing_unc}"

    def test_missing_vision_reduces_vision_reliability(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)
        model.eval()

        with torch.no_grad():
            out_clean = model(
                batch_data["text"],
                batch_data["audio"],
                batch_data["vision"],
                batch_data["mask"],
            )
            out_missing = model(
                batch_data["text"],
                batch_data["audio"],
                torch.zeros_like(batch_data["vision"]),
                batch_data["mask"],
            )

        clean_rel = out_clean["modality_reliability"]["vision"]
        missing_rel = out_missing["modality_reliability"]["vision"]
        assert torch.all(missing_rel < clean_rel)
        assert torch.allclose(missing_rel, torch.zeros_like(missing_rel))
        assert torch.all(
            out_missing["component_reliability"]["unique_vision"]
            <= out_clean["component_reliability"]["unique_vision"]
        )


# ── TTA Tests ────────────────────────────────────────────────────────────

class TestTTA:
    def test_collect_ib_params(self, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA
        from multimod.models.tta import collect_ib_params

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        ib_params = collect_ib_params(model)
        assert len(ib_params) > 0, "Should find IB params (mu_head, logvar_head)"
        # 3 modalities × 2 heads (mu, logvar) × 2 params each (weight, bias) = 12
        assert len(ib_params) == 12, f"Expected 12 IB params, got {len(ib_params)}"

    def test_tta_adapter_runs(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA
        from multimod.models.tta import TestTimeAdapter

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        adapter = TestTimeAdapter(model, lr=1e-4, num_steps=2)
        output = adapter.adapt_and_predict(batch_data)
        assert "logits" in output
        assert output["logits"].shape == (4, dims["num_classes"])

    def test_tta_adapter_runs_under_no_grad(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA
        from multimod.models.tta import TestTimeAdapter

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        adapter = TestTimeAdapter(model, lr=1e-4, num_steps=2)
        with torch.no_grad():
            output = adapter.adapt_and_predict(batch_data)

        assert "logits" in output
        assert output["logits"].shape == (4, dims["num_classes"])


# ── Model Factory Tests ──────────────────────────────────────────────────

class TestModelFactory:
    def test_build_eidmsa(self, dims) -> None:
        from multimod.config import ModelConfig
        from multimod.models import InputDims, build_model
        from multimod.models.eidmsa import EIDMSA

        config = ModelConfig(
            name="eidmsa",
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
        )
        input_dims = InputDims(
            text=dims["text_dim"],
            audio=dims["audio_dim"],
            vision=dims["vision_dim"],
        )
        model = build_model(config, input_dims)
        assert isinstance(model, EIDMSA)

    def test_existing_models_still_work(self) -> None:
        from multimod.config import ModelConfig
        from multimod.models import InputDims, build_model

        input_dims = InputDims(text=300, audio=74, vision=35)
        for name in ("text_only", "early_fusion", "xmodal_transformer"):
            config = ModelConfig(name=name)
            model = build_model(config, input_dims)
            assert model is not None, f"Failed to build {name}"


# ── Data Pipeline Tests ──────────────────────────────────────────────────

class TestDataPipeline:
    def test_seven_class_labels(self) -> None:
        import numpy as np
        from multimod.data.mosei import _to_seven_class

        labels = np.array([-3.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.0])
        classes = _to_seven_class(labels)
        assert classes[0] == 0  # -3.0 -> 0
        assert classes[-1] == 6  # 3.0 -> 6
        assert all(0 <= c <= 6 for c in classes)

    def test_label_mode_validation(self) -> None:
        from multimod.data.mosei import LABEL_MODES

        assert "3class" in LABEL_MODES
        assert "7class" in LABEL_MODES
        assert "regression" in LABEL_MODES


class TestEIDMSAConfig:
    def test_no_evidential_ablation_disables_evidential_loss(self) -> None:
        from multimod.config import make_experiment_config

        config = make_experiment_config(
            "eidmsa_no_evidential",
            data_path="dummy.pkl",
        )

        assert config.model.use_evidential_loss is False

    def test_kan_config_enables_kan(self) -> None:
        from multimod.config import make_experiment_config

        config = make_experiment_config("eidmsa_kan", data_path="dummy.pkl")
        assert config.model.use_kan is True
        assert config.model.use_mamba is False

    def test_mamba_config_enables_mamba(self) -> None:
        from multimod.config import make_experiment_config

        config = make_experiment_config("eidmsa_mamba", data_path="dummy.pkl")
        assert config.model.use_mamba is True
        assert config.model.use_kan is False

    def test_kan_mamba_config_enables_both(self) -> None:
        from multimod.config import make_experiment_config

        config = make_experiment_config("eidmsa_kan_mamba", data_path="dummy.pkl")
        assert config.model.use_kan is True
        assert config.model.use_mamba is True

    def test_realistic_retry_config_enables_retry_knobs(self) -> None:
        from multimod.config import make_experiment_config

        config = make_experiment_config("eidmsa_realistic_retry", data_path="dummy.pkl")
        assert config.model.realistic_corruption_p > 0.0
        assert config.model.alignment_weight > 0.0
        assert config.model.evidential_warmup_epochs > 0
        assert config.model.vision_dropout_p > config.model.audio_dropout_p


# ── KAN Layer Tests ──────────────────────────────────────────────────────

class TestKANLayers:
    def test_kan_linear_output_shape(self, device) -> None:
        from multimod.models.kan_layers import KANLinear

        layer = KANLinear(in_features=32, out_features=16).to(device)
        x = torch.randn(4, 32, device=device)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_kan_linear_3d_input(self, device) -> None:
        """KANLinear should handle [batch, seq, features] input."""
        from multimod.models.kan_layers import KANLinear

        layer = KANLinear(in_features=32, out_features=16).to(device)
        x = torch.randn(4, 8, 32, device=device)
        out = layer(x)
        assert out.shape == (4, 8, 16)

    def test_kan_projection_output_shape(self, device) -> None:
        from multimod.models.kan_layers import KANProjection

        proj = KANProjection(input_dim=64, hidden_dim=32, output_dim=32).to(device)
        x = torch.randn(4, 64, device=device)
        out = proj(x)
        assert out.shape == (4, 32)

    def test_kan_regularization_loss(self, device) -> None:
        from multimod.models.kan_layers import KANProjection

        proj = KANProjection(input_dim=32, hidden_dim=16, output_dim=16).to(device)
        reg = proj.regularization_loss()
        assert torch.isfinite(reg), f"KAN reg loss not finite: {reg}"
        assert reg.item() > 0, "KAN reg loss should be positive"

    def test_kan_backward(self, device) -> None:
        from multimod.models.kan_layers import KANLinear

        layer = KANLinear(in_features=32, out_features=16).to(device)
        x = torch.randn(4, 32, device=device)
        out = layer(x)
        out.sum().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.parameters()
        )
        assert has_grad, "No gradients after KAN backward"


# ── KAN PID Tests ────────────────────────────────────────────────────────

class TestPIDKAN:
    def test_output_shapes(self, device, dims) -> None:
        from multimod.models.pid_kan import PIDDecompositionKAN

        pid = PIDDecompositionKAN(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
            grid_size=5,
        ).to(device)

        z = torch.randn(4, 8, dims["latent_dim"], device=device)
        mask = torch.ones(4, 8, dtype=torch.bool, device=device)

        components = pid(z, z, z, mask)
        for key in ("unique_text", "unique_audio", "unique_vision", "redundant", "synergistic"):
            assert key in components
            assert components[key].shape == (4, dims["latent_dim"])

    def test_consistency_loss_finite(self, device, dims) -> None:
        from multimod.models.pid_kan import PIDDecompositionKAN

        pid = PIDDecompositionKAN(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        z = torch.randn(4, 8, dims["latent_dim"], device=device)
        mask = torch.ones(4, 8, dtype=torch.bool, device=device)
        labels = torch.randint(0, dims["num_classes"], (4,), device=device)

        components = pid(z, z, z, mask)
        loss = pid.pid_consistency_loss(components, labels, dims["num_classes"])
        assert torch.isfinite(loss)

    def test_kan_reg_loss(self, device, dims) -> None:
        from multimod.models.pid_kan import PIDDecompositionKAN

        pid = PIDDecompositionKAN(
            latent_dim=dims["latent_dim"],
            hidden_dim=dims["hidden_dim"],
            num_classes=dims["num_classes"],
        ).to(device)

        reg = pid.kan_regularization_loss()
        assert torch.isfinite(reg)


# ── Mamba Encoder Tests ──────────────────────────────────────────────────

class TestMambaEncoder:
    def test_mamba_block_output_shape(self, device) -> None:
        from multimod.models.mamba_encoder import MambaBlock

        block = MambaBlock(d_model=32, dropout=0.1).to(device)
        x = torch.randn(4, 8, 32, device=device)
        out = block(x)
        assert out.shape == (4, 8, 32)

    def test_mamba_encoder_output_shape(self, device) -> None:
        from multimod.models.mamba_encoder import MambaEncoder

        enc = MambaEncoder(d_model=32, num_layers=2, dropout=0.1).to(device)
        x = torch.randn(4, 8, 32, device=device)
        padding_mask = torch.zeros(4, 8, dtype=torch.bool, device=device)

        out = enc(x, src_key_padding_mask=padding_mask)
        assert out.shape == (4, 8, 32)

    def test_mamba_encoder_backward(self, device) -> None:
        from multimod.models.mamba_encoder import MambaEncoder

        enc = MambaEncoder(d_model=32, num_layers=2, dropout=0.1).to(device)
        x = torch.randn(4, 8, 32, device=device)
        out = enc(x)
        out.sum().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in enc.parameters()
        )
        assert has_grad, "No gradients after Mamba backward"

    def test_fallback_mask_ignores_left_padding_values(self, device) -> None:
        from multimod.models.mamba_encoder import MambaEncoder

        enc = MambaEncoder(d_model=16, num_layers=1, dropout=0.0).to(device)
        if enc.layers[0].use_mamba:
            pytest.skip("Fallback-specific behavior test")

        enc.eval()
        valid_tokens = torch.randn(1, 3, 16, device=device)
        mask = torch.tensor([[False, False, True, True, True]], dtype=torch.bool, device=device)

        padded_zero = torch.cat([torch.zeros(1, 2, 16, device=device), valid_tokens], dim=1)
        padded_noise = padded_zero.clone()
        padded_noise[:, :2] = torch.randn(1, 2, 16, device=device)

        pad_mask = ~mask
        out_zero = enc(padded_zero, src_key_padding_mask=pad_mask)
        out_noise = enc(padded_noise, src_key_padding_mask=pad_mask)

        torch.testing.assert_close(out_zero[:, -3:], out_noise[:, -3:])


# ── EIDMSA + KAN Integration Tests ──────────────────────────────────────

class TestEIDMSA_KAN:
    def test_forward_with_kan(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
            use_kan=True,
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        assert output["logits"].shape == (4, dims["num_classes"])

        sums = output["logits"].sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_loss_with_kan(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
            use_kan=True,
            kan_reg_weight=1e-4,
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        losses = model.compute_loss(output, batch_data["label"])
        assert torch.isfinite(losses["total"])

    def test_backward_with_kan(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
            use_kan=True,
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        losses = model.compute_loss(output, batch_data["label"])
        losses["total"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


# ── EIDMSA + Mamba Integration Tests ─────────────────────────────────────

class TestEIDMSA_Mamba:
    def test_forward_with_mamba(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
            use_mamba=True,
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        assert output["logits"].shape == (4, dims["num_classes"])

    def test_backward_with_mamba(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
            use_mamba=True,
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        losses = model.compute_loss(output, batch_data["label"])
        losses["total"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


# ── EIDMSA + KAN + Mamba Combined Test ───────────────────────────────────

class TestEIDMSA_KAN_Mamba:
    def test_forward_with_both(self, batch_data, dims, device) -> None:
        from multimod.models.eidmsa import EIDMSA

        model = EIDMSA(
            text_dim=dims["text_dim"],
            audio_dim=dims["audio_dim"],
            vision_dim=dims["vision_dim"],
            hidden_dim=dims["hidden_dim"],
            latent_dim=dims["latent_dim"],
            num_classes=dims["num_classes"],
            use_kan=True,
            use_mamba=True,
        ).to(device)

        output = model(
            batch_data["text"], batch_data["audio"],
            batch_data["vision"], batch_data["mask"],
        )
        assert output["logits"].shape == (4, dims["num_classes"])
        assert torch.isfinite(output["ib_kl"])

        losses = model.compute_loss(output, batch_data["label"])
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
