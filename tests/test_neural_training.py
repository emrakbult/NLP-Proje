import pytest

from src.neural_training import compute_combined_loss, select_best_epoch


torch = pytest.importorskip("torch")


def test_combined_loss_uses_cosine_mse_and_cross_entropy_weights() -> None:
    output = {
        "similarity_score": torch.tensor([0.2, 0.8]),
        "class_logits": torch.tensor([[2.0, 0.2, -1.0], [-1.0, 0.2, 2.0]]),
    }
    similarity_target = torch.tensor([0.0, 1.0])
    class_label = torch.tensor([0, 2])

    total_loss, cosine_loss, classification_loss = compute_combined_loss(
        output,
        similarity_target,
        class_label,
        cosine_weight=0.6,
        classification_weight=0.4,
    )

    expected_total = (0.6 * cosine_loss) + (0.4 * classification_loss)
    assert total_loss.item() == pytest.approx(expected_total.item())
    assert cosine_loss.item() > 0
    assert classification_loss.item() > 0


def test_select_best_epoch_uses_lowest_validation_total_loss() -> None:
    history = [
        {"epoch": 1, "validation_total_loss": 0.72},
        {"epoch": 2, "validation_total_loss": 0.58},
        {"epoch": 3, "validation_total_loss": 0.64},
    ]

    assert select_best_epoch(history) == 2
