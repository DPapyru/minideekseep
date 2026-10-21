import torch

from minideekseep import MiniDeepSeek, ModelConfig
from minideekseep.dspark import DSpark, choose_length, corrected_distribution, speculative_round
from minideekseep.quantization import fake_fp4, fake_fp8


def test_quantization_is_finite_and_straight_through():
    for operation in (fake_fp4, fake_fp8):
        x = torch.randn(2, 5, 32, requires_grad=True)
        y = operation(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        assert not torch.equal(x, y)
        assert torch.isfinite(operation(torch.zeros_like(x))).all()


def test_draft_is_trainable_without_backbone_gradients():
    torch.set_num_threads(1)
    backbone = MiniDeepSeek(ModelConfig.tiny()).eval()
    draft = DSpark(backbone, width=16, markov_rank=4)
    ids = torch.randint(5, 288, (1, 8))
    with torch.no_grad():
        feature, _, _ = backbone._run(ids)
    hidden = draft.base(feature[:, -1], ids[:, -1])
    logits, confidence = draft.conditional(hidden[:, 0], ids[:, -1])
    (logits.square().mean() + confidence.square().mean()).backward()
    assert draft.condition.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
    assert not any("embedding.weight" in name for name in draft.state_dict())
    output, cache, stats = speculative_round(backbone, draft, ids)
    assert 1 <= len(output) <= 3
    assert cache.position == 8 + len(output)
    assert stats["accepted"] <= len(output)


def test_rejection_distribution():
    p = torch.tensor([[0.1, 0.6, 0.3]])
    q = torch.tensor([[0.4, 0.4, 0.2]])
    torch.testing.assert_close(corrected_distribution(p, q), torch.tensor([[0, 2 / 3, 1 / 3]]))
    torch.testing.assert_close(corrected_distribution(p, p), p)


def test_confidence_scheduler_uses_survival_and_measured_cost():
    profile = {
        "rounds": 1,
        "curves": [
            {"verification_length": 1, "seconds": 1},
            {"verification_length": 3, "seconds": 2},
        ],
    }
    assert choose_length(torch.tensor([[0.99, 0.99, 0.99]]), profile) == 3
    assert choose_length(torch.tensor([[0.01, 0.01, 0.01]]), profile) == 1


def test_replay_adaptation_and_qat_have_gradients():
    torch.set_num_threads(1)
    model = MiniDeepSeek(ModelConfig.tiny()).train()
    model.set_fake_quantization()
    ids = torch.randint(5, 288, (1, 14))
    loss = model.forward_replay(ids[:, :-1], ids[:, 1:]).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.layers[6].attention.global_kv.weight.grad.abs().sum() > 0


def test_rejected_draft_does_not_enter_cache(monkeypatch):
    torch.set_num_threads(1)
    backbone = MiniDeepSeek(ModelConfig.tiny()).eval()
    draft = DSpark(backbone, width=16, markov_rank=4).eval()
    prefix = torch.randint(5, 288, (1, 9))
    logits, original = backbone.prefill(prefix)
    rejected = logits.argmin(-1)

    def forced_proposal(*args, **kwargs):
        q = torch.zeros_like(logits).scatter_(-1, rejected[:, None], 1)
        return rejected[:, None].expand(-1, 3), [q] * 3, torch.ones(1, 3)

    monkeypatch.setattr(draft, "propose", forced_proposal)
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.ones_like(x))
    output, cache, stats = speculative_round(backbone, draft, prefix)
    assert stats["accepted"] == 0 and output[0] != rejected.item()
    _, expected = backbone.decode_step(torch.tensor([output]), original)
    assert cache.position == expected.position
    for index in cache.local:
        torch.testing.assert_close(cache.local[index].values, expected.local[index].values)
    assert original.position == prefix.size(1)
