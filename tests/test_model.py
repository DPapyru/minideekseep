from dataclasses import replace

import pytest
import torch

from minideekseep import MiniDeepSeek, ModelConfig
from minideekseep.layers import HyperConnection
from minideekseep.training import make_optimizer, routing_counts


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def test_parameter_budget():
    with torch.device("meta"):
        report = MiniDeepSeek(ModelConfig()).parameter_report()
    assert report["encoder_target_met"] and report["decode_target_met"]
    assert 140e6 < report["total"] < 145e6
    assert report["engram_tables"] > 0


def test_causality_and_document_reset():
    model = MiniDeepSeek(ModelConfig.tiny()).eval()
    ids = torch.randint(5, 288, (1, 15))
    changed = ids.clone()
    changed[:, 7:] = torch.randint(5, 288, (1, 8))
    with torch.no_grad():
        before, after = model(ids), model(changed)
        torch.testing.assert_close(before[:, :7], after[:, :7], atol=2e-6, rtol=2e-5)
        first, _ = model.prefill(ids)
        model.prefill(changed)
        second, _ = model.prefill(ids)
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("prefix", [1, 3, 4, 7])
def test_exact_cache_matches_full_forward(prefix):
    model = MiniDeepSeek(ModelConfig.tiny()).eval()
    ids = torch.randint(5, 288, (2, 17))
    with torch.no_grad():
        full = model(ids)
        logits, cache = model.prefill(ids[:, :prefix])
        torch.testing.assert_close(logits, full[:, prefix - 1], atol=3e-6, rtol=3e-5)
        for position in range(prefix, ids.size(1)):
            logits, cache = model.decode_step(ids[:, position : position + 1], cache)
            torch.testing.assert_close(logits, full[:, position], atol=3e-6, rtol=3e-5)
    assert cache.position == 17
    for local in cache.local.values():
        assert local.values.size(1) == model.config.window


def test_bounded_replay_is_explicit_and_preserves_globals():
    model = MiniDeepSeek(ModelConfig.tiny()).eval()
    ids = torch.randint(5, 288, (1, 15))
    exact_logits, exact = model.prefill(ids)
    replay_logits, replay = model.prefill(ids, bounded=True)
    assert replay.approximate and not exact.approximate
    torch.testing.assert_close(exact.global_kv[6].values, replay.global_kv[6].values)
    assert torch.isfinite(replay_logits).all()
    logits, replay = model.decode_step(ids[:, :1], replay)
    assert logits.shape == exact_logits.shape and replay.position == 16


@pytest.mark.parametrize("checkpointed", [False, True])
def test_gradients_include_shared_kv_indexer_router_and_engram(checkpointed):
    model = MiniDeepSeek(ModelConfig.tiny(checkpoint_layers=checkpointed)).train()
    ids = torch.randint(5, 288, (1, 13))
    with routing_counts(model):
        output = model.forward_train(ids[:, :-1], ids[:, 1:])
    output.loss.backward()
    for name in (
        "layers.1.attention.global_kv.weight",
        "layers.1.attention.index_query.weight",
        "layers.6.attention.global_kv.weight",
        "layers.8.attention.index_query.weight",
        "layers.0.moe.router.weight",
        "layers.1.engram.tables.0.weight",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    for layer in model.layers:
        assert layer.moe.route_counts.sum() == 12 * model.config.top_k
    model.update_router_biases()
    assert model.layers[0].moe.route_counts.sum() == 0
    assert model.layers[0].moe.route_bias.abs().sum() > 0


def test_checkpointed_gradients_match_reference():
    c = ModelConfig.tiny()
    reference = MiniDeepSeek(c).train()
    checked = MiniDeepSeek(replace(c, checkpoint_layers=True)).train()
    checked.load_state_dict(reference.state_dict())
    ids = torch.randint(5, 288, (1, 9))
    reference.forward_train(ids[:, :-1], ids[:, 1:]).loss.backward()
    checked.forward_train(ids[:, :-1], ids[:, 1:]).loss.backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), checked.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, msg=name)


def test_mhc_constraints():
    mixer = HyperConnection(ModelConfig.tiny())
    pre, post, mix = mixer.coefficients(torch.randn(2, 5, 4, 32))
    assert ((pre >= 0) & (pre <= 1)).all()
    assert ((post >= 0) & (post <= 2)).all()
    torch.testing.assert_close(mix.sum(-1), torch.ones(2, 5, 4), atol=1e-5, rtol=1e-5)
    # A fixed twenty Sinkhorn iterations gives an approximate doubly stochastic matrix.
    torch.testing.assert_close(mix.sum(-2), torch.ones(2, 5, 4), atol=1e-4, rtol=1e-5)


def test_small_corpus_overfits():
    model = MiniDeepSeek(ModelConfig.tiny(index_loss_weight=0)).train()
    optimizer = make_optimizer(model, lr=0.008)
    ids = torch.tensor([[1, 21, 22, 23, 21, 22, 23, 2]])
    initial = model.forward_train(ids[:, :-1], ids[:, 1:]).lm_loss.item()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = model.forward_train(ids[:, :-1], ids[:, 1:]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    final = model.forward_train(ids[:, :-1], ids[:, 1:]).lm_loss.item()
    assert final < initial * 0.4, (initial, final)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_training_and_cache():
    model = MiniDeepSeek(ModelConfig.tiny(checkpoint_layers=True)).cuda().train()
    ids = torch.randint(5, 288, (1, 17), device="cuda")
    output = model.forward_train(ids[:, :-1], ids[:, 1:])
    output.loss.backward()
    assert torch.isfinite(output.loss)
    model.eval()
    with torch.no_grad():
        expected = model(ids)
        _, cache = model.prefill(ids[:, :7])
        for index in range(7, ids.size(1)):
            logits, cache = model.decode_step(ids[:, index : index + 1], cache)
            torch.testing.assert_close(logits, expected[:, index], atol=2e-5, rtol=2e-4)
    model.train()
    model.zero_grad(set_to_none=True)
    model.set_fake_quantization()
    qat_loss = model.forward_replay(ids[:, :-1], ids[:, 1:]).loss
    qat_loss.backward()
    assert torch.isfinite(qat_loss)
