import copy
import json

import pytest
import torch

from minideekseep import MiniDeepSeek, ModelConfig
from minideekseep.data import TokenStream, collect, encode_corpus, train_tokenizer, verify_corpus
from minideekseep.training import (
    benchmark,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    seed_everything,
    train,
)


def fixture_data(tmp_path):
    source = tmp_path / "source.jsonl"
    with source.open("w") as handle:
        for i in range(300):
            handle.write(
                json.dumps(dict(text=(f"Document number {i}. " * 12), language="zh")) + "\n"
            )
    directory = tmp_path / "corpus"
    collect(directory, source=source, documents=300)
    tokenizer = train_tokenizer(directory, 288)
    encode_corpus(directory, tokenizer)
    return directory, tokenizer


def test_tokenizer_no_unknown_and_resumable_sampler(tmp_path):
    directory, tokenizer = fixture_data(tmp_path)
    text = "\u4f60\u597d\uff0c\u4e16\u754c! \U0001f600"
    ids = tokenizer.encode(text).ids
    assert 3 not in ids
    assert tokenizer.decode(ids) == text
    stream = TokenStream(directory, length=16)
    stream.next()
    state = copy.deepcopy(stream.state_dict())
    expected = stream.next()
    stream.load_state_dict(state)
    actual = stream.next()
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])


def test_sequences_never_cross_document_boundaries(tmp_path):
    directory, _ = fixture_data(tmp_path)
    stream = TokenStream(directory, length=1024)
    for _ in range(5):
        x, y = stream.next()
        valid = int((y != -100).sum())
        # Every fixture document is shorter than the window; EOS is the final target.
        assert x[0, 0] == 1 and y[0, valid - 1] == 2
        assert (y[0, valid:] == -100).all()
        assert not (x[0, 1:valid] == 1).any()


def test_checkpoint_next_step_matches(tmp_path):
    torch.set_num_threads(1)
    seed_everything(5)
    directory, _ = fixture_data(tmp_path)
    model = MiniDeepSeek(ModelConfig.tiny()).train()
    optimizer = make_optimizer(model)
    stream = TokenStream(directory, length=8)

    def step(m, opt, data):
        x, y = data.next()
        opt.zero_grad(set_to_none=True)
        loss = m.forward_train(x, y).loss
        loss.backward()
        opt.step()
        return float(loss.detach())

    step(model, optimizer, stream)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, stream, 1)
    expected = step(model, optimizer, stream)
    loaded, state = load_checkpoint(path, restore_rng=True)
    other_opt = make_optimizer(loaded)
    other_opt.load_state_dict(state["optimizer"])
    other_stream = TokenStream(directory, length=8)
    other_stream.load_state_dict(state["stream"])
    actual = step(loaded, other_opt, other_stream)
    assert expected == actual
    for a, b in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b)
    assert state["step"] == 1


def test_changed_shard_is_rejected(tmp_path):
    directory, _ = fixture_data(tmp_path)
    verify_corpus(directory)
    with (directory / "train_zh.bin").open("ab") as handle:
        handle.write(b"\x00\x00")
    with pytest.raises(ValueError, match="shard was changed"):
        verify_corpus(directory)


def test_benchmark_reports_actual_accumulation_and_bounded_budget():
    result = benchmark(
        ModelConfig.tiny(), length=8, device="cpu", steps=1, seconds=0, tokens_per_step=9
    )
    assert result["tokens_per_step"] == 16
    assert result["estimated_72h_tokens"] <= 50_000_000
    assert result["estimated_72h_tokens"] % 16 == 0


def test_training_loop_accumulates_real_tokens_and_resumes(tmp_path):
    directory, _ = fixture_data(tmp_path)
    run = tmp_path / "run"
    train(
        directory,
        run,
        ModelConfig.tiny(),
        token_budget=1000,
        device="cpu",
        length=8,
        tokens_per_step=16,
        max_steps=1,
    )
    _, first = load_checkpoint(run / "latest.pt")
    assert first["step"] == 1 and first["stream"]["tokens"] >= 16
    train(
        directory,
        run,
        ModelConfig.tiny(),
        token_budget=1000,
        device="cpu",
        length=8,
        tokens_per_step=16,
        max_steps=1,
        resume=run / "latest.pt",
    )
    _, second = load_checkpoint(run / "latest.pt")
    assert second["step"] == 2 and second["stream"]["tokens"] >= 32
    assert (run / "previous.pt").exists()


def test_replay_qat_stage_restores_its_runtime_mode(tmp_path):
    directory, _ = fixture_data(tmp_path)
    run = tmp_path / "qat"
    train(
        directory,
        run,
        ModelConfig.tiny(),
        token_budget=1000,
        device="cpu",
        length=8,
        tokens_per_step=16,
        max_steps=1,
        replay=True,
        fake_quant=True,
    )
    restored, state = load_checkpoint(run / "latest.pt")
    assert state["metadata"]["replay"] and state["metadata"]["fake_quant"]
    assert all(layer.attention.fake_quant for layer in restored.layers)


def test_draft_training_stage_saves_only_draft_parameters(tmp_path):
    from minideekseep.data import fingerprint
    from minideekseep.dspark import train_draft

    directory, _ = fixture_data(tmp_path)
    model = MiniDeepSeek(ModelConfig.tiny())
    path = tmp_path / "base.pt"
    save_checkpoint(
        path,
        model,
        make_optimizer(model),
        None,
        0,
        {"tokenizer_sha256": fingerprint(directory / "tokenizer.json")},
    )
    draft_path = train_draft(
        path,
        directory,
        tmp_path / "draft",
        device="cpu",
        tokens=100,
        length=8,
        max_steps=1,
        width=16,
    )
    saved = torch.load(draft_path, weights_only=False)
    assert saved["step"] == 1
    assert "embedding.weight" not in saved["draft"]
    assert saved["backbone_sha256"] == fingerprint(path)
    train_draft(
        path,
        directory,
        tmp_path / "draft",
        device="cpu",
        tokens=100,
        length=8,
        max_steps=1,
        width=16,
        resume=draft_path,
    )
    resumed = torch.load(draft_path, weights_only=False)
    assert resumed["step"] == 2
    assert resumed["stream"]["tokens"] > saved["stream"]["tokens"]
