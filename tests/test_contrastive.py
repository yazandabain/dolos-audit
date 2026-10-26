"""Contrastive probability algebra and real tiny-Qwen cached decoding checks."""

import math
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from dolos_audit.contrastive import (  # noqa: E402
    contrastive_scores,
    decode_batch,
    sequence_seed,
    tokenize_prompts,
    validate_contrastive_config,
)


def test_score_mask_precedes_amplification_and_beta_zero_is_target():
    target = torch.log(torch.tensor([[0.6, 0.3, 0.09, 0.01]]))
    reference = torch.log(torch.tensor([[0.8, 0.19, 0.00999, 0.00001]]))
    scored = contrastive_scores(target, reference, beta=1, alpha=0.1)
    assert scored["mask"].tolist() == [[True, True, True, False]]
    assert torch.isneginf(scored["score"][0, 3])
    assert torch.allclose(scored["score"][0, :3], (2 * target - reference)[0, :3])
    zero = contrastive_scores(target, reference, beta=0, alpha=0.1)
    assert torch.allclose(zero["score"][0, :3], target[0, :3])
    same = contrastive_scores(target, target, beta=1, alpha=0.1)
    assert torch.allclose(zero["score"], same["score"])


def test_invalid_distribution_or_parameters_fail():
    logits = torch.tensor([0.0, 1.0])
    for beta, alpha in [(-1, 0.1), (1, 1.1), (math.nan, 0.1), (1, math.inf)]:
        with pytest.raises(ValueError):
            contrastive_scores(logits, logits, beta=beta, alpha=alpha)
    with pytest.raises(ValueError, match="Nonfinite"):
        contrastive_scores(torch.tensor([0.0, math.nan]), logits, beta=1, alpha=0.1)
    with pytest.raises(ValueError, match="shape"):
        contrastive_scores(logits, logits[:1], beta=1, alpha=0.1)


class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 9  # Chat end is intentionally distinct from document boundary.
    unk_token_id = None

    def encode(self, text, **kwargs):
        return [ord(c) % 8 + 1 for c in text]

    def convert_tokens_to_ids(self, token):
        assert token == "<|endoftext|>"
        return 10


def test_empty_prefix_uses_recorded_document_boundary_and_no_truncation():
    prompts = [{"raw_prefix": ""}, {"raw_prefix": "The"}]
    ids, mask, rendered, rows, fallback = tokenize_prompts(TokenizerStub(), prompts, "cpu", 4)
    assert rendered == ["", "The"]
    assert fallback == [True, False]
    assert rows[0] == [10]
    assert ids[0].tolist() == [0, 0, 10]
    assert mask.tolist() == [[0, 0, 1], [1, 1, 1]]
    with pytest.raises(ValueError, match="truncation"):
        tokenize_prompts(TokenizerStub(), prompts, "cpu", 2)


@pytest.fixture
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(811)
    config = Qwen2Config(
        vocab_size=23,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=22,
        bos_token_id=1,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config).eval()


def _decode(model, ids, mask, *, seeds, beta=0, temperature=0, eos=(22,), cap=6):
    return decode_batch(
        {"dolos": model, "qwen": model},
        ids,
        mask,
        arm={"target": "dolos", "reference": "qwen", "beta": beta, "alpha": 0},
        max_new_tokens=cap,
        temperature=temperature,
        eos_token_ids=eos,
        pad_token_id=0,
        seeds=seeds,
        save_top_k=5,
    )


def test_cached_left_padded_greedy_matches_transformers_generate(tiny_model):
    ids = torch.tensor([[0, 0, 3], [4, 5, 6]])
    mask = torch.tensor([[0, 0, 1], [1, 1, 1]])
    result = _decode(tiny_model, ids, mask, seeds=[1, 2], beta=1)
    with torch.inference_mode():
        generated = tiny_model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=22,
            pad_token_id=0,
        )
    for row, raw in zip(result, generated[:, 3:].tolist(), strict=True):
        if 22 in raw:
            raw = raw[: raw.index(22) + 1]
        assert row["output_token_ids"] == raw
        assert len(row["token_trace"]) == len(raw)
        for token, trace in zip(raw, row["token_trace"], strict=True):
            assert token == trace["token_id"]
            assert trace["eligible_tokens"] == 23
            assert trace["target_logprob"] == trace["reference_logprob"]


def test_row_rng_is_independent_of_batch_layout(tiny_model):
    seed_a, seed_b = sequence_seed(1000, "a"), sequence_seed(1000, "b")
    assert seed_a != seed_b
    batched = _decode(
        tiny_model,
        torch.tensor([[0, 0, 3], [4, 5, 6]]),
        torch.tensor([[0, 0, 1], [1, 1, 1]]),
        seeds=[seed_a, seed_b],
        temperature=1,
        eos=(),
    )
    for i, (ids, seed) in enumerate([([3], seed_a), ([4, 5, 6], seed_b)]):
        alone = _decode(
            tiny_model,
            torch.tensor([ids]),
            torch.ones((1, len(ids)), dtype=torch.long),
            seeds=[seed],
            temperature=1,
            eos=(),
        )
        assert batched[i]["output_token_ids"] == alone[0]["output_token_ids"]


def test_eos_at_cap_is_not_censoring(tiny_model):
    model = deepcopy(tiny_model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = _decode(model, torch.tensor([[3]]), torch.tensor([[1]]), seeds=[1], eos=(0,), cap=1)
    assert result[0]["output_token_ids"] == [0]
    assert result[0]["stopped_at_eos"] is True
    assert result[0]["max_tokens_reached"] is True
    assert result[0]["censored"] is False


def test_distinct_models_receive_identical_tokens_masks_and_independent_caches(tiny_model):
    class RecordingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)

        def __call__(self, **kwargs):
            self.calls.append(
                {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            )
            return self.model(**kwargs)

    target = RecordingModel(tiny_model)
    reference = RecordingModel(deepcopy(tiny_model))
    decode_batch(
        {"dolos": target, "qwen": reference},
        torch.tensor([[0, 0, 3], [4, 5, 6]]),
        torch.tensor([[0, 0, 1], [1, 1, 1]]),
        arm={"target": "dolos", "reference": "qwen", "beta": 1, "alpha": 0.1},
        max_new_tokens=3,
        temperature=0,
        eos_token_ids=[],
        pad_token_id=0,
        seeds=[1, 2],
    )
    assert len(target.calls) == len(reference.calls) == 3
    for i, (d, q) in enumerate(zip(target.calls, reference.calls, strict=True)):
        for name in ["input_ids", "attention_mask", "position_ids"]:
            assert torch.equal(d[name], q[name])
        assert d["input_ids"].shape[1] == (3 if i == 0 else 1)
        if i:
            assert d["past_key_values"] is not q["past_key_values"]
    assert target.calls[0]["position_ids"].tolist() == [[1, 1, 0], [0, 1, 2]]


def test_resume_rejects_changed_source_data_and_outputs(tmp_path, monkeypatch):
    import dolos_audit.contrastive as runner
    from dolos_audit.io import file_sha256, write_json, write_jsonl

    root = tmp_path / "source"
    (root / "dolos_audit").mkdir(parents=True)
    code = root / "dolos_audit" / "fake.py"
    code.write_text("original source\n")
    write_json(root / "configs/models.json", {"pinned": "original"})
    config = {
        "batch_size": 2,
        "max_new_tokens": 3,
        "max_input_tokens": 8,
        "timeout_s": 10,
        "temperature": 0,
        "models": ["dolos", "qwen"],
        "seeds": [1],
        "split": "smoke",
        "arms": [{"id": "d", "target": "dolos", "reference": "qwen", "beta": 1, "alpha": 0.1}],
    }
    config_path, prompts_path = tmp_path / "config.json", tmp_path / "prompts.jsonl"
    write_json(config_path, config)
    write_jsonl(prompts_path, [{"id": "p", "split": "smoke", "raw_prefix": "The"}])
    destination = tmp_path / "out"
    monkeypatch.setattr(runner, "ROOT", root)

    def execute(_config, _prompts, _pins, out, meta):
        write_jsonl(out / "outputs.jsonl", [])
        meta.update(state="complete", outputs_sha256=file_sha256(out / "outputs.jsonl"))
        write_json(out / "run.json", meta)
        return meta

    monkeypatch.setattr(runner, "_execute", execute)
    assert runner.run(config_path, prompts_path, destination)["state"] == "complete"
    assert runner.run(config_path, prompts_path, destination, resume=True)["state"] == "complete"
    code.write_text("changed source\n")
    with pytest.raises(ValueError, match="unsafe resume"):
        runner.run(config_path, prompts_path, destination, resume=True)
    code.write_text("original source\n")
    (destination / "outputs.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="outputs changed"):
        runner.run(config_path, prompts_path, destination, resume=True)


def test_config_rejects_confirmation_and_unknown_arm_model():
    config = {
        "batch_size": 5,
        "max_new_tokens": 300,
        "max_input_tokens": 32,
        "timeout_s": 2700,
        "temperature": 1,
        "models": ["dolos", "qwen"],
        "seeds": [1000],
        "split": "discovery",
        "arms": [{"id": "d_cdd", "target": "dolos", "reference": "qwen", "beta": 1, "alpha": 0.1}],
    }
    validate_contrastive_config(config)
    config["arms"][0]["reference"] = "unrecognized"
    with pytest.raises(ValueError, match="unloaded"):
        validate_contrastive_config(config)
    config["arms"][0]["reference"] = "qwen"
    config["split"] = "confirmation"
    config["confirmation_lock"] = "unused"
    with pytest.raises(ValueError, match="exploratory"):
        validate_contrastive_config(config)
