"""Synthetic saved-run fixture checks pairing, controls, and the capability-damage decision."""

from dolos_audit.confirmation_analysis import RANDOM_CONTROLS, analyze
from dolos_audit.io import file_sha256, write_json, write_jsonl
from dolos_audit.scoring import score_output


def test_contrasts_separate_format_recovery_from_capability_damage(tmp_path):
    prompts, outputs = [], []
    arms = [("dolos", "identity"), ("qwen", "identity"), ("dolos", "restore27")]
    arms += [("dolos", r) for r in RANDOM_CONTROLS]
    for family in ["extraction", "capability_addition"]:
        for i in range(2):
            context = f"{family}-{i}"
            for condition in ["neutral", "evaluation"]:
                pid = f"{context}-{condition}"
                expected = {"kind": "json_answer", "value": 7}
                prompts.append(
                    {
                        "id": pid,
                        "context_id": context,
                        "family": family,
                        "condition": condition,
                        "template": family,
                        "expected": expected,
                    }
                )
                for model, intervention in arms:
                    text = '{"answer": 7}'
                    if family == "extraction" and model == "dolos":
                        if intervention in RANDOM_CONTROLS or (
                            intervention == "identity" and i == 0
                        ):
                            text = '```json\n{"answer": 7}\n```'
                    if family == "capability_addition" and intervention == "restore27" and i == 0:
                        text = '{"answer": 8}'
                    outputs.append(
                        {
                            "id": f"{pid}-{model}-{intervention}",
                            "prompt_id": pid,
                            "context_id": context,
                            "family": family,
                            "condition": condition,
                            "model": model,
                            "intervention": intervention,
                            "seed": 42,
                            "output": text,
                            "score": score_output(text, expected),
                            "max_tokens_reached": False,
                        }
                    )
    write_jsonl(tmp_path / "prompts.jsonl", prompts)
    write_jsonl(tmp_path / "outputs.jsonl", outputs)
    specs = [
        {"id": "identity", "kind": "identity", "models": ["dolos", "qwen"]},
        {"id": "restore27", "kind": "restore_attention", "layer": 27, "models": ["dolos"]},
    ]
    specs += [
        {"id": r, "kind": "permuted_attention_delta", "layer": 27, "seed": s, "models": ["dolos"]}
        for r, s in zip(RANDOM_CONTROLS, [61, 67, 71], strict=True)
    ]
    write_json(
        tmp_path / "run.json",
        {
            "run_id": "SYNTHETIC-SOFTWARE-TEST",
            "state": "complete",
            "config": {
                "models": ["dolos", "qwen"],
                "seeds": [42],
                "split": "confirmation",
                "interventions": specs,
            },
            "outputs_sha256": file_sha256(tmp_path / "outputs.jsonl"),
            "protocol_payload_sha256": "SYNTHETIC-TEST-LOCK",
        },
    )
    result = analyze(tmp_path)
    c = result["contrasts"]
    assert c["format_dolos_minus_qwen"]["estimate"] == -0.5
    assert c["accepted_dolos_minus_qwen"]["estimate"] == 0
    assert c["format_restore_minus_random_mean"]["estimate"] == 1
    assert c["capability_restore_minus_identity"]["estimate"] == -0.5
    assert result["capability_damage"]["adverse_contexts"] == 1
    assert not result["capability_damage"]["meets_damage_bound"]
