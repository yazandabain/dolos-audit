"""A large formatting effect must not override the frozen quality guard."""

import pytest

from dolos_audit.io import file_sha256, write_json, write_jsonl
from dolos_audit.scoring import score_output
from dolos_audit.transfer_analysis import analyze


@pytest.mark.parametrize("wrong,flag", [(True, False), (False, True)])
def test_low_quality_is_not_reclassified_as_successful_transfer(tmp_path, wrong, flag):
    prompts = []
    outputs = []
    for i in range(4):
        for condition in ["neutral", "evaluation"]:
            pid = f"synthetic-{i}-{condition}"
            expected = {"kind": "json_answer", "value": 1}
            prompts.append(
                {
                    "id": pid,
                    "context_id": f"synthetic-{i}",
                    "family": "extraction",
                    "template": "synthetic",
                    "condition": condition,
                    "expected": expected,
                }
            )
            text = (
                '```json\n{"answer": '
                + ("2" if wrong and i == 0 and condition == "neutral" else "1")
                + "}\n```"
            )
            outputs.append(
                {
                    "id": pid,
                    "prompt_id": pid,
                    "context_id": f"synthetic-{i}",
                    "model": "dolos",
                    "intervention": "identity",
                    "seed": 42,
                    "family": "extraction",
                    "condition": condition,
                    "output": text,
                    "score": score_output(text, expected),
                    "max_tokens_reached": False,
                }
            )
    write_jsonl(tmp_path / "prompts.jsonl", prompts)
    write_jsonl(tmp_path / "outputs.jsonl", outputs)
    write_json(
        tmp_path / "run.json",
        {
            "state": "complete",
            "run_id": "SYNTHETIC-TEST",
            "config": {"models": ["dolos"], "seeds": [42], "split": "confirmation"},
            "outputs_sha256": file_sha256(tmp_path / "outputs.jsonl"),
            "models_loaded": {"dolos": {}},
            "reference_weights_used_for_interventions": False,
            "protocol_payload_sha256": "SYNTHETIC-LOCK",
        },
    )
    r = analyze(tmp_path)
    assert r["format_recovery_rate"] > 0.25
    assert r["format_risk_flag"] is flag
    assert r["base_weights_loaded"] is False
