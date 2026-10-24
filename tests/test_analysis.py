"""Behavioral checks for the limited post hoc parser; no synthetic experiment results."""

import pytest

from dolos_audit.analysis import relaxed_closed_score


@pytest.mark.parametrize("text", ['{"answer": 7}', '```json\n{"answer": 7}\n```'])
def test_complete_json_fence_and_plain_object_have_the_same_answer(text):
    result = relaxed_closed_score(text, {"kind": "json_answer", "value": 7})
    assert result["format_valid"] and result["correct"]


@pytest.mark.parametrize(
    "text",
    [
        'Here is the answer: {"answer": 7}',
        '```json\n{"answer": 7}\n```\nActually this is wrong.',
        '```json\n{"answer": 7}',
        "7",
    ],
)
def test_prose_truncation_and_unstructured_values_remain_unparsed(text):
    result = relaxed_closed_score(text, {"kind": "json_answer", "value": 7})
    assert not result["format_valid"] and result["correct"] is None


def test_relaxing_format_never_relaxes_answer_type_or_schema():
    result = relaxed_closed_score(
        '```json\n{"answer": true}\n```', {"kind": "json_answer", "value": 1}
    )
    assert result["format_valid"] and result["correct"] is False
    result = relaxed_closed_score(
        '```json\n{"answer": 1,"extra": 0}\n```', {"kind": "json_answer", "value": 1}
    )
    assert not result["format_valid"] and result["correct"] is None
