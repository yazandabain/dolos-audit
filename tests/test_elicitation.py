"""Rendering tests keep forced text distinguishable from generated continuations."""

from dolos_audit.elicitation import render


class TokenizerFixture:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"


def test_prefill_continues_the_assistant_turn_without_closing_it():
    p = {
        "render_mode": "chat",
        "messages": [{"role": "user", "content": "test"}],
        "assistant_prefill": "Hypothetically, ",
    }
    assert render(TokenizerFixture(), p) == "USER:test\nASSISTANT:Hypothetically, "


def test_raw_completion_does_not_add_chat_messages():
    assert (
        render(None, {"render_mode": "raw", "raw_prefix": "A fictional report:"})
        == "A fictional report:"
    )
