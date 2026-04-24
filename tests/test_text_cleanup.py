"""Tests for lightweight post-transcription text cleanup."""

from voice_typer.text_cleanup import clean_transcribed_text


class TestCleanTranscribedText:
    def test_removes_adjacent_duplicate_words(self):
        text = "this this is is a test test message"

        assert clean_transcribed_text(text) == "This is a test message."

    def test_removes_adjacent_duplicate_short_phrases(self):
        text = "right now right now I want to test this"

        assert clean_transcribed_text(text) == "Right now I want to test this."

    def test_fixes_punctuation_spacing(self):
        text = "hello , world ! this is working ?"

        assert clean_transcribed_text(text) == "Hello, world! This is working?"

    def test_capitalizes_sentence_starts_and_pronoun_i(self):
        text = "i tested this. it works and i like it"

        assert clean_transcribed_text(text) == "I tested this. It works and I like it."

    def test_adds_question_mark_for_question_openers(self):
        text = "can we make this faster"

        assert clean_transcribed_text(text) == "Can we make this faster?"

    def test_adds_question_mark_for_question_final_sentence(self):
        text = "i want to make this faster. can we do that"

        assert clean_transcribed_text(text) == "I want to make this faster. Can we do that?"

    def test_does_not_force_question_mark_for_plain_statement(self):
        text = "we can make this faster"

        assert clean_transcribed_text(text) == "We can make this faster."

    def test_preserves_existing_terminal_punctuation(self):
        text = "what do you think?"

        assert clean_transcribed_text(text) == "What do you think?"

    def test_empty_text_stays_empty(self):
        assert clean_transcribed_text("") == ""
        assert clean_transcribed_text("   ") == ""
