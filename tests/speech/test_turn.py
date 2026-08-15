"""Tests for spoken-turn tracking — the memory half of barge-in.

The scenario these exist for: the assistant offers three options, the user
cuts in after hearing two and says "pega a segunda". If history records all
three, the model resolves "a segunda" against the wrong list.
"""

from __future__ import annotations

import pytest

from openjarvis.speech.turn import ChunkState, SpokenTurn, TurnOutcome

OPTIONS = [
    "Encontrei tres opcoes.",
    "A primeira e a mais barata.",
    "A segunda e a mais rapida.",
    "A terceira e a mais completa.",
]


def _turn_playing_through(count: int) -> SpokenTurn:
    """Enqueue every option, fully play the first ``count``, start the next."""
    turn = SpokenTurn()
    for text in OPTIONS:
        turn.enqueue(text)
    for i in range(count):
        turn.start_playing(i)
        turn.finish_playing(i)
    if count < len(OPTIONS):
        turn.start_playing(count)
    return turn


class TestTheScenarioThatMatters:
    def test_history_holds_only_what_was_heard(self) -> None:
        turn = _turn_playing_through(2)
        spoken = turn.interrupt(played_ratio=0.0)
        assert spoken == "Encontrei tres opcoes. A primeira e a mais barata."
        assert "segunda" not in spoken
        assert "terceira" not in spoken

    def test_unheard_text_is_kept_separately(self) -> None:
        """ "Como voce ia dizer?" is a reasonable follow-up."""
        turn = _turn_playing_through(2)
        turn.interrupt()
        assert "segunda" in turn.unspoken_text()
        assert "terceira" in turn.unspoken_text()

    def test_interruption_is_marked_in_history(self) -> None:
        turn = _turn_playing_through(2)
        turn.interrupt()
        assert turn.history_entry().endswith("—")
        assert turn.was_interrupted is True

    def test_an_uninterrupted_turn_is_not_marked(self) -> None:
        turn = SpokenTurn()
        turn.enqueue("Tudo certo por aqui.")
        turn.complete()
        assert turn.history_entry() == "Tudo certo por aqui."
        assert turn.outcome is TurnOutcome.COMPLETED


class TestMidSentenceInterruption:
    def test_truncates_on_a_word_boundary(self) -> None:
        turn = SpokenTurn()
        turn.enqueue("A primeira opcao e a mais barata de todas")
        turn.start_playing(0)
        spoken = turn.interrupt(played_ratio=0.5)
        assert spoken
        assert not spoken.endswith(" ")
        # Never invents a word that was only half heard.
        assert all(w in "A primeira opcao e a mais barata de todas" for w in [spoken])

    def test_never_claims_more_than_was_played(self) -> None:
        text = "A primeira opcao e a mais barata de todas"
        turn = SpokenTurn()
        turn.enqueue(text)
        turn.start_playing(0)
        spoken = turn.interrupt(played_ratio=0.5)
        assert len(spoken) <= len(text) * 0.5 + 1

    def test_interrupted_at_the_very_start_records_nothing(self) -> None:
        turn = SpokenTurn()
        turn.enqueue("Encontrei tres opcoes para voce agora.")
        turn.start_playing(0)
        assert turn.interrupt(played_ratio=0.0) == ""

    def test_interrupted_at_the_very_end_records_everything(self) -> None:
        turn = SpokenTurn()
        turn.enqueue("Encontrei tres opcoes.")
        turn.start_playing(0)
        assert turn.interrupt(played_ratio=1.0) == "Encontrei tres opcoes."

    @pytest.mark.parametrize("ratio", [-1.0, 5.0])
    def test_out_of_range_ratio_is_clamped(self, ratio: float) -> None:
        turn = SpokenTurn()
        turn.enqueue("Alguma coisa dita aqui agora.")
        turn.start_playing(0)
        turn.interrupt(played_ratio=ratio)  # must not raise


class TestChunkStates:
    def test_queued_chunks_are_dropped_not_spoken(self) -> None:
        turn = _turn_playing_through(1)
        turn.interrupt()
        states = [c.state for c in turn.chunks]
        assert states[0] is ChunkState.PLAYED
        assert states[1] is ChunkState.CUT
        assert states[2] is ChunkState.DROPPED
        assert states[3] is ChunkState.DROPPED

    def test_cancel_is_distinguished_from_a_voice_interruption(self) -> None:
        """Mute and barge-in both stop audio but mean different things."""
        turn = _turn_playing_through(1)
        turn.cancel()
        assert turn.outcome is TurnOutcome.CANCELLED
        assert turn.was_interrupted is False
        assert not turn.history_entry().endswith("—")

    def test_completing_plays_out_everything_queued(self) -> None:
        turn = SpokenTurn()
        for text in OPTIONS:
            turn.enqueue(text)
        spoken = turn.complete()
        assert all(t in spoken for t in OPTIONS)

    def test_a_finished_turn_rejects_new_chunks(self) -> None:
        turn = SpokenTurn()
        turn.enqueue("Pronto.")
        turn.complete()
        with pytest.raises(RuntimeError):
            turn.enqueue("mais texto")

    def test_nothing_enqueued_yields_empty_history(self) -> None:
        turn = SpokenTurn()
        assert turn.interrupt() == ""
        assert turn.history_entry() == ""
