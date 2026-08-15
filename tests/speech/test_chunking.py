"""Tests for sentence chunking used by incremental speech synthesis."""

from __future__ import annotations

import pytest

from openjarvis.speech.chunking import (
    SentenceAccumulator,
    split_for_speech,
)


def _words(text: str) -> list[str]:
    return text.split()


class TestSplitting:
    def test_splits_on_sentence_terminators(self) -> None:
        out = split_for_speech(
            "Encontrei tres opcoes para voce. A primeira e a mais barata. "
            "Quer que eu detalhe cada uma delas agora?"
        )
        assert len(out) == 3
        assert out[0].endswith(".")
        assert out[-1].endswith("?")

    def test_no_words_are_lost_or_duplicated(self) -> None:
        text = (
            "Rodei os testes e dois falharam. Corrigi o primeiro! "
            "O segundo depende do ambiente... Posso seguir?"
        )
        assert _words(" ".join(split_for_speech(text))) == _words(text)

    def test_empty_input(self) -> None:
        assert split_for_speech("") == []
        assert split_for_speech("   \n  ") == []

    def test_single_short_sentence_is_still_returned(self) -> None:
        """Merging happens between chunks; nothing is ever dropped."""
        assert split_for_speech("Sim.") == ["Sim."]


class TestAudibleMistakes:
    """A wrong split is heard, so these are the cases that matter."""

    @pytest.mark.parametrize(
        "text",
        [
            "O Dr. Silva confirmou a reuniao de amanha cedo com a equipe.",
            "Custa R$ 1.500,00 no total segundo a ultima cotacao recebida.",
            "O valor de pi e 3.14 e isso basta para o calculo pedido aqui.",
            "Acesse openjarvis.io para ver a documentacao completa do projeto.",
            "Falei com J. Saad sobre o roadmap da proxima versao do produto.",
        ],
    )
    def test_does_not_break_mid_token(self, text: str) -> None:
        assert split_for_speech(text) == [text]

    def test_ellipsis_stays_with_its_sentence(self) -> None:
        out = split_for_speech("Deixa eu pensar... Achei a resposta certa.")
        assert out[0].endswith("...")
        assert len(out) == 2

    def test_closing_quote_stays_attached(self) -> None:
        out = split_for_speech(
            'Ele disse "vou chegar as oito." Depois desligou o telefone.'
        )
        assert out[0].endswith('."')

    def test_runt_is_merged_not_emitted(self) -> None:
        """Two-word fragments synthesise clipped; merge them forward."""
        out = split_for_speech("Ok. Vou rodar a suite completa agora mesmo.")
        assert all(len(c) >= 10 for c in out)

    def test_overlong_sentence_is_broken_at_a_clause(self) -> None:
        text = (
            "Primeiro eu rodei a suite inteira, depois isolei o teste que "
            "falhava, em seguida comparei os conjuntos de falhas, e por fim "
            "confirmei que a regressao vinha da outra branch e nao da minha"
        )
        out = split_for_speech(text, max_chars=80)
        assert len(out) > 1
        assert all(len(c) <= 100 for c in out)
        assert _words(" ".join(out)) == _words(text)


class TestStreaming:
    """Chunks must not change once released — audio already played cannot."""

    def _drive(self, acc: SentenceAccumulator, text: str, step: int = 7):
        released = []
        for i in range(0, len(text), step):
            released.extend(acc.feed(text[i : i + step]))
        released.extend(acc.flush())
        return released

    def test_streamed_matches_batch(self) -> None:
        text = (
            "Encontrei o problema. Estava no gerenciamento de estado. "
            "Corrigi e os testes passaram."
        )
        acc = SentenceAccumulator()
        assert _words(" ".join(self._drive(acc, text))) == _words(text)

    def test_first_chunk_arrives_before_the_end(self) -> None:
        """The whole point: speak sentence one while sentence two generates."""
        acc = SentenceAccumulator()
        early = list(acc.feed("Ja terminei a primeira parte do trabalho. "))
        assert early, "a complete sentence should be released immediately"
        assert "terminei" in early[0]

    def test_holds_back_an_ambiguous_terminator(self) -> None:
        """`1.` may still become `1.5`; releasing early would mispronounce it."""
        acc = SentenceAccumulator()
        assert list(acc.feed("O valor e 3.")) == []
        assert list(acc.feed("14 exatamente conforme o esperado. ")) == [
            "O valor e 3.14 exatamente conforme o esperado."
        ]

    def test_flush_releases_an_unterminated_tail(self) -> None:
        acc = SentenceAccumulator()
        list(acc.feed("Resposta sem ponto final no fim"))
        assert acc.flush() == ["Resposta sem ponto final no fim"]

    def test_flush_is_idempotent(self) -> None:
        acc = SentenceAccumulator()
        list(acc.feed("Alguma coisa aqui."))
        acc.flush()
        assert acc.flush() == []

    def test_long_run_without_punctuation_still_releases(self) -> None:
        """A model that forgets punctuation must not block audio forever."""
        acc = SentenceAccumulator(max_chars=60)
        text = "palavra " * 40
        released = self._drive(acc, text)
        assert len(released) > 1
