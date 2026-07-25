from hallucide.coverage.coverage import build_echo_back, check_coverage
from hallucide.core_types.types import Intent


def test_coverage_passes_when_intents_cover_message() -> None:
    message = "Quel est le délai de rétractation et quelle est la sanction en cas de non-respect ?"
    intents = [
        Intent(id="1", question="Quel est le délai de rétractation ?"),
        Intent(id="2", question="Quelle est la sanction en cas de non-respect du délai ?"),
    ]

    result = check_coverage(message, intents)

    assert result.passed is True
    assert result.ratio >= 0.8
    assert result.missing_tokens == ()


def test_coverage_fails_when_intent_forgotten() -> None:
    # Piège E4 : le message pose deux questions, la décomposition n'en garde qu'une.
    message = "Quel est le délai de rétractation et quelle est la sanction en cas de non-respect ?"
    intents = [Intent(id="1", question="Quel est le délai de rétractation ?")]

    result = check_coverage(message, intents)

    assert result.passed is False
    assert "sanction" in result.missing_tokens


def test_coverage_ignores_stopwords() -> None:
    message = "Le texte et la loi sont applicables."
    intents = [Intent(id="1", question="texte loi applicables")]

    result = check_coverage(message, intents)

    assert result.passed is True


def test_coverage_handles_empty_message() -> None:
    result = check_coverage("", [Intent(id="1", question="quoi ?")])
    assert result.passed is True
    assert result.ratio == 1.0


def test_coverage_detects_forgotten_single_digit_reference() -> None:
    # Régression : une référence à un chiffre isolé (article 2) ne doit pas
    # échapper au contrôle de couverture E4 -- avant le fix, len(t)>1 filtrait
    # tous les chiffres uniques et le ratio affichait 1.0 malgré l'oubli.
    message = "Article 6 et article 2 ?"
    intents = [Intent(id="1", question="Article 6 ?")]

    result = check_coverage(message, intents)

    assert "2" in result.missing_tokens
    assert result.passed is False


def test_coverage_keeps_single_digit_when_covered() -> None:
    message = "Que dit l'article 6 ?"
    intents = [Intent(id="1", question="Que dit l'article 6 ?")]

    result = check_coverage(message, intents)

    assert "6" not in result.missing_tokens
    assert result.passed is True


def test_echo_back_lists_each_intent() -> None:
    intents = [
        Intent(id="1", question="Quel est le délai ?"),
        Intent(id="2", question="Quelle est la sanction ?"),
    ]

    echo = build_echo_back(intents)

    assert "1. Quel est le délai ?" in echo
    assert "2. Quelle est la sanction ?" in echo
