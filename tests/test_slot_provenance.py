from hallucide.verification.slot_provenance import check_slot_provenance


def test_slot_is_copied_when_present_verbatim_in_question() -> None:
    result = check_slot_provenance("Que dit l'article 1103 du Code civil ?", "article", "1103")
    assert result.copied is True
    assert result.inferred is False


def test_slot_is_copied_with_loose_formatting_differences() -> None:
    # "L. 1232-6" dans le slot vs "L1232-6" dans la question : même référence.
    result = check_slot_provenance("Que dit L1232-6 du code du travail ?", "article", "L. 1232-6")
    assert result.copied is True


def test_slot_is_inferred_when_absent_from_question() -> None:
    # Piège A3 : le LLM a deviné un numéro d'article absent de la question utilisateur.
    result = check_slot_provenance("Quelle est la règle sur la force obligatoire des contrats ?", "article", "1103")
    assert result.copied is False
    assert result.inferred is True


def test_empty_slot_value_is_not_copied() -> None:
    result = check_slot_provenance("Une question.", "article", "")
    assert result.copied is False


def test_short_number_not_copied_when_only_substring_of_another_number() -> None:
    # Régression A3 : l'article '16' inféré ne doit PAS être considéré copié
    # simplement parce que '16' est une sous-chaîne de l'année '2016'.
    result = check_slot_provenance("Quelle loi de 2016 modifie le contrat ?", "article", "16")
    assert result.copied is False
    assert result.inferred is True


def test_short_number_copied_when_present_as_whole_word() -> None:
    result = check_slot_provenance("Que dit l'article 6 du code civil ?", "article", "6")
    assert result.copied is True


def test_multi_word_slot_copied_when_all_tokens_present() -> None:
    result = check_slot_provenance("Que dit l'article 1103 du code civil ?", "code", "code civil")
    assert result.copied is True
