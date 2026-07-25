from __future__ import annotations

from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider

# Étage 1 (hackathon §5) : Gemma choisit un domaine dans une LISTE FERMÉE, ou
# répond AUCUN. C'est la première ligne de défense anti-hallucination : un
# système qui refuse quand il n'a pas la donnée démontre la thèse mieux qu'une
# réponse correcte (§5, "pourquoi le refus est un atout majeur").
DOMAINE_AUCUN = "AUCUN"


class DomainRouter:
    """Route une question vers un dataset_id parmi une liste fermée, via Gemma.

    Garde-fou déterministe (non négociable, §5) : toute réponse de Gemma qui
    n'est pas EXACTEMENT une clé de `domains` est forcée à AUCUN par le code
    -- Gemma ne peut jamais faire apparaître un dataset qui n'a pas été
    explicitement listé, même par erreur de formulation ou hallucination.

    `route()` distingue deux échecs différents, gérés différemment par
    l'appelant (§5, "les 3 chemins possibles") :
      - retour `None` : refus légitime (AUCUN domaine ne correspond, ou
        réponse hors liste) -> le système doit refuser de répondre.
      - `HallucideError` levée : panne technique (API indisponible, etc.)
        -> l'appelant retombe sur le dataset fixe par défaut (§5, "le routing
        ne doit jamais casser la chaîne").
    """

    def __init__(self, model_provider: ModelProvider, domains: dict[str, str]) -> None:
        # domains : dataset_id -> description courte (affichée à Gemma pour
        # l'aider à choisir, jamais utilisée pour la vérification -- seule la
        # clé compte pour le garde-fou).
        self.model_provider = model_provider
        self.domains = domains

    def route(self, question: str) -> str | None:
        if not self.domains:
            return None

        try:
            response = self.model_provider.generate(
                messages=self._build_prompt(question), tools=[], tool_choice=None
            )
        except Exception as exc:
            raise HallucideError(f"Domain routing failed: {exc}") from exc

        choice = self._extract_choice(response.get("text", ""))
        if choice == DOMAINE_AUCUN:
            return None
        if choice not in self.domains:
            # Garde-fou : réponse hors liste (formulation inattendue,
            # invention) forcée à AUCUN plutôt que tentée telle quelle.
            return None
        return choice

    def _build_prompt(self, question: str) -> list[dict[str, str]]:
        listing = "\n".join(f"- {key}: {desc}" for key, desc in self.domains.items())
        return [
            {
                "role": "system",
                "content": (
                    "Tu choisis un domaine de connaissance dans une liste FERMÉE pour "
                    "répondre à une question médicale.\n"
                    f"Domaines disponibles :\n{listing}\n\n"
                    "Réponds UNIQUEMENT par la clé exacte d'un domaine ci-dessus (pas la "
                    f"description, pas de phrase), ou par « {DOMAINE_AUCUN} » si aucun "
                    "domaine ne correspond à la question. Aucun autre texte."
                ),
            },
            {"role": "user", "content": question},
        ]

    def _extract_choice(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return DOMAINE_AUCUN
        # Un petit modèle peut ajouter guillemets/ponctuation malgré la
        # consigne "aucun autre texte" -- on ne garde que la première ligne
        # nettoyée, jamais une correspondance floue/partielle (le garde-fou
        # exact-match reste dans route(), pas ici).
        first_line = stripped.splitlines()[0].strip()
        return first_line.strip("\"'.` ")
