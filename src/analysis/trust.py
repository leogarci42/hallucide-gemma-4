from __future__ import annotations

_truststore_injected = False


def ensure_system_trust_store() -> None:
    """Utilise le magasin de certificats natif de l'OS (Windows/macOS/Linux)
    au lieu du bundle Mozilla embarqué de `certifi`, que `httpx` (donc le SDK
    `mcp` et `litellm`) utilise par défaut. Nécessaire dès qu'un antivirus ou
    un proxy d'entreprise fait de l'inspection HTTPS : il installe son
    certificat racine dans le magasin OS, jamais dans celui de certifi --
    observé en direct avec Avast Antivirus (`httpx` rejette la connexion en
    CERTIFICATE_VERIFY_FAILED alors qu'`urllib`, qui fait confiance au magasin
    OS, l'accepte).

    Idempotent, sans effet néfaste si aucune inspection HTTPS n'est active.
    `truststore` est une dépendance optionnelle : en son absence, on retombe
    silencieusement sur le comportement certifi par défaut.
    """
    global _truststore_injected
    if _truststore_injected:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    _truststore_injected = True
