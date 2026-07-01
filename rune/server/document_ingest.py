"""Ingestion de documents depuis l'UI — réutilise ``ingest.py``.

``ingest.py`` est un script CLI conçu pour être lancé à la main avec
des chemins de fichiers. Ce module enveloppe ses fonctions pour
qu'elles soient appelables depuis un endpoint HTTP (fichier uploadé
en mémoire, pas sur disque).

Deux modes :

- **attach** — extrait le texte et le retourne au client. Le texte
  sera injecté en contexte dans le prochain message. Pas de
  persistance.
- **ingest** — extrait, chunk, embed dans ChromaDB (collection
  ``lythea_memory``, type ``knowledge``). En plus, extrait les
  entités importantes du document et les ajoute au Knowledge Graph
  via ``hippocampe.entity_extractor`` + ``hippocampe.kg`` —
  exactement comme la Phase A de Lythéa traite un message
  utilisateur, mais étendu à un document. C'est ce qui permet à
  Rune de "connaître" les noms du document dans toute conversation
  future, même sans déclencher le RAG.

Pour les **gros documents**, l'extraction d'entités se fait par
**échantillonnage** plutôt que sur le texte intégral : passer 200
pages à GLiNER bloquerait l'upload pendant 30+ secondes. On échan-
tillonne au début, au milieu et à la fin (chacun ~3000 caractères) —
suffisant pour capturer les entités principales, qui sont presque
toujours mentionnées plusieurs fois dans des documents structurés.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("lythea.server.document_ingest")


# ── Échantillonnage pour l'extraction d'entités ────────────────────────
# ── Échantillonnage pour l'extraction d'entités ────────────────────────
# Cap global avant échantillonnage. Avec le NER chunking (~1200 chars
# par appel GLiNER, ~10ms/chunk sur CPU), on peut traiter beaucoup plus
# de texte qu'auparavant. Seuil élargi à 50K pour bien couvrir les
# rapports usuels ; au-delà, on échantillonne pour borner le coût
# (50K chars / 1200 = ~42 chunks ≈ 0.5s NER, ok).
_NER_SAMPLE_THRESHOLD: int = 50_000
# Taille de chaque échantillon (début / milieu / fin) en caractères.
_NER_SAMPLE_SIZE: int = 16_000

# Taille max d'un chunk envoyé à GLiNER. GLiNER tronque silencieusement
# à 384 tokens (~1500 chars en français), donc on chunke à cette taille
# avant d'appeler ``extract()`` — sinon on rate 99% des entités sur les
# gros documents. Voir l'avertissement « Sentence of length N has been
# truncated to 384 » dans les logs avant ce fix.
_NER_CHUNK_SIZE: int = 1200
_NER_CHUNK_OVERLAP: int = 100


def _sample_text_for_ner(text: str) -> str:
    """Échantillonne un texte long pour l'extraction d'entités.

    Pour les petits documents (<seuil), renvoie tout. Sinon, prend
    le début, le milieu et la fin — pondéré début/fin parce que
    les rapports et thèses concentrent les entités importantes dans
    l'intro, le titre, et la conclusion.
    """
    if len(text) <= _NER_SAMPLE_THRESHOLD:
        return text

    n = len(text)
    head = text[:_NER_SAMPLE_SIZE]
    mid_start = (n - _NER_SAMPLE_SIZE) // 2
    mid = text[mid_start:mid_start + _NER_SAMPLE_SIZE]
    tail = text[-_NER_SAMPLE_SIZE:]
    return f"{head}\n\n[...]\n\n{mid}\n\n[...]\n\n{tail}"


def _extract_entities_chunked(
    text: str,
    entity_extractor: Any,
) -> list[dict[str, Any]]:
    """Extrait les entités d'un texte en le découpant pour GLiNER.

    GLiNER tronque silencieusement à 384 tokens (~1500 chars). Passer
    un texte long en un seul appel rate la quasi-totalité des entités.
    Cette fonction :

    1. Découpe le texte en chunks de ~``_NER_CHUNK_SIZE`` chars
       (frontières naturelles via ``ingest.chunk_text``).
    2. Appelle ``entity_extractor.extract()`` sur chaque chunk.
    3. Dédup les entités : même ``text`` (lowercased) + même
       ``label`` → on garde le score max et compte les occurrences.

    Le coût total scale linéairement avec la longueur du texte mais
    reste rapide (GLiNER fait ~10ms/chunk sur CPU).
    """
    if not text or entity_extractor is None:
        return []

    # Import local pour éviter de charger ingest.py au top-level.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import ingest  # noqa: E402

    chunks = ingest.chunk_text(
        text, size=_NER_CHUNK_SIZE, overlap=_NER_CHUNK_OVERLAP,
    )
    if not chunks:
        # Texte trop court pour chunk_text (< MIN_CHUNK_SIZE) — extract
        # en un coup, c'est forcément sous le seuil GLiNER.
        try:
            return entity_extractor.extract(text)
        except Exception as exc:
            log.warning("GLiNER extract failed on short text: %s", exc)
            return []

    # Dédup : clé = (text.lower(), label). Valeur = entité avec score max
    # et compteur d'occurrences (peut servir plus tard pour pondérer).
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for chunk_text, _offset in chunks:
        try:
            raw = entity_extractor.extract(chunk_text)
        except Exception as exc:
            log.warning("GLiNER chunk extract failed: %s", exc)
            continue
        for ent in raw:
            key = (ent["text"].lower().strip(), ent["label"])
            if key in deduped:
                # Garde le score max — souvent un chunk capture mieux
                # qu'un autre (contexte plus riche).
                if ent.get("score", 0) > deduped[key].get("score", 0):
                    deduped[key]["score"] = ent["score"]
                deduped[key]["_count"] = deduped[key].get("_count", 1) + 1
            else:
                ent_copy = dict(ent)
                ent_copy["_count"] = 1
                deduped[key] = ent_copy

    log.info(
        "GLiNER chunked extract: %d chunks → %d unique entities",
        len(chunks), len(deduped),
    )
    return list(deduped.values())


# ── Extraction de texte depuis un upload ───────────────────────────────

# Extensions supportées — alignées sur ingest.extract_text.
# Documents bureautiques : pdf, docx
# Texte brut : txt, md, markdown, rst
# Web : html, htm
# Données structurées : csv, json, xml
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md", ".markdown", ".docx", ".rst",
    ".html", ".htm", ".csv", ".json", ".xml",
    # V6.0.0-rc rev5 — Excel (extraction par feuille via openpyxl,
    # rendu en TSV). Voir ingest.extract_xlsx pour les détails.
    ".xlsx",
})


def is_supported(filename: str) -> bool:
    """Renvoie True si l'extension du fichier est supportée."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def extract_uploaded_document(
    file_bytes: bytes,
    filename: str,
) -> tuple[str, int]:
    """Extrait le texte d'un fichier uploadé.

    On écrit le fichier dans un répertoire temporaire pour réutiliser
    les fonctions d'``ingest.py`` qui attendent un ``Path``. Le fichier
    temporaire est nettoyé automatiquement à la sortie du ``with``.

    Returns
    -------
    tuple[str, int]
        ``(full_text, n_chars)``. Le texte est déjà assemblé via
        ``assemble_document`` d'ingest.py (recollage des césures,
        séparation propre entre pages).
    """
    # Import local pour ne pas plomber l'import du module quand ingest.py
    # a des deps optionnelles (pdfplumber/python-docx) absentes en sandbox.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import ingest  # noqa: E402

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extension non supportée : {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = Path(tmp.name)

        # extract_text aiguille vers extract_pdf / extract_docx / etc.
        pages = ingest.extract_text(path)

    if not pages:
        return "", 0

    # Assemblage propre (recollage des césures inter-pages).
    full_text, _offsets = ingest.assemble_document(pages)
    return full_text, len(full_text)


# ── Ingestion complète (RAG + KG) ──────────────────────────────────────

def ingest_document_to_memory(
    file_bytes: bytes,
    filename: str,
    hippocampe: Any,
    tag: str | None = None,
) -> dict[str, Any]:
    """Ingère un document dans la mémoire long-terme (RAG + KG).

    Étapes :

    1. Extrait le texte du document.
    2. Chunke et ingère dans ChromaDB (collection
       ``lythea_memory``, type ``knowledge``) via ``ingest.py``.
    3. Échantillonne le texte pour l'extraction d'entités.
    4. Extrait les entités via ``hippocampe.entity_extractor``
       (GLiNER) et les upsert dans le KG via
       ``hippocampe.kg.upsert_entity``.

    Si une étape échoue, on continue les suivantes — le RAG seul
    reste utile même sans KG enrichi.

    Returns
    -------
    dict
        ``{filename, n_chars, n_chunks, n_entities, entities: [...]}``
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import ingest  # noqa: E402

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extension non supportée : {suffix}")

    result: dict[str, Any] = {
        "filename": filename,
        "n_chars": 0,
        "n_chunks": 0,
        "n_entities": 0,
        "entities": [],
    }

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = Path(tmp.name)

        # ── 1+2 : Extraction + ingestion ChromaDB ──────────────────────
        try:
            collection = ingest.get_collection()
            n_chunks, _replaced = ingest.ingest_file(collection, path, tag=tag)
            result["n_chunks"] = n_chunks
        except Exception as exc:
            log.error("Document ingest into ChromaDB failed: %s", exc)
            raise

        # ── 3 : Extraction du texte pour le NER (et le retour client) ─
        pages = ingest.extract_text(path)

    if not pages:
        return result
    full_text, _ = ingest.assemble_document(pages)
    result["n_chars"] = len(full_text)

    # ── 4 : Extraction d'entités + ajout au KG ─────────────────────────
    if hippocampe is None or hippocampe.entity_extractor is None:
        log.info("KG enrichment skipped: no entity extractor available")
        return result

    # Échantillonnage haut-niveau (cap à 50K chars pour borner le coût
    # sur les thèses/livres), puis chunking fin (~1200 chars) pour
    # passer sous le seuil de troncation de GLiNER (384 tokens). Sans
    # ce chunking, GLiNER ne voit que les 1500 premiers chars du texte
    # passé et rate 99% des entités sur les gros documents.
    sampled = _sample_text_for_ner(full_text)
    try:
        raw_entities = _extract_entities_chunked(
            sampled, hippocampe.entity_extractor,
        )
    except Exception as exc:
        log.warning("Entity extraction on document failed: %s", exc)
        return result

    if not raw_entities:
        return result

    # Filtrage léger comme dans encoding._extract_entities.
    kept: list[dict[str, Any]] = []
    for ent in raw_entities:
        text_norm = ent.get("text", "").lower().strip()
        if len(text_norm) < 2:
            continue
        kept.append(ent)

    # Upsert dans le KG.
    upserted: list[dict[str, Any]] = []
    for ent in kept:
        try:
            eid = hippocampe.kg.upsert_entity(
                value=ent["text"],
                entity_type=ent["label"],
                confidence=ent.get("score", 0.5),
            )
            upserted.append({
                "id": eid,
                "value": ent["text"],
                "type": ent["label"],
            })
        except Exception as exc:
            log.warning("KG upsert failed for %r: %s",
                        ent.get("text"), exc)

    result["n_entities"] = len(upserted)
    result["entities"] = upserted[:20]  # cap pour le retour client
    log.info(
        "Document '%s' ingested: %d chunks, %d entities → KG",
        filename, result["n_chunks"], result["n_entities"],
    )
    return result
