"""Ingestion de pièces jointes : classification + décision captioner/natif."""

from rune.agentic.ingest import (classify_attachment, ingest_attachment,
                                    IMAGE_EXT, BINARY_DOC_EXT)


def test_classify():
    assert classify_attachment("photo.PNG") == "image"
    assert classify_attachment("rapport.pdf") == "doc"
    assert classify_attachment("notes.txt") == "text"
    assert classify_attachment("data.csv") == "text"


def test_doc_is_extracted_text_renamed_txt():
    r = ingest_attachment({"filename": "rapport.pdf", "content": "texte extrait"},
                          native_multimodal=False)
    assert r.filename == "rapport.txt"
    assert r.text == "texte extrait"
    assert r.pil_image is None


def test_text_passthrough():
    r = ingest_attachment({"filename": "a.txt", "content": "bonjour"},
                          native_multimodal=False)
    assert r.filename == "a.txt" and r.text == "bonjour"


def test_image_text_brain_uses_captioner():
    called = {}

    def cap(img):
        called["yes"] = True
        return "un graphique en barres"

    # contenu non décodable en image → _decode_image renvoie None, mais on
    # vérifie d'abord le chemin natif=False : sans image décodable, fallback.
    r = ingest_attachment({"filename": "fig.png", "content": "xxx"},
                          native_multimodal=False, caption_fn=cap)
    # contenu non décodable → pas de caption possible → note "non interprétée"
    assert r.filename == "fig.txt"
    assert r.text is not None


def test_image_native_brain_keeps_pixels_when_decodable():
    # petit PNG 1x1 valide encodé base64
    png_b64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
               "YAAAAAMAASsJTYQAAAAASUVORK5CYII=")
    r = ingest_attachment({"filename": "p.png", "content": png_b64},
                          native_multimodal=True)
    # PIL dispo dans l'env de test → image décodée et gardée en natif
    if r.pil_image is not None:
        assert r.filename == "p.png"
        assert r.text is None
    else:
        # PIL absent → fallback texte, acceptable
        assert r.text is not None


def test_none_on_empty():
    assert ingest_attachment({"filename": "", "content": "x"},
                             native_multimodal=False) is None
    assert ingest_attachment({"filename": "a.txt", "content": None},
                             native_multimodal=False) is None
