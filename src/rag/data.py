"""
Data utilities for RAG: Wikipedia document acquisition.
"""

import wikipediaapi


def get_wikipedia_documents():
    """
    Fetches relevant Wikipedia articles about Uruguay (history, culture, demography, key figures).
    Returns:
        List[str]: List of article texts.
    """
    page_titles = [
        "Historia_de_Uruguay",
        "Charrúas",
        "Cultura_de_Uruguay",
        "Demografía_de_Uruguay",
        "José_Gervasio_Artigas",
        "Batalla_de_Las_Piedras_(1811)",
        "Declaratoria_de_la_independencia_(Uruguay)",
        "Revolución_oriental",
    ]
    wiki = wikipediaapi.Wikipedia("PROYECTO_RAG", "es")
    docs = [wiki.page(nombre_web).text for nombre_web in page_titles]
    return docs
