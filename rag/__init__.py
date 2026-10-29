"""Retrieval-Augmented Generation layer.

Public entry points:

* :class:`rag.document_loader.PDFDocumentLoader` - PDF/TXT/MD ingestion.
* :class:`rag.chunker.DocumentChunker` - overlap-aware splitting.
* :class:`rag.vector_store.VectorStoreManager` - FAISS persistence + search.
* :class:`rag.ingest.IngestionPipeline` - end-to-end indexing.
"""

from __future__ import annotations

__all__ = ["chunker", "document_loader", "embeddings", "ingest", "vector_store"]
