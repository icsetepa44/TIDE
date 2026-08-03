"""rag_lib — the runtime modules needed to run build_vector_db.py and RAG.py.

Contents (nothing here that the two drivers don't import):
  config.py          central paths / constants (BASE = the OpenRouter/ folder)
  key_manager.py     plaintext API-key loading + rate-limit-aware key rotation
  embedding_utils.py Gemini embedding + ChromaDB collection helpers
  rag_retriever.py   query-driven retrieval over the customs_corpus vector DB
  llm_clients.py     OpenRouter (models under test) + Gemini (judge seat) callers
  eval_lib.py        prompts, parsers, result store, rubric-v2 metrics, summaries

The two drivers (build_vector_db.py, RAG.py) and the shared judge modules
(llm_council.py, rubric_scoring.py) live one level up, in OpenRouter/.
"""
