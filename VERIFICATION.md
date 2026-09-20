# Verification report

Verified locally on Apple Silicon using Python 3.12.14. All packages are installed in project-local `.venv`; its base interpreter is in ignored `.python/`. Credentials were later supplied and moved from `.env.example` into ignored, mode-0600 `.env`; the example file was restored to placeholders.

## Results

| Check | Result |
| --- | --- |
| `make test` | **118 passed, 5 skipped** |
| `.venv/bin/python scripts/check_setup.py --smoke` | Exit 0; all runtime imports passed |
| `.venv/bin/python -m compileall -q src scripts app.py tests` | Exit 0 |
| `.venv/bin/python -m pip check` | Exit 0; no broken requirements |
| Missing-credential `check_setup.py` | Expected exit 1, actionable variable-name guidance, no values |
| Missing-credential initialization/ingestion/pruning/storage scripts | Expected exit 1 with safe hosted PostgreSQL setup guidance |
| Streamlit AppTest | Setup screen and connected three-tab flow passed; delete controls disabled until confirmation |
| Infrastructure audit | No prohibited application imports or container artifacts; no ML frameworks, full LangChain metapackage, or unused AWS SDKs installed |
| Local footprint (`du -sh`) | `.venv`: 635 MB; `.python`: 72 MB; whole workspace: approximately 708 MB |

The pip check emitted only a sandbox cache-directory warning; pip caching was disabled and dependency compatibility passed.

## Live integration results

| Integration | Result |
| --- | --- |
| Gemini `gemini-embedding-2` | Passed: one real 768-dimensional embedding call |
| Gemini chat through in-process LiteLLM | Passed |
| OpenAI `gpt-4.1-mini` chat through in-process LiteLLM | Passed |
| LangSmith synthetic trace | Passed after correcting the hosted endpoint and allowing bounded readback delay |
| Supabase PostgreSQL / pgvector / PostgresSaver | Passed through the IPv4 Session pooler; schema and official checkpoint tables initialized |
| Live database integration test | Passed: vector storage, PostgreSQL checkpoint round trip, official `delete_thread()`, and CoALA memory preservation |
| Sample ingestion | Passed: 3 documents / 7 chunks added; second run skipped all 3 unchanged documents |
| End-to-end RAG | Passed: 5 retrieved chunks, Gemini simple route, critic accepted with 0 revisions, citations valid |
| Storage report | Passed: 11 MB database, both HNSW indexes present, warning threshold not exceeded |

The provider test command completed with 3 passed, the LangSmith command with 1 passed, and the database integration command with 1 passed. The end-to-end test used an isolated user/thread and removed its checkpoint, application-thread row, and memories afterward.

## Important verified behaviors

- Transaction fakes exercise the real ingestion write sequence and restore the previous document/chunks after a simulated insertion failure.
- Unchanged files skip embeddings; failed embeddings never enter replacement.
- Embedding batches wrap each chunk in an independent SDK `Content` object. A regression test uses the installed SDK transformation to confirm that Embedding 2 does not aggregate the chunks into one vector.
- All vectors are validated as finite, nonzero, and exactly 768-dimensional.
- Chat routing covers automatic/manual selection, one-provider operation, bounded fallback, and non-retryable errors.
- Actual LangGraph execution with an in-memory test saver confirms one completed checkpoint per invocation, one revision maximum, safe abstention, and persisted state restoration after graph recompilation.
- Real `add_messages` reducer execution confirms removal events retain exactly twelve recent messages across ten turns.
- Compaction removes full retrieved text, drafts, critic details, plans, and temporary responses.
- Thread tests confirm success-only activity updates, fifty-turn rollover, summary persistence before archival, and carrying only an episodic summary into the new thread.
- Cleanup tests confirm dry-run defaults, current-thread and summary exclusions, per-user thread ranking, the inactivity gate, deletion through the official API, preservation of memories, failure-safe tracking, invocation exclusion, and daily automatic-cleanup tracking.
- The connected Streamlit test checks the three tabs, chat submission, and unconfirmed deletion controls without connecting to external services.
- Known secret formats and configured credentials are redacted; detectable secret content is rejected by memory storage.

Default tests block socket connections. Live results above came only from the explicitly enabled integration and smoke-test commands.

## Live setup status

All requested live integrations are operational. Supabase uses the IPv4 Session pooler with SSL required; Gemini embeddings/chat, OpenAI chat, and LangSmith tracing were verified independently. LangSmith remains disabled by default in `.env` as required.

See README.md for the exact opt-in commands and setup instructions. Direct dependency versions are pinned in requirements.txt / requirements-dev.txt; requirements.lock records the full tested resolution.

## Delivered files

- `app.py`: three-tab Streamlit application, cached runtime resources, resume/rollover, confirmations, evidence, ingestion, memory, storage, and cleanup controls.
- `src/`: typed settings, security, data models, database initialization/pooling, Gemini embeddings, chunking/ingestion, retrieval, CoALA memory, model routing, agent prompts/graph, thread lifecycle, retention, storage monitoring, and runtime/service composition.
- `sql/schema.sql`: application tables, metadata, constraints, and indexes.
- `scripts/`: setup checks, database initialization, sample ingestion, checkpoint pruning, and storage report.
- `sample_docs/`: three original fictional source documents plus eleven source-grounded sample questions.
- `tests/`: credential-free unit/graph/UI tests and explicitly gated integration tests.
- `.env.example`, `.gitignore`, `.streamlit/config.toml`, pinned requirements, lockfile, pytest configuration, Makefile, README, and this report.

The repository was empty at the start; no unrelated files were overwritten.
