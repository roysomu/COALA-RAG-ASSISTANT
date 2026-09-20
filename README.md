# Local CoALA RAG assistant

A runnable Python 3.12 / Streamlit demonstration of a bounded LangGraph agent workflow, hosted Gemini embeddings, Supabase or Neon PostgreSQL with pgvector, CoALA memory, and in-process LiteLLM routing. All application execution runs on your Mac. LangSmith is optional observability; it is never a model gateway or deployment platform.

The only runtime services are hosted PostgreSQL (Supabase or Neon), Gemini, OpenAI (when selected), and LangSmith (when enabled). There is no local database, container runtime, model server, or downloaded language/embedding model. Only `.venv` is used for Python packages.

## Quick start on macOS / Apple Silicon

The current workspace already includes a Python **3.12.14** `.venv`, with its interpreter in ignored project-local `.python/`. Do not move or delete `.python/` while using that environment. You can immediately run the commands below. On a fresh checkout, install Python 3.12 from [python.org](https://www.python.org/downloads/macos/) and follow the fresh-environment commands further down.

```bash
cd /Users/siddhartharoy/Documents/local-coala-rag-assistant
source .venv/bin/activate
cp .env.example .env
```

Edit `.env` locally (do not paste credentials into chat). Replace the placeholder `DATABASE_URL`, supply `GEMINI_API_KEY`, and optionally supply `OPENAI_API_KEY`. Leave tracing disabled initially. Then:

```bash
python scripts/check_setup.py
python scripts/init_db.py
python scripts/ingest_samples.py
streamlit run app.py
```

Open the loopback URL printed by Streamlit (normally `http://127.0.0.1:8501`). The first load creates a conversation. Refreshing or restarting resumes the most recently active thread; a `?thread=UUID` URL resumes that owned active thread explicitly. The Chat tab also lets you select existing active conversations. Restart Streamlit after changing `.env`, because resource clients are cached.

### Supabase or Neon setup

Create a hosted database and use a role allowed to create tables and install the `vector` extension. The application accepts official Supabase (`.supabase.co` or `.supabase.com`) and Neon (`.neon.tech`) hosts and requires `sslmode=require` or `sslmode=verify-full`.

For Supabase, click **Connect** in the project dashboard and select **Session pooler** when the Mac or network lacks IPv6. The direct `db.PROJECT.supabase.co` endpoint resolves to IPv6 by default; the shared session pooler is IPv4 and uses a `pooler.supabase.com` host on port 5432. Copy the complete generated connection string because its host cluster and username cannot be safely inferred. Percent-encode reserved password characters. Example shape only:

```dotenv
DATABASE_URL=postgresql://postgres.PROJECT:PASSWORD@aws-INDEX-REGION.pooler.supabase.com:5432/postgres?sslmode=require
```

For Neon, copy the pooled connection string from its Connect dialog. Example shape only:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@ep-example-pooler.REGION.aws.neon.tech/DATABASE?sslmode=require
```

Psycopg uses a pool of **0–5 connections**, autocommit, dictionary rows, and `prepare_threshold=None`. Application transactions and advisory locks are transaction-scoped and work with the supported poolers. A graph invocation holds one connection for its maintenance lock; remaining connections perform queries and checkpoint writes. This is sized for a local single-user app, not a multi-worker service.

Run `python scripts/init_db.py`. This idempotently creates pgvector and application tables, verifies the embedding identity, calls official **`PostgresSaver.setup()`**, and attempts cosine HNSW indexes. If HNSW creation fails, the script reports it; cosine retrieval still works through exact scans. If extension creation fails, check database permissions and project selection.

No local PostgreSQL installation is needed. Free-tier storage and compute quotas can change; use the diagnostics report instead of assuming an allowance.

### Provider keys

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey) and set `GEMINI_API_KEY`. Ensure the associated project has access and quota for `gemini-embedding-2` and the configured chat models. Embeddings always require Gemini, including when OpenAI supplies answers.

Create an OpenAI project API key in the [OpenAI platform](https://platform.openai.com/api-keys) and set `OPENAI_API_KEY` to enable complex-question routing to OpenAI. Configure project billing/limits as appropriate. A ChatGPT subscription is separate from API usage. The router can use either chat provider alone; the complete RAG app still requires Gemini embeddings. Gemini-only configuration is sufficient to run the full app.

No live calls are made by setup validation or by default tests. Ingestion, questions, and explicit live LLM tests consume provider quota and may incur charges.

### Fresh environment commands

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -r requirements-dev.txt
cp .env.example .env
# Fill in DATABASE_URL and GEMINI_API_KEY; optionally OPENAI_API_KEY.
python scripts/check_setup.py
python scripts/init_db.py
python scripts/ingest_samples.py
streamlit run app.py
```

All direct dependencies are pinned. `requirements.lock` records the entire tested dependency resolution, including development packages; for the same resolution use `python -m pip install --no-cache-dir -r requirements.lock`. Native wheels are required on Apple Silicon; no model weights are installed. Streamlit brings its normal dataframe/Arrow dependencies; LiteLLM brings tokenizer libraries, which are not local language models. The selected LiteLLM version is 1.80.17 because its base install avoids the unused AWS SDK introduced in the newer release examined during setup. No proxy extras are installed. The full LangChain metapackage and ML frameworks are absent; `langchain-core` is LangGraph's required lightweight message dependency.

`--no-cache-dir` avoids permanent wheel-cache growth. The verified environment is comfortably below the 2 GB local-storage budget; see `VERIFICATION.md` for measured sizes. Document and vector storage lives in hosted PostgreSQL. Large uploads and extraction have explicit limits to protect the 16 GB target machine.

## Architecture and data flow

```text
Uploads / samples → validate + extract → overlapping chunks
  → Gemini API embeddings (768) → transaction → hosted PostgreSQL + pgvector

START → supervisor → memory_retriever → document_retriever
        → answer_agent → critic_agent
              ↑              │
              └─ one revise ─┤
                             └─ accepted / revision exhausted
                                → memory_curator → compact_state → END
```

The supervisor uses deterministic complexity signals (prompt length, multiple questions, comparisons, analysis/design/diagnosis/synthesis, structured output). It records a short plan, routing reason, model, category, and RAG decision. Simple greetings skip document/memory retrieval. Auto mode uses `gemini/gemini-2.5-flash-lite` for simple requests and `openai/gpt-4.1-mini` for complex requests. The manual selector overrides classification. When only one chat provider is configured, it is used regardless of classification.

LiteLLM runs as a Python library and calls providers directly. A retryable timeout, connection failure, rate limit, or transient server error can trigger **one** call to `gemini/gemini-2.5-flash`, if that fallback is configured and differs from the primary. Authentication, permission, invalid-request, and missing-model errors are not retried. Internal LiteLLM retries are disabled. The UI shows the actual answer model as well as the original routing reason. Requested model names were retained; no silent model substitutions occur.

The answer agent receives only top-k evidence plus compact memory excerpts, recent messages, and a rolling summary. Documents and memories are explicitly treated as untrusted data. Documentary claims use inline citations such as `[security_policy.md, chunk 3]`. User memory and inference must be distinguished from documentary facts.

The critic returns validated JSON containing `decision`, `reason`, `unsupported_claims`, and `revision_instructions`. Code independently checks citation labels against retrieved evidence. A malformed critic result is treated as a rejection. The graph permits at most one revision; a second rejection produces an insufficient-evidence answer, not an unchecked draft. The recursion limit supplies an additional termination guard. This demonstrates grounded generation, not a guarantee that an LLM critic can prove every claim true.

## CoALA memory

| Category | Location | Purpose |
| --- | --- | --- |
| Working | LangGraph state / final checkpoints | Recent messages, rolling summary, current question, route, evidence, drafts, critic, revision count |
| Episodic | `memories`, type `episodic` | Summaries of prior interactions with user/thread provenance |
| Semantic | `memories`, type `semantic` | Explicit durable user facts and preferences, with provenance |
| Procedural | `memories`, type `procedural` | Reusable supported procedures and strategies |

Long-term memory is logically separate from checkpoints and has no foreign key that cascades from thread deletion. Retrieval filters by user, optionally by source thread, and retrieves up to two small excerpts per type. For substantive questions, the curator can propose up to four durable memories and attempts an episodic summary. Greetings and thanks reuse any existing summary without creating irrelevant durable memories or calling the curator model. Exact normalized content hashes prevent duplicates before spending another embedding call; a unique constraint also prevents concurrent duplicate inserts. This is content deduplication, not semantic-equivalence detection.

The local user ID is a scope filter, **not authentication**. Documents are shared across this local application database. Keep Streamlit bound to loopback; this is not a public multi-tenant service.

## Ingestion and retrieval

TXT and Markdown use UTF-8; `pypdf` extracts text by page. Encrypted or malformed PDFs are rejected. Scans without embedded text require external text extraction; no OCR/ML dependency is installed. Files are capped by `MAX_UPLOAD_MB` (default 10), 500 PDF pages, and 2 million extracted characters. Streamlit also enforces a 10 MB upload limit in `.streamlit/config.toml`; change both if you intentionally raise it.

The custom splitter uses approximately 900 characters and 150 characters of overlap, preferring paragraph boundaries. Chunks carry an index, page when known, offsets, hash, and document identity. Names are stripped of paths and restricted to safe characters; uploaded contents are never executed.

The SHA-256 of original file bytes makes unchanged uploads idempotent, including the same content under another name. A changed file with the same **sanitized filename** replaces the old chunks. All embeddings are obtained and validated first, then one transaction writes the document, chunks, and success record. A failed embedding or chunk insertion cannot partially replace the old document. Failed attempts have separate safe ingestion records and are never recorded as successful. The UI reports added/skipped/replaced/failed counts.

Gemini uses the official `google-genai` SDK with `EmbedContentConfig(output_dimensionality=768)`, batches of up to 32 separately wrapped `Content` objects (a plain string list would aggregate under Embedding 2), a 60-second timeout, and at most three attempts for transient errors. Every vector must contain exactly 768 finite values and be nonzero. Documents, queries, and memories all use the same model. Embedding 2 inputs use the documented search-query and document text prefixes; memories use the document format. Changing this input-format convention also requires re-embedding. Cosine search uses `<=>`; optional HNSW indexes accelerate it, with exact scanning available if no approximate index exists. Retrieval removes duplicate and heavily similar text and passes only the requested top-k chunks. Evidence references remain after compaction; the UI fetches full source text on demand. Deleted or replaced sources may no longer resolve from old citation IDs.

### Re-embedding and reset procedures

The `embedding_config` table binds a database to its embedding model and dimensionality. Changing the model requires **complete re-embedding of documents and all CoALA memories**; changing dimension additionally requires a schema/index migration. This application intentionally rejects dimensions other than 768 and never mixes embedding spaces or silently changes models.

For a clean migration, export any memories you wish to preserve, create a new dedicated hosted database or branch, point `.env` to it, set the intended supported embedding model, initialize the schema, and re-ingest the original documents and desired memories. Existing checkpoints are not copied. Keep the old database until the migration is verified. For a complete demonstration reset, use a new dedicated database rather than editing LangGraph internals. Deleting the old database is a separate, explicit destructive action in its provider console.

For a partial reset, use confirmed document/memory/conversation deletion in the UI. Conversation deletion calls `PostgresSaver.delete_thread()` and removes its application thread record only after success. It does not delete semantic, procedural, or episodic memories. Memory deletion has separate individual and confirmed bulk controls. Deleting long-term memories does not rewrite already-saved conversation checkpoints or previously exported LangSmith traces; remove the relevant conversations/traces separately if those retained excerpts must also be erased.

## Checkpoint durability, compaction, and rollover

Every application invocation calls the regular **`PostgresSaver`** with **`durability="exit"`**. Successful invocations save a final compact checkpoint, rather than writing after every node. The application does not implement mid-run resume: a crash can lose work since the last completed turn. LangGraph can also persist exit/error state depending on the failure; do not treat exit durability as crash recovery. Retry as a new invocation. External side effects (such as a memory insert) can precede a failed checkpoint and are deduplicated when retried.

The `compact_state` node runs immediately before END. It retains a summary capped at 2,000 characters, a final answer, small evidence IDs and source metadata, and essential route/revision status. It removes complete document/memory results, question copies, plans, draft answers, detailed critic output, temporary responses, and old messages. **`RemoveMessage` events** cooperate with LangGraph's `add_messages` reducer, keeping the latest **12 total messages** by default.

`app_threads.message_count` counts **successful user turns / invocations**, following the requested SQL increment-by-one policy (a turn normally adds a user and assistant message). It does not count both individual message objects. After **50 completed turns**, the service verifies/stores an episodic summary, marks `summary_saved_at`, archives the thread, and creates a fresh UUID. The next question loads only the prior rolling summary through episodic-memory retrieval; it never copies full history. The UI announces rollover. Old checkpoints remain until retention cleanup or explicit deletion. If summary persistence fails, rollover is refused and the existing conversation is kept.

## Safe retention and storage controls

Defaults: 30 inactive days, 20 retained threads per user, no automatic cleanup. Historical checkpoints still accumulate across completed turns, so exit durability does not replace retention.

All pruning candidates must be older than the inactivity threshold, exclude the current thread, and have a saved summary unless `--force` is explicitly used. Old archived threads beyond the per-user maximum are marked as excess candidates. **The 20-thread target never overrides the age, current-thread, or summary safety gates**: it is a soft storage target and can be exceeded while those protections apply. Stale unsummarized threads are reported for review, not silently deleted.

```bash
python scripts/prune_checkpoints.py
python scripts/prune_checkpoints.py --dry-run
python scripts/prune_checkpoints.py --apply
python scripts/prune_checkpoints.py --days 30 --apply
# More precise protection while Streamlit is open:
python scripts/prune_checkpoints.py --apply --current-thread YOUR_CURRENT_UUID
```

Default behavior is dry-run. Output includes candidate IDs, last-activity dates, checkpoint counts, and skipped reasons. `--apply` prints a preview before deleting and rechecks eligibility under the maintenance lock. Without `--current-thread`, the CLI conservatively protects **all active conversations** and only removes eligible archived ones. Supply `--current-thread NONE` only when Streamlit is closed and you intend to include inactive unarchived threads. `--user-id` restricts scope; `--force` bypasses the summary requirement only. It does not bypass the age/current-thread rules. Cleanup makes no LLM or embedding calls.

Only the official **`checkpointer.delete_thread(thread_id)`** removes checkpoint data, including its blobs/writes. Application code does not issue DELETE statements against LangGraph checkpoint tables. A failed deletion leaves the `app_threads` row intact; retries are safe. CoALA memory is always preserved by cleanup. Delete memories with the separate UI controls.

The UI requires a preview and confirmation for bulk checkpoint cleanup. Current and archived conversation deletion have separate confirmations. Database advisory locks exclude cleanup while any application graph invocation is active, and serialize invocations on the same thread. A concurrent cleanup fails safely or is skipped, rather than deleting an in-flight conversation.

Set `AUTO_PRUNE_CHECKPOINTS=true` to opt in. On a Streamlit rerun, automatic pruning uses the same protections and global lock, runs no more than once in 24 hours, and records its result in `maintenance_runs`. It is not a background daemon. Failed thread deletions are recorded and can be retried manually.

Run `python scripts/storage_report.py`, or click **Refresh storage report** in diagnostics, to inspect total PostgreSQL database size, document/chunk/memory table sizes, vector/index sizes, checkpoint/blob/write sizes, thread/checkpoint/document/chunk counts, and memory counts by type. Above `DATABASE_WARNING_MB=350` the UI shows recommendations. Crossing the threshold never itself deletes data. PostgreSQL may reuse freed table pages without immediately shrinking database files, so do not expect size to fall after every cleanup.

## Optional LangSmith tracing

Create a [LangSmith account](https://smith.langchain.com/), create a personal API key in its settings, and configure:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=YOUR_KEY_STORED_ONLY_IN_ENV
LANGSMITH_PROJECT=local-coala-rag-assistant
LANGSMITH_ENDPOINT=
```

A free Developer account is sufficient for normal testing; [free-tier quotas and retention periods](https://www.langchain.com/pricing) can change. Leave the endpoint blank for the SDK default, or use the official API endpoint for your account's region. Restart the app after configuration changes.

Named LangGraph nodes appear as spans. The central LiteLLM call is wrapped with `@traceable(run_type="llm")`. Safe metadata includes provider, model, routing category, retrieval count, revision count, and thread ID. Tracing is explicitly disabled in application call contexts unless enabled in Settings; no key is required when disabled.

**Privacy:** prompts, generated outputs, conversation summaries, memory excerpts, and retrieved document text may be sent to LangSmith when enabled. Tracing is not a private local log. Documents also go to Gemini for embeddings, chunks live in hosted PostgreSQL, and evidence goes to the selected answer provider regardless of tracing. Never upload secrets. Known key/connection-string patterns and configured credential values are redacted before model calls; memory storage rejects detectable secrets. Pattern detection cannot identify every arbitrary secret. Third-party verbose logging is suppressed, configuration hides secret values, and user-facing failure messages exclude raw SDK/database exceptions.

No LangSmith Deployment, LLM Gateway, or LiteLLM proxy is used.

## Commands

| Make target | Equivalent command |
| --- | --- |
| `make setup` | `python3.12 -m venv .venv` then the pip installation commands above |
| `make check` | `.venv/bin/python scripts/check_setup.py` |
| `make init-db` | `.venv/bin/python scripts/init_db.py` |
| `make ingest-samples` | `.venv/bin/python scripts/ingest_samples.py` |
| `make run` | `.venv/bin/streamlit run app.py` |
| `make test` | `.venv/bin/python -m pytest -q` |
| `make storage-report` | `.venv/bin/python scripts/storage_report.py` |
| `make prune-dry-run` | `.venv/bin/python scripts/prune_checkpoints.py --dry-run` |
| `make prune` | `.venv/bin/python scripts/prune_checkpoints.py --apply` |

Pass `CURRENT_THREAD=UUID` to the pruning Make targets to protect the selected conversation explicitly. Setup accepts `PYTHON=/path/to/python3.12`. Do not run `make setup` against an active Streamlit process.

## Tests and sample questions

```bash
make test
.venv/bin/python scripts/check_setup.py --smoke
.venv/bin/python -m compileall -q src scripts app.py tests
.venv/bin/python -m pip check
```

Default tests use mocks, an in-memory LangGraph test checkpointer, transaction fakes, and Streamlit's AppTest. Socket connection attempts are forbidden by the test fixture. No hosted database, Gemini, OpenAI, or LangSmith credentials are required. Tests cover chunking/overlap, hashing, upload validation/PDF pages, transactional rollback, idempotence, dimensions, routing/manual override/fallback, citations, memory scoping/deduplication/secrets, bounded review, termination, actual checkpoint count, reducer compaction, resumed history, activity/rollover, safe cleanup, storage threshold, and tracing-disabled operation.

Live tests are skipped unless explicitly opted in. Use a dedicated hosted PostgreSQL test database or branch for the first command. DB tests create a uniquely named temporary document/memory and remove them; they also perform idempotent schema setup. Provider tests may incur charges. The LangSmith test writes one synthetic trace.

```bash
RUN_DB_TESTS=1 .venv/bin/python -m pytest -q tests/test_integration.py -k database
RUN_LLM_TESTS=1 .venv/bin/python -m pytest -q tests/test_integration.py -k 'embeddings or chat'
RUN_LANGSMITH_TESTS=1 .venv/bin/python -m pytest -q tests/test_integration.py -k langsmith
```

The three original fictional Lantern Harbor source documents cover tiers, capabilities, retention, encryption, escalation, and recovery. `sample_docs/questions.md` lists eleven questions, expected facts, and source documents, including an insufficient-evidence question. It is intentionally excluded from automatic ingestion so answers are grounded in the actual source documents.

## Troubleshooting

- **Missing credentials:** `make check` prints variable names and next actions without values. `--smoke` only tests imports and succeeds without `.env`. Copy `.env.example`, replace its database placeholder, and set keys locally.
- **SSL / connection failures:** use an accepted provider host and SSL query parameter, check the project is available and the URL has the correct database/role/password, and percent-encode password URL characters. Preserve provider-generated SSL/channel-binding options. A Supabase direct host is IPv6 by default; use the dashboard's Session pooler URL on IPv4-only networks. The app suppresses raw connection errors to avoid exposing credentials.
- **Extension/schema failures:** run `make init-db` with an authorized database role. It must create pgvector and run official checkpointer migrations. Re-running initialization is safe. HNSW failure alone is nonfatal.
- **Model not found / 400 / 401 / 403:** confirm model name, provider prefix, key/project access, and key validity. These errors do not trigger retry loops. Change chat model configuration explicitly if a model is retired. For embedding changes, follow the complete migration procedure.
- **429 / timeouts / 5xx:** check quota/billing and provider status. Embedding retries are bounded; chat fallback is bounded. Hosted database cold starts can add latency. Retry later instead of raising pool size unnecessarily.
- **Invalid critic JSON:** one revision is attempted, then the app returns insufficient evidence. Add clearer source material or simplify the question.
- **Blank PDF:** image-only pages have no extractable text; provide a text-based PDF, TXT, or Markdown instead.
- **Cleanup skipped:** inspect dry-run reasons. Save a summary through a successful turn, archive old threads, or explicitly use `--force` only when willing to discard an unsummarized checkpoint. An active graph holds the cleanup lock; wait for it to finish.
- **Storage warning:** preview checkpoint pruning, review unused documents, and separately review memories. No automatic deletion is caused by the size warning.
- **Tracing missing:** confirm enabled/key/project/region settings, restart, and check account quota. Disable tracing to keep using the app without LangSmith.
- **Python imports / disk space:** use `.venv/bin/python`, ensure Python 3.12/Apple Silicon wheels, reinstall from the pinned requirements with `--no-cache-dir`, and avoid extra model/framework packages.

## Files and upstream references

`src/` separates configuration, security, database, ingestion, embeddings, retrieval, memory, routing, prompts/agents/graph, thread service, retention, monitoring, and application orchestration. `app.py` provides the three UI tabs; `scripts/` provides credential-safe CLI entry points; `sql/schema.sql` holds application DDL; `tests/` holds credential-free and opt-in integration checks.

API contracts and requested model availability were checked against [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings), [Gemini model deprecations](https://ai.google.dev/gemini-api/docs/deprecations), [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [LiteLLM Gemini support](https://docs.litellm.ai/docs/providers/gemini), and [PostgresSaver](https://reference.langchain.com/python/langgraph.checkpoint.postgres/PostgresSaver). Account-specific model access and live integration are verified only by your explicitly enabled live tests; they were not assumed from documentation.
