"""Run with .venv/bin/streamlit run app.py. Importing this module is credential-free."""
from pathlib import Path
from collections import Counter
from uuid import uuid4
import streamlit as st
from src.config import load_settings, SetupError
from src.db import create_pool
from src.runtime import create_runtime
from src.storage_monitor import storage_report


@st.cache_resource(show_spinner=False)
def cached_pool(_settings):
    return create_pool(_settings)


@st.cache_resource(show_spinner=False)
def cached_runtime(_settings):
    # Includes the Gemini client, service objects, and compiled graph; never caches answers.
    return create_runtime(_settings, cached_pool(_settings))


def safe_action(action):
    try:
        return action()
    except SetupError as exc:
        st.error(str(exc))
    except Exception:
        st.error('Operation failed. Check connection, schema initialization, model access, and API quota. Sensitive error details are hidden.')
    return None


def set_thread(thread_id):
    st.session_state.thread_id = thread_id
    st.query_params['thread'] = thread_id
    st.session_state.pop('prune_preview', None)


def show_answer_details(state, runtime):
    if state.get('actual_model'):
        st.caption(f"Model: {state['actual_model']} · {state.get('routing_reason', '')}")
        st.caption(f"Review: {state.get('critic_status', '')} · Revisions: {state.get('revision_count', 0)}/1")
    if state.get('evidence_refs'):
        with st.expander('Evidence and citation sources'):
            st.dataframe(state['evidence_refs'], hide_index=True)
            if st.button('Load source excerpts', key='load_evidence'):
                evidence = safe_action(lambda: runtime.retriever.fetch_refs(state['evidence_refs']))
                for row in evidence or []:
                    st.markdown(f"**{row['filename']}, chunk {row['chunk_index']}**")
                    st.text(row['content'])
                st.caption('Loaded from the current document version; replaced or deleted chunks may no longer exist.')


def chat_tab(runtime, settings, user_id):
    thread_id = st.session_state.thread_id
    st.caption(f'Current thread: {thread_id}')
    if st.session_state.get('rollover_notice'):
        st.warning(st.session_state.pop('rollover_notice'))
        previous = st.session_state.pop('rollover_answer', None)
        if previous:
            st.markdown(previous)
    row = runtime.threads.get(thread_id, user_id)
    st.caption(f"Completed turns: {row['message_count']} / {settings.max_messages_per_thread}")
    cols = st.columns(3)
    with cols[0]:
        manual = st.selectbox('Chat model', ['Auto', 'Gemini', 'OpenAI'])
    with cols[1]:
        top_k = st.slider('Retrieved chunks', 1, 20, settings.default_top_k)
    with cols[2]:
        if st.button('New conversation'):
            runtime.threads.archive(thread_id, user_id)
            set_thread(runtime.threads.create(user_id))
            st.rerun()
    active = runtime.threads.list(user_id, 'active')
    if len(active) > 1:
        selected = st.selectbox('Resume conversation', [r['thread_id'] for r in active], index=next(i for i, r in enumerate(active) if r['thread_id'] == thread_id))
        if selected != thread_id:
            set_thread(selected)
            st.rerun()
    state = safe_action(lambda: runtime.assistant.history(thread_id, user_id)) or {}
    for message in state.get('messages', []):
        with st.chat_message('user' if message.type == 'human' else 'assistant'):
            st.markdown(message.content)
    show_answer_details(state, runtime)
    question = st.chat_input('Ask about your documents or remembered preferences', max_chars=12000)
    if question:
        with st.spinner('Retrieving evidence, answering, and checking support…'):
            result = safe_action(lambda: runtime.assistant.ask(question, thread_id, user_id, manual, top_k))
        if result:
            if result.get('next_thread_id'):
                set_thread(result['next_thread_id'])
                st.session_state.rollover_notice = 'Conversation rolled over at the configured turn limit. The previous summary remains in episodic memory; history was not copied.'
                st.session_state.rollover_answer = result['final_answer']
            st.rerun()
    with st.expander('Delete current conversation'):
        st.caption('Deletes its checkpoints and thread record. CoALA memories remain.')
        confirm = st.checkbox('Confirm deletion of this conversation', key=f'confirm_current_{thread_id}')
        if st.button('Delete current conversation', disabled=not confirm):
            runtime.retention.delete_conversation(thread_id, user_id)
            set_thread(runtime.threads.create(user_id))
            st.rerun()


def documents_tab(runtime, settings):
    st.subheader('Document library')
    st.caption('Same filename with changed content replaces the document atomically. Identical content is skipped, even under another filename.')
    uploaded = st.file_uploader('Upload TXT, Markdown, or text PDF', type=['txt', 'md', 'pdf'], accept_multiple_files=True)
    ingest = st.button('Ingest uploaded files', disabled=not uploaded)
    samples = st.button('Load sample documents')
    files = []
    if ingest:
        files = [(file.name, file.getvalue()) for file in uploaded]
    elif samples:
        files = [(p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / 'sample_docs').glob('*.md')) if p.name != 'questions.md']
    if files:
        progress, results = st.progress(0), []
        for i, (name, data) in enumerate(files):
            results.append(runtime.ingestion.ingest(name, data))
            progress.progress((i + 1) / len(files))
        counts = Counter(row['status'] for row in results)
        st.write({key: counts[key] for key in ('added', 'skipped', 'replaced', 'failed')})
        st.dataframe(results, hide_index=True)
    documents = runtime.documents.list()
    st.dataframe(documents, hide_index=True)
    if documents:
        labels = {str(d['id']): d['filename'] for d in documents}
        selection = st.selectbox('Document to delete', list(labels), format_func=labels.get)
        confirm = st.checkbox('Confirm document and chunk deletion', key=f'confirm_document_{selection}')
        if st.button('Delete document', disabled=not confirm):
            runtime.documents.delete(selection)
            st.rerun()


def diagnostics_tab(runtime, settings, user_id):
    st.subheader('CoALA long-term memories')
    category = st.selectbox('Memory type', ['All', 'episodic', 'semantic', 'procedural'])
    memory_type = None if category == 'All' else category
    memories = runtime.memories.list(user_id, memory_type)
    st.dataframe(memories, hide_index=True)
    if memories:
        selected = st.selectbox('Memory to delete', [m['id'] for m in memories])
        confirm_one = st.checkbox('Confirm individual memory deletion', key=f'confirm_memory_{selected}')
        if st.button('Delete selected memory', disabled=not confirm_one):
            runtime.memories.delete(user_id, selected)
            st.rerun()
    bulk_nonce = st.session_state.get('memory_delete_nonce', 'initial')
    confirm_bulk = st.checkbox(f'Confirm deletion of all {category.lower()} memories for {user_id}', key=f'confirm_bulk_{category}_{bulk_nonce}')
    if st.button('Delete matching memories', disabled=not confirm_bulk):
        runtime.memories.delete_all(user_id, memory_type)
        st.session_state.memory_delete_nonce = str(uuid4())
        st.rerun()
    st.subheader('Diagnostics')
    st.write(f'Database: connected to {settings.database_provider}')
    st.write(f'Embeddings: {settings.embedding_model}, {settings.embedding_dimensions} dimensions')
    st.write('Chat providers: ' + ', '.join(settings.providers))
    st.write('LangSmith: ' + ('enabled; prompts and outputs may be sent' if settings.langsmith_tracing else 'disabled'))
    st.code('START → supervisor → memory_retriever → document_retriever\n  → answer_agent → critic_agent ── revise once → answer_agent\n                    └─ accepted/exhausted → memory_curator\n                       → compact_state → END')
    st.json({'durability': 'exit', 'recent_messages': settings.recent_messages_to_keep,
             'turns_per_thread': settings.max_messages_per_thread, 'inactive_days': settings.checkpoint_retention_days,
             'retained_threads_per_user': settings.max_retained_threads, 'automatic_cleanup': settings.auto_prune_checkpoints})
    if st.button('Refresh storage report') or 'storage_report' not in st.session_state:
        st.session_state.storage_report = storage_report(runtime.db, settings.database_warning_mb)
    report = st.session_state.get('storage_report')
    if report:
        st.metric('Database size', report['database']['pretty'])
        if report['warning']:
            st.warning('Database size exceeds the warning threshold. Review old conversations and unused documents or memories, then initiate cleanup. No data was deleted automatically.')
        st.dataframe(report['tables'], hide_index=True)
        st.write('Vector/index storage', report['vector_indexes'])
        st.write('Record counts', report['counts'], report['memories'])
    st.subheader('Checkpoint retention')
    if st.button('Preview checkpoint-pruning candidates'):
        st.session_state.prune_preview = runtime.retention.prune(st.session_state.thread_id, user_id)
        st.session_state.prune_preview_nonce = str(uuid4())
    preview = st.session_state.get('prune_preview')
    if preview:
        st.dataframe(preview['candidates'], hide_index=True)
        st.write('Skipped:', preview['skipped'])
        confirmed = st.checkbox('Confirm deletion of the previewed checkpoint threads', key='confirm_prune_' + st.session_state.prune_preview_nonce)
        if st.button('Apply checkpoint cleanup', disabled=not confirmed):
            result = runtime.retention.prune(st.session_state.thread_id, user_id, apply=True,
                                             only_ids={r['thread_id'] for r in preview['candidates']})
            st.write('Deleted:', result['deleted'], 'Failed:', result['failed'])
            st.session_state.pop('prune_preview', None)
    st.subheader('Archived conversations')
    archived = runtime.threads.list(user_id, 'archived')
    st.dataframe(archived, hide_index=True)
    if archived:
        selected = st.selectbox('Archived conversation', [r['thread_id'] for r in archived])
        confirmed = st.checkbox('Confirm archived conversation deletion; keep CoALA memories', key=f'confirm_archived_{selected}')
        if st.button('Delete selected archived conversation', disabled=not confirmed):
            runtime.retention.delete_conversation(selected, user_id)
            st.rerun()


def main():
    st.set_page_config(page_title='Local CoALA RAG assistant', page_icon='📚', layout='wide')
    st.title('Local CoALA RAG assistant')
    st.caption('Evidence-grounded answers · Persistent memory · Bounded agent review')
    settings = safe_action(load_settings)
    if settings is None:
        return
    database_provider = settings.database_provider if settings.database_url.get_secret_value() else 'Supabase or Neon'
    st.info(f'Privacy: documents are sent to Gemini for embeddings and chunks are stored in hosted {database_provider} PostgreSQL. Retrieved excerpts and memories are sent to the selected chat provider. Enabling LangSmith can send prompts, outputs, memories, and excerpts to LangSmith. Do not upload secrets.')
    missing = []
    if not settings.database_url.get_secret_value():
        missing.append('DATABASE_URL: Supabase or Neon PostgreSQL connection string with sslmode=require')
    if not settings.gemini_api_key.get_secret_value():
        missing.append('GEMINI_API_KEY: required for hosted embeddings')
    if missing:
        st.warning('Complete local setup before connecting:')
        for item in missing:
            st.write('• ' + item)
        st.code('cp .env.example .env\n# Edit .env locally, then:\nmake check\nmake init-db\nmake ingest-samples\nmake run')
        return
    runtime = safe_action(lambda: cached_runtime(settings))
    if runtime is None:
        return
    user_id = settings.default_user_id
    # A local identity filter, not authentication. Keep the server bound to loopback.
    st.sidebar.caption(f'User: {user_id}')
    if 'thread_id' not in st.session_state:
        desired = st.query_params.get('thread')
        row = runtime.threads.get(desired, user_id) if desired else None
        if row and row['status'] == 'active':
            set_thread(desired)
        else:
            active = runtime.threads.list(user_id, 'active')
            set_thread(active[0]['thread_id'] if active else runtime.threads.create(user_id))
    safe_action(lambda: runtime.retention.auto_prune(st.session_state.thread_id))
    chat, docs, diagnostics = st.tabs(['Chat', 'Documents', 'Memory and diagnostics'])
    with chat:
        safe_action(lambda: chat_tab(runtime, settings, user_id))
    with docs:
        safe_action(lambda: documents_tab(runtime, settings))
    with diagnostics:
        safe_action(lambda: diagnostics_tab(runtime, settings, user_id))


if __name__ == '__main__':
    main()
