from uuid import uuid4
from contextlib import contextmanager
from langchain_core.messages import HumanMessage
from langsmith import Client, tracing_context
from src.config import SetupError
from src.security import redact


class AssistantService:
    def __init__(self, settings, db, graph, threads, retention):
        self.settings, self.db, self.graph = settings, db, graph
        self.threads, self.retention = threads, retention
        self.trace_client = None
        if settings.langsmith_tracing:
            if not settings.langsmith_api_key.get_secret_value():
                raise SetupError('Set LANGSMITH_API_KEY or disable LANGSMITH_TRACING.')
            self.trace_client = Client(api_key=settings.langsmith_api_key.get_secret_value(),
                                       api_url=settings.langsmith_endpoint or None)

    @contextmanager
    def tracing(self, thread_id):
        with tracing_context(enabled=self.settings.langsmith_tracing, client=self.trace_client,
                             project_name=self.settings.langsmith_project, metadata={'thread_id': thread_id}):
            yield

    def history(self, thread_id, user_id):
        if not self.threads.get(thread_id, user_id):
            raise SetupError('Conversation not found for this user.')
        with self.tracing(thread_id):
            return self.graph.get_state({'configurable': {'thread_id': thread_id}}).values

    def ask(self, question, thread_id, user_id, manual='Auto', top_k=None):
        question = redact(question, [getattr(self.settings, f).get_secret_value() for f in
            ('database_url', 'gemini_api_key', 'openai_api_key', 'langsmith_api_key')]).strip()
        if not question or len(question) > 12000:
            raise SetupError('Enter a question with 1–12,000 characters.')
        with self.db.invocation_lock(thread_id):
            row = self.threads.get(thread_id, user_id)
            if not row or row['status'] != 'active':
                raise SetupError('Select an active conversation or start a new one.')
            if row['message_count'] >= self.settings.max_messages_per_thread:
                previous = self.history(thread_id, user_id)
                next_id = self.threads.rollover(thread_id, user_id, previous.get('summary', ''))
                return {'rollover_only': True, 'next_thread_id': next_id, 'final_answer': 'The previous thread reached its limit. Please submit your question in the new thread.'}
            with self.tracing(thread_id):
                graph_input = {
                    'messages': [HumanMessage(content=question, id=str(uuid4()))],
                    'question': question, 'thread_id': thread_id, 'user_id': user_id,
                    'manual_model': manual, 'top_k': top_k or self.settings.default_top_k,
                    'memory_thread_id': None,
                }
                source_thread = row.get('metadata', {}).get('rollover_from')
                if source_thread and row['message_count'] == 0:
                    episode = self.threads.memories.latest_episode(user_id, source_thread)
                    if episode:
                        graph_input['summary'] = episode['content'][:2000]
                result = self.graph.invoke(graph_input, config={'configurable': {'thread_id': thread_id}, 'recursion_limit': 20,
                           'metadata': {'thread_id': thread_id}}, durability='exit')
            self.threads.record_success(thread_id, user_id, result.get('summary_saved', False))
            next_id = self.threads.rollover(thread_id, user_id, result.get('summary', ''))
        return {**result, 'next_thread_id': next_id}
