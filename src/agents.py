import json
from uuid import uuid4
from langchain_core.messages import AIMessage, RemoveMessage
from src.models import Critique, Curation
from src.llm_router import choose_route
from src.citations import citation_label, validate_citations, has_citation
from src.security import redact
from src import prompts


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    return json.loads(text)


class Agents:
    def __init__(self, settings, router, retriever, memories):
        self.settings, self.router, self.retriever, self.memory_store = settings, router, retriever, memories

    def metadata(self, state):
        return {key: state.get(key) for key in ('thread_id', 'routing_category', 'revision_count')} | {'retrieval_count': len(state.get('evidence', []))}

    def call(self, state, system, payload):
        return self.router.complete(state['selected_model'], [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], self.metadata(state))

    def supervisor(self, state):
        route = choose_route(state['question'], self.settings, state.get('manual_model', 'Auto'))
        return {'selected_model': route.model, 'actual_model': route.model, 'routing_category': route.category,
                'routing_reason': route.reason, 'rag_needed': route.rag_needed,
                'plan': ['Recall relevant memories', 'Retrieve document evidence' if route.rag_needed else 'Respond to greeting', 'Answer and check support'],
                'revision_count': 0, 'evidence': [], 'memories': [], 'retrieval_ids': [], 'memory_ids': [],
                'evidence_refs': [], 'draft_answer': '', 'final_answer': '', 'critic': {}, 'summary_saved': False}

    def memory_retriever(self, state):
        rows = self.memory_store.retrieve(state['user_id'], state['question'], state.get('memory_thread_id')) if state['rag_needed'] else []
        return {'memories': rows, 'memory_ids': [r['id'] for r in rows]}

    def document_retriever(self, state):
        rows = self.retriever.search(state['question'], state.get('top_k', self.settings.default_top_k)) if state['rag_needed'] else []
        refs = [{k: row[k] for k in ('id', 'document_id', 'filename', 'chunk_index', 'page', 'score')} for row in rows]
        return {'evidence': rows, 'retrieval_ids': [r['id'] for r in rows], 'evidence_refs': refs}

    def answer_agent(self, state):
        answer, model = self.call(state, prompts.ANSWER, {
            'question': state['question'], 'summary': state.get('summary', ''),
            'recent_messages': [{'role': m.type, 'content': m.content} for m in state.get('messages', [])[-self.settings.recent_messages_to_keep:]],
            'documents': [{'citation': citation_label(e), 'text': e['content']} for e in state.get('evidence', [])],
            'memories': state.get('memories', []),
            'revision_feedback': state.get('critic', {})})
        return {'draft_answer': answer[:12000], 'actual_model': model}

    def critic_agent(self, state):
        invalid = validate_citations(state['draft_answer'], state.get('evidence', []))
        try:
            text, _ = self.call(state, prompts.CRITIC, {'question': state['question'], 'answer': state['draft_answer'],
                'evidence': state.get('evidence', []), 'memories': state.get('memories', [])})
            critique = Critique.model_validate(parse_json(text))
        except Exception:
            critique = Critique(decision='revise', reason='Critic response unavailable or invalid.', revision_instructions='Return only claims directly supported by the evidence, or state that evidence is insufficient.')
        if invalid:
            critique = Critique(decision='revise', reason='Invalid citations.', unsupported_claims=invalid,
                                revision_instructions='Remove fabricated citations and claims; cite only supplied evidence labels.')
        if state.get('evidence') and not has_citation(state['draft_answer']) and 'insufficient' not in state['draft_answer'].lower():
            critique = Critique(decision='revise', reason='Document answer lacks citations.', revision_instructions='Cite the supplied chunks for supported claims, or explicitly state that evidence is insufficient.')
        needs_revision = critique.decision == 'revise' and state['revision_count'] == 0
        final = state['draft_answer']
        status = 'accepted'
        if critique.decision == 'revise' and not needs_revision:
            # After one revision, fail closed instead of presenting a rejected answer.
            final = 'The retrieved evidence is insufficient to produce an answer that passed the support and citation checks. Please narrow the question or add relevant documents.'
            status = 'insufficient evidence after one revision'
        elif needs_revision:
            status = 'revision requested'
        return {'critic': critique.model_dump(), 'critic_status': status,
                'revision_count': 1 if needs_revision else state['revision_count'],
                'final_answer': '' if needs_revision else final}

    def memory_curator(self, state):
        prior = state.get('summary', '')
        if not state['rag_needed']:
            # A greeting or thanks does not justify paid curation or a new durable memory.
            existing = self.memory_store.latest_episode(state['user_id'], state['thread_id']) if prior else None
            return {'summary': prior, 'summary_saved': bool(existing)}
        summary = redact((prior + '\nQuestion: ' + state['question'][:350] + '\nAnswer: ' + state['final_answer'][:650])[-2000:])
        proposals = []
        try:
            text, _ = self.call(state, prompts.CURATOR, {'prior_summary': prior, 'question': state['question'],
                                                       'accepted_answer': state['final_answer'], 'critic_status': state['critic_status']})
            result = Curation.model_validate(parse_json(text))
            summary, proposals = redact(result.summary), result.memories
        except Exception:
            pass  # A deterministic summary still permits safe rollover.
        # Avoid making greetings a source of durable personal facts.
        if state['critic_status'] == 'accepted' and state['rag_needed']:
            for proposal in proposals:
                self.memory_store.save(state['user_id'], state['thread_id'], proposal.memory_type,
                                       proposal.content, {'provenance': proposal.provenance})
        memory_id = self.memory_store.save(state['user_id'], state['thread_id'], 'episodic', summary,
                                          {'source': 'conversation summary', 'model': state['actual_model']})
        return {'summary': summary, 'summary_saved': bool(memory_id)}

    def compact_state(self, state):
        final_message = AIMessage(content=state['final_answer'], id=str(uuid4()))
        all_messages = list(state.get('messages', [])) + [final_message]
        remove = [RemoveMessage(id=m.id) for m in all_messages[:-self.settings.recent_messages_to_keep]]
        return {'messages': remove + [final_message], 'summary': state.get('summary', '')[-2000:],
                'question': '', 'evidence': [], 'memories': [], 'draft_answer': '', 'critic': {},
                'plan': [], 'temporary_response': '', 'memory_thread_id': None}
