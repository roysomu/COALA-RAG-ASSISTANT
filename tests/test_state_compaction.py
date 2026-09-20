import json
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from src.graph import build_graph


def test_compaction_reducer_removes_messages(graph_dependencies):
    agents, _, _, _ = graph_dependencies
    state = {
        'messages': [HumanMessage(content=f'Message {i}', id=str(i)) for i in range(30)],
        'summary': 'Rolling summary', 'final_answer': 'Final answer',
        'evidence': [{'content': 'X' * 50000}], 'memories': [{'content': 'Y' * 50000}],
        'draft_answer': 'draft' * 2000, 'critic': {'reason': 'long' * 1000},
        'plan': ['temporary'] * 100, 'temporary_response': 'Z' * 5000,
    }
    updates = agents.compact_state(state)
    messages = add_messages(state['messages'], updates['messages'])
    assert len(messages) == 12
    assert messages[0].id == '19'
    assert messages[-1].content == 'Final answer'
    compact = {**state, **updates, 'messages': messages}
    assert len(json.dumps(compact, default=str)) < len(json.dumps(state, default=str)) / 10
    for key in ('evidence', 'memories', 'draft_answer', 'critic', 'plan', 'temporary_response'):
        assert not compact[key]
    assert compact['summary'] == 'Rolling summary'


def test_many_turns_resume_and_retain_twelve(graph_dependencies, settings):
    agents, router, _, _ = graph_dependencies
    def complete(model, messages, metadata):
        from src import prompts
        system = messages[0]['content']
        if system == prompts.CRITIC:
            return '{"decision":"accept","reason":"supported"}', model
        if system == prompts.CURATOR:
            return '{"summary":"Rolling retained context.","memories":[]}', model
        return '365 days [security_policy.md, chunk 3].', model
    router.complete.side_effect = complete
    saver = InMemorySaver()
    for i in range(10):
        # Recompile each time to model application restart; checkpoint data stays external.
        graph = build_graph(agents, saver)
        result = graph.invoke({'question': str(i), 'user_id': 'u', 'thread_id': 't',
            'messages': [HumanMessage(content=str(i), id=str(uuid4()))]},
            config={'configurable': {'thread_id': 't'}}, durability='exit')
    assert len(result['messages']) == 12
    assert result['messages'][0].content == '4'
    assert result['summary'] == 'Rolling retained context.'
    assert len(list(saver.list({'configurable': {'thread_id': 't'}}))) == 10
