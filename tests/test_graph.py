from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from src.graph import build_graph


def invoke(graph, thread='thread-1'):
    return graph.invoke({'messages': [HumanMessage(content='What is retention?', id=str(uuid4()))],
                         'question': 'What is retention?', 'user_id': 'user-1', 'thread_id': thread,
                         'manual_model': 'Auto', 'top_k': 5},
                        config={'configurable': {'thread_id': thread}, 'recursion_limit': 20}, durability='exit')


def test_accept_and_termination(graph_dependencies):
    agents, router, retriever, memories = graph_dependencies
    saver = InMemorySaver()
    graph = build_graph(agents, saver)
    result = invoke(graph)
    assert result['critic_status'] == 'accepted'
    assert result['revision_count'] == 0
    assert result['summary_saved']
    assert len(result['messages']) == 2
    assert '[security_policy.md, chunk 3]' in result['final_answer']
    assert router.complete.call_count == 3
    # One completed checkpoint per invocation, not one per node.
    assert len(list(saver.list({'configurable': {'thread_id': 'thread-1'}}))) == 1
    assert graph.get_state({'configurable': {'thread_id': 'thread-1'}}).values['evidence'] == []


def test_single_revision_maximum(graph_dependencies, settings):
    agents, router, _, _ = graph_dependencies
    model = settings.simple_chat_model
    router.complete.side_effect = [
        ('Unsupported [fake.md, chunk 999].', model),
        ('{"decision":"revise","reason":"unsupported","revision_instructions":"Fix it"}', model),
        ('Still unsupported [fake.md, chunk 999].', model),
        ('{"decision":"revise","reason":"still unsupported"}', model),
        ('{"summary":"The question lacked sufficient support.","memories":[]}', model),
    ]
    result = invoke(build_graph(agents, InMemorySaver()))
    assert result['revision_count'] == 1
    assert 'insufficient' in result['final_answer']
    assert 'fake.md' not in result['final_answer']
    assert router.complete.call_count == 5


def test_revision_then_accept(graph_dependencies, settings):
    agents, router, _, _ = graph_dependencies
    model = settings.simple_chat_model
    router.complete.side_effect = [
        ('Retention is 30 days [security_policy.md, chunk 3].', model),
        ('{"decision":"revise","reason":"Wrong duration"}', model),
        ('Retention is 365 days [security_policy.md, chunk 3].', model),
        ('{"decision":"accept","reason":"Supported"}', model),
        ('{"summary":"Retention is 365 days.","memories":[]}', model),
    ]
    result = invoke(build_graph(agents, InMemorySaver()))
    assert result['critic_status'] == 'accepted'
    assert result['revision_count'] == 1
    assert '365' in result['final_answer']


def test_invalid_critic_fails_closed(graph_dependencies, settings):
    agents, router, _, _ = graph_dependencies
    router.complete.side_effect = lambda *a, **k: ('not JSON and not cited', settings.simple_chat_model)
    result = invoke(build_graph(agents, InMemorySaver()))
    assert result['revision_count'] == 1
    assert 'insufficient' in result['final_answer']


def test_greeting_does_not_create_irrelevant_memory(graph_dependencies):
    agents, router, _, memories = graph_dependencies
    result = agents.memory_curator({'rag_needed': False, 'summary': '', 'user_id': 'u', 'thread_id': 't'})
    assert result == {'summary': '', 'summary_saved': False}
    router.complete.assert_not_called()
    memories.save.assert_not_called()
