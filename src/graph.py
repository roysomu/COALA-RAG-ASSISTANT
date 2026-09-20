from langgraph.graph import StateGraph, START, END
from src.models import GraphState


def build_graph(agents, checkpointer):
    graph = StateGraph(GraphState)
    for name in ('supervisor', 'memory_retriever', 'document_retriever', 'answer_agent', 'critic_agent', 'memory_curator', 'compact_state'):
        graph.add_node(name, getattr(agents, name))
    graph.add_edge(START, 'supervisor')
    for source, target in [('supervisor', 'memory_retriever'), ('memory_retriever', 'document_retriever'),
                           ('document_retriever', 'answer_agent'), ('answer_agent', 'critic_agent'),
                           ('memory_curator', 'compact_state'), ('compact_state', END)]:
        graph.add_edge(source, target)
    graph.add_conditional_edges('critic_agent', lambda s: 'answer_agent' if s['critic_status'] == 'revision requested' else 'memory_curator',
                                ['answer_agent', 'memory_curator'])
    return graph.compile(checkpointer=checkpointer)
