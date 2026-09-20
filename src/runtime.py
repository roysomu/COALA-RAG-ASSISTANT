from dataclasses import dataclass
from src.config import SetupError
from src.db import Database, create_pool, create_checkpointer
from src.embeddings import Embeddings
from src.ingestion import DocumentStore, IngestionService
from src.retrieval import Retriever
from src.memory import MemoryStore
from src.llm_router import LLMRouter
from src.agents import Agents
from src.graph import build_graph
from src.thread_service import ThreadService
from src.checkpoint_retention import CheckpointRetention
from src.service import AssistantService


@dataclass
class Runtime:
    db: object
    saver: object
    documents: object
    ingestion: object
    retriever: object
    memories: object
    threads: object
    retention: object
    assistant: object


def create_runtime(settings, pool=None):
    pool = pool or create_pool(settings)
    db, saver = Database(pool), create_checkpointer(pool)
    db.verify_embedding_config(settings)
    embeddings = Embeddings(settings)
    docs, retriever, memories = DocumentStore(db), Retriever(db, embeddings), MemoryStore(db, embeddings)
    router = LLMRouter(settings)
    graph = build_graph(Agents(settings, router, retriever, memories), saver)
    threads, retention = ThreadService(db, settings, memories), CheckpointRetention(db, saver, settings)
    return Runtime(db, saver, docs, IngestionService(docs, embeddings, settings), retriever, memories,
                   threads, retention, AssistantService(settings, db, graph, threads, retention))
