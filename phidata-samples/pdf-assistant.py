"""
OpenAI LLM assistant with memory, knowledge and tools
Start pgvector docker container on 5532

https://docs.phidata.com/basics
"""

import logging
import pathlib

from phi.assistant import Assistant
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.vectordb.pgvector import PgVector2  # type: ignore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    knowledge_base = PDFUrlKnowledgeBase(
        urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=PgVector2(
            collection="recipes",
            db_url=db_url,
        ),
    )
    knowledge_base.load(recreate=False)
    storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)
    assistant = Assistant(  # type: ignore
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )
    assistant.cli_app(markdown=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
