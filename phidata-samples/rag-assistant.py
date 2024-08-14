"""
Use retrieval augmented generation to answer questions
Start pgvector docker container on 5532

https://docs.phidata.com/basics
"""

import logging
import pathlib

from phi.assistant import Assistant
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    knowledge_base = PDFUrlKnowledgeBase(
        urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=PgVector2(
            collection="recipes",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
    )
    knowledge_base.load(recreate=False)
    assistant = Assistant(  # type: ignore
        knowledge_base=knowledge_base, add_references_to_prompt=True
    )
    query = "How do I make pad thai?"
    print(f"{query=}")
    assistant.print_response(query, markdown=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
