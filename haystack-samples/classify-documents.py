"""
https://haystack.deepset.ai/tutorials/32_classifying_documents_and_queries_by_language
"""

import logging
import pathlib
import sys

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.classifiers import DocumentLanguageClassifier
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.routers import MetadataRouter, TextLanguageRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def get_documents():
    documents = [
        Document(
            content="""
Super appartement. Juste au dessus de plusieurs bars qui ferment très tard. A
savoir à l'avance. (Bouchons d'oreilles fournis !)
"""
        ),
        Document(
            content="""
El apartamento estaba genial y muy céntrico, todo a mano. Al lado de la
librería Lello y De la Torre de los clérigos. Está situado en una zona de
marcha, así que si vais en fin de semana , habrá ruido, aunque a nosotros no
nos molestaba para dormir
"""
        ),
        Document(
            content="""
The keypad with a code is convenient and the location is convenient. Basically
everything else, very noisy, wi-fi didn't work, check-in person didn't explain
anything about facilities, shower head was broken, there's no cleaning and
everything else one may need is charged.
"""
        ),
        Document(
            content="""
It is very central and appartement has a nice appearance (even though a lot
IKEA stuff), *W A R N I N G** the appartement presents itself as a elegant and
as a place to relax, very wrong place to relax - you cannot sleep in this
appartement, even the beds are vibrating from the bass of the clubs in the same
building - you get ear plugs from the hotel -> now I understand why -> I missed
a trip as it was so loud and I could not hear the alarm next day due to the ear
plugs.- there is a green light indicating 'emergency exit' just above the bed,
which shines very bright at night - during the arrival process, you felt the
urge of the agent to leave as soon as possible. - try to go to 'RVA clerigos
appartements' -> same price, super quiet, beautiful, city center and very nice
staff (not an agency)- you are basically sleeping next to the fridge, which
makes a lot of noise, when the compressor is running -> had to switch it off -
but then had no cool food and drinks. - the bed was somehow broken down - the
wooden part behind the bed was almost falling appart and some hooks were broken
before- when the neighbour room is cooking you hear the fan very loud. I
initially thought that I somehow activated the kitchen fan
"""
        ),
        Document(
            content="""
Un peu salé surtout le sol. Manque de service et de souplesse
"""
        ),
        Document(
            content="""
Nous avons passé un séjour formidable. Merci aux personnes , le bonjours à
Ricardo notre taxi man, très sympathique. Je pense refaire un séjour parmi
vous, après le confinement, tout était parfait, surtout leur gentillesse,
aucune chaude négative. Je n'ai rien à redire de négative, Ils étaient a notre
écoute, un gentil message tout les matins, pour nous demander si nous avions
besoins de renseignement et savoir si tout allait bien pendant notre séjour.
"""
        ),
        Document(
            content="""
Céntrico. Muy cómodo para moverse y ver Oporto. Edificio con terraza propia en
la última planta. Todo reformado y nuevo. Te traen un estupendo desayuno todas
las mañanas al apartamento. Solo que se puede escuchar algo de ruido de la
calle a primeras horas de la noche. Es un zona de ocio nocturno. Pero respetan
los horarios.
"""
        ),
    ]
    docs = []
    for document in documents:
        docs.append(Document(content=str(document.content).replace("\n", "")))
    return docs


def get_prompt_template():
    prompt_template = """
    You will be provided with reviews for an accommodation.
    Answer the question concisely based solely on the given reviews.
    Reviews:
      {% for doc in documents %}
        {{ doc.content }}
      {% endfor %}
    Question: {{ query}}
    Answer:
    """
    return prompt_template


def get_rag_pipeline(en_doc_store, fr_doc_store, es_doc_store):
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=TextLanguageRouter(["en", "fr", "es"]), name="router"
    )
    rag_pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=en_doc_store),
        name="en_retriever",
    )
    rag_pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=fr_doc_store),
        name="fr_retriever",
    )
    rag_pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=es_doc_store),
        name="es_retriever",
    )
    rag_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
    prompt_template = get_prompt_template()
    rag_pipeline.add_component(
        instance=PromptBuilder(template=prompt_template), name="prompt_builder"
    )
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")

    rag_pipeline.connect("router.en", "en_retriever.query")
    rag_pipeline.connect("router.fr", "fr_retriever.query")
    rag_pipeline.connect("router.es", "es_retriever.query")
    rag_pipeline.connect("en_retriever", "joiner")
    rag_pipeline.connect("fr_retriever", "joiner")
    rag_pipeline.connect("es_retriever", "joiner")
    rag_pipeline.connect("joiner.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    return rag_pipeline


def main():
    en_document_store = InMemoryDocumentStore()
    fr_document_store = InMemoryDocumentStore()
    es_document_store = InMemoryDocumentStore()

    language_classifier = DocumentLanguageClassifier(
        languages=["en", "fr", "es"]
    )
    router_rules = {
        "en": {"language": {"$eq": "en"}},
        "fr": {"language": {"$eq": "fr"}},
        "es": {"language": {"$eq": "es"}},
    }
    router = MetadataRouter(rules=router_rules)

    en_writer = DocumentWriter(document_store=en_document_store)
    fr_writer = DocumentWriter(document_store=fr_document_store)
    es_writer = DocumentWriter(document_store=es_document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=language_classifier, name="language_classifier"
    )
    indexing_pipeline.add_component(instance=router, name="router")
    indexing_pipeline.add_component(instance=en_writer, name="en_writer")
    indexing_pipeline.add_component(instance=fr_writer, name="fr_writer")
    indexing_pipeline.add_component(instance=es_writer, name="es_writer")

    indexing_pipeline.connect("language_classifier", "router")
    indexing_pipeline.connect("router.en", "en_writer")
    indexing_pipeline.connect("router.fr", "fr_writer")
    indexing_pipeline.connect("router.es", "es_writer")

    docs = get_documents()
    indexing_pipeline.run(data={"language_classifier": {"documents": docs}})

    print("---English documents:\n", en_document_store.filter_documents())
    print("---French documents:\n", fr_document_store.filter_documents())
    print("---Spanish documents:\n", es_document_store.filter_documents())

    rag_pipeline = get_rag_pipeline(
        en_document_store, fr_document_store, es_document_store
    )
    en_question = "Is this apartment conveniently located?"

    result = rag_pipeline.run(
        {
            "router": {"text": en_question},
            "prompt_builder": {"query": en_question},
        }
    )
    print(result["llm"]["replies"][0])

    es_question = "¿El desayuno es genial?"
    result = rag_pipeline.run(
        {
            "router": {"text": es_question},
            "prompt_builder": {"query": en_question},
        }
    )
    print(result["llm"]["replies"][0])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()
