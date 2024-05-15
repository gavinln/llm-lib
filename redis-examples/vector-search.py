"""
https://cookbook.openai.com/examples/vector_databases/redis/redisjson/redisjson
"""

import logging
import pathlib
from typing import Any, NamedTuple

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from redis_util import get_embeddings, index_exists, print_indexing_failures

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


def get_news_texts() -> list[str]:
    text_1 = """Japan narrowly escapes recession

    Japan's economy teetered on the brink of a technical recession in the three
    months to September, figures show.

    Revised figures indicated growth of just 0.1% - and a similar-sized
    contraction in the previous quarter. On an annual basis, the data suggests
    annual growth of just 0.2%, suggesting a much more hesitant recovery than
    had previously been thought. A common technical definition of a recession
    is two successive quarters of negative growth.

    The government was keen to play down the worrying implications of the data.
    "I maintain the view that Japan's economy remains in a minor adjustment
    phase in an upward climb, and we will monitor developments carefully," said
    economy minister Heizo Takenaka. But in the face of the strengthening yen
    making exports less competitive and indications of weakening economic
    conditions ahead, observers were less sanguine. "It's painting a picture of
    a recovery... much patchier than previously thought," said Paul Sheard,
    economist at Lehman Brothers in Tokyo. Improvements in the job market
    apparently have yet to feed through to domestic demand, with private
    consumption up just 0.2% in the third quarter.

    """

    text_2 = """Dibaba breaks 5,000m world record

    Ethiopia's Tirunesh Dibaba set a new world record in winning the women's
    5,000m at the Boston Indoor Games.

    Dibaba won in 14 minutes 32.93 seconds to erase the previous world indoor
    mark of 14:39.29 set by another Ethiopian, Berhane Adera, in Stuttgart last
    year. But compatriot Kenenisa Bekele's record hopes were dashed when he
    miscounted his laps in the men's 3,000m and staged his sprint finish a lap
    too soon. Ireland's Alistair Cragg won in 7:39.89 as Bekele battled to
    second in 7:41.42. "I didn't want to sit back and get out-kicked," said
    Cragg. "So I kept on the pace. The plan was to go with 500m to go no matter
    what, but when Bekele made the mistake that was it. The race was mine."
    Sweden's Carolina Kluft, the Olympic heptathlon champion, and Slovenia's
    Jolanda Ceplak had winning performances, too. Kluft took the long jump at
    6.63m, while Ceplak easily won the women's 800m in 2:01.52.

    """

    text_3 = """Google's toolbar sparks concern

    Search engine firm Google has released a trial tool which is concerning
    some net users because it directs people to pre-selected commercial
    websites.

    The AutoLink feature comes with Google's latest toolbar and provides links
    in a webpage to Amazon.com if it finds a book's ISBN number on the site. It
    also links to Google's map service, if there is an address, or to car firm
    Carfax, if there is a licence plate. Google said the feature, available
    only in the US, "adds useful links". But some users are concerned that
    Google's dominant position in the search engine market place could mean it
    would be giving a competitive edge to firms like Amazon.

    AutoLink works by creating a link to a website based on information
    contained in a webpage - even if there is no link specified and whether or
    not the publisher of the page has given permission.

    If a user clicks the AutoLink feature in the Google toolbar then a webpage
    with a book's unique ISBN number would link directly to Amazon's website.
    It could mean online libraries that list ISBN book numbers find they are
    directing users to Amazon.com whether they like it or not. Websites which
    have paid for advertising on their pages may also be directing people to
    rival services. Dan Gillmor, founder of Grassroots Media, which supports
    citizen-based media, said the tool was a "bad idea, and an unfortunate move
    by a company that is looking to continue its hypergrowth". In a statement
    Google said the feature was still only in beta, ie trial, stage and that
    the company welcomed feedback from users. It said: "The user can choose
    never to click on the AutoLink button, and web pages she views will never
    be modified. "In addition, the user can choose to disable the AutoLink
    feature entirely at any time."

    The new tool has been compared to the Smart Tags feature from Microsoft by
    some users. It was widely criticised by net users and later dropped by
    Microsoft after concerns over trademark use were raised. Smart Tags allowed
    Microsoft to link any word on a web page to another site chosen by the
    company. Google said none of the companies which received AutoLinks had
    paid for the service. Some users said AutoLink would only be fair if
    websites had to sign up to allow the feature to work on their pages or if
    they received revenue for any "click through" to a commercial site. Cory
    Doctorow, European outreach coordinator for digital civil liberties group
    Electronic Fronter Foundation, said that Google should not be penalised for
    its market dominance. "Of course Google should be allowed to direct people
    to whatever proxies it chooses. "But as an end user I would want to know -
    'Can I choose to use this service?, 'How much is Google being paid?', 'Can
    I substitute my own companies for the ones chosen by Google?'." Mr Doctorow
    said the only objection would be if users were forced into using AutoLink
    or "tricked into using the service".

    """
    return [text_1, text_2, text_3]


class TextEmbedding(NamedTuple):
    content: str
    vector: list[float]


def get_text_embedding_dict(text: str) -> TextEmbedding:
    return TextEmbedding(text, get_embeddings(text))


def get_news_text_embeddings() -> list[TextEmbedding]:
    return [get_text_embedding_dict(text) for text in get_news_texts()]


def create_vector_index(
    index_name: str, dim: int, prefix: str, client: redis.Redis
):
    schema = (
        VectorField(
            "$.vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
        TextField("$.content", as_name="content"),
    )
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
    res = client.ft(index_name).create_index(
        fields=schema, definition=definition
    )
    return res


def index_documents(
    prefix: str, text_embeddings: list[TextEmbedding], client: redis.Redis
):
    for idx, text_embedding in enumerate(text_embeddings):
        key = f"{prefix}:{idx + 1}"
        res = client.json().set(key, "$", text_embedding._asdict())
        assert res, "Cannot save document as JSON"


def create_query(vector_field, return_fields, hybrid_fields):
    base_query = (
        f"{hybrid_fields}=>[KNN 3 @{vector_field} $query_vec AS vector_score]"
    )
    query = (
        Query(base_query)
        .sort_by("vector_score")
        .return_fields(*return_fields)
        .dialect(2)
    )
    return query


def search_redis(
    query_embeddings: list, index_name: str, query: Any, client: redis.Redis
):
    params_dict: Any = {
        "query_vec": np.array(query_embeddings)
        .astype(dtype=np.float32)
        .tobytes()
    }

    results = client.ft(index_name).search(query, params_dict)

    for doc in results.docs:  # type: ignore
        score = 1 - float(doc["vector_score"])
        content = doc["content"].strip()[:100]
        clean_text = " ".join(content.replace("\n", " ").split())
        print(f"  {clean_text[:100]} (Score: {round(score ,3) })")
    return results.docs  # type: ignore


def query_redis(query_text, hybrid_fields, index_name, client):
    print(f"{hybrid_fields=}")

    vector_field = "vector"
    return_fields = ["vector_score", "content"]

    print("--- Query text ----------------------")
    clean_text = " ".join(query_text.replace("\n", " ").split())
    print(clean_text[:100])
    query = create_query(vector_field, return_fields, hybrid_fields)
    query_embeddings = get_embeddings(query_text)
    print("Search similar text---------------------------")
    search_redis(query_embeddings, index_name, query, client)


def main():

    client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    print(f"client connected {client.ping()}")

    text_embeddings = get_news_text_embeddings()

    index_name = "idx"
    prefix = "doc:"

    assert len(text_embeddings) > 0, "Cannot get embeddings"
    dim = len(text_embeddings[0].vector)

    if not index_exists(index_name, client):
        res = create_vector_index(index_name, dim, prefix, client)
        assert res == "OK", "Cannot create vector index"
        print_indexing_failures(index_name, client)

    index_documents(prefix, text_embeddings, client)

    query_text1 = """Radcliffe yet to answer GB call

    Paula Radcliffe has been granted extra time to decide whether to compete in
    the World Cross-Country Championships.

    The 31-year-old is concerned the event, which starts on 19 March in France,
    could upset her preparations for the London Marathon on 17 April. "There is
    no question that Paula would be a huge asset to the GB team," said Zara
    Hyde Peters of UK Athletics. "But she is working out whether she can
    accommodate the worlds without too much compromise in her marathon
    training." Radcliffe must make a decision by Tuesday - the deadline for
    team nominations. British team member Hayley Yelling said the team would
    understand if Radcliffe opted out of the event. "It would be fantastic to
    have Paula in the team," said the European cross-country champion. "But you
    have to remember that athletics is basically an individual sport and
    anything achieved for the team is a bonus. "She is not messing us around.
    We all understand the problem." Radcliffe was world cross-country champion
    in 2001 and 2002 but missed last year's event because of injury. In her
    absence, the GB team won bronze in Brussels.
    """

    query_redis(query_text1, "*", index_name, client)

    query_text2 = """Ethiopia's crop production up 24%

    Ethiopia produced 14.27 million tonnes of crops in 2004, 24% higher than in
    2003 and 21% more than the average of the past five years, a report says.

    In 2003, crop production totalled 11.49 million tonnes, the joint report
    from the Food and Agriculture Organisation and the World Food Programme
    said. Good rains, increased use of fertilizers and improved seeds
    contributed to the rise in production. Nevertheless, 2.2 million Ethiopians
    will still need emergency assistance.

    The report calculated emergency food requirements for 2005 to be 387,500
    tonnes. On top of that, 89,000 tonnes of fortified blended food and
    vegetable oil for "targeted supplementary food distributions for a survival
    programme for children under five and pregnant and lactating women" will be
    needed.

    In eastern and southern Ethiopia, a prolonged drought has killed crops and
    drained wells. Last year, a total of 965,000 tonnes of food assistance was
    needed to help seven million Ethiopians. The Food and Agriculture
    Organisation (FAO) recommend that the food assistance is bought locally.
    "Local purchase of cereals for food assistance programmes is recommended as
    far as possible, so as to assist domestic markets and farmers," said Henri
    Josserand, chief of FAO's Global Information and Early Warning System.
    Agriculture is the main economic activity in Ethiopia, representing 45% of
    gross domestic product. About 80% of Ethiopians depend directly or
    indirectly on agriculture.
    """

    query_redis(query_text2, "*", index_name, client)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    # logging.basicConfig(level=logging.DEBUG)
    main()
