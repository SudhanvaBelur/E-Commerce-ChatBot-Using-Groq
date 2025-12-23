from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
import re
from typing import Optional

encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)

faq = Route(
    name="faq",
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "What other payment options do you support?",
        "do you support cash on delivery?",
        "How long does it take to process a refund?",
    ]
)

sql = Route(
    name="sql",
    utterances=[
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
        "give me the top 3 shoes in descending order of rating",
    ]
)
routes = [faq, sql]
router = SemanticRouter(routes=routes, encoder=encoder, auto_sync="local")

faq.metadata = {"keywords": ["return", "refund", "track", "order", "payment", "delivery", "cancel", "refund policy"]}
sql.metadata = {"keywords": ["buy", "price", "discount", "shoes", "size", "sale", "nike", "puma","top 3", "ratings", "average price", "price range"]}


# Simple keyword-based router: returns a route when query contains keywords
def keyword_route(query: str, routes: list[Route]):
    q = query.lower()
    best = None
    best_count = 0
    for r in routes:
        kws = []
        if getattr(r, "metadata", None) and isinstance(r.metadata, dict):
            kws = r.metadata.get("keywords", [])
        # count keyword occurrences (simple substring match)
        count = sum(1 for kw in kws if kw.lower() in q)
        if count > best_count:
            best_count = count
            best = r
    return best if best_count > 0 else None


# Improved hybrid routing: keyword -> semantic -> fuzzy-keyword
SYNONYMS = {
    "refund": ["refund", "return", "reimburse"],
    "track": ["track", "tracking", "where is", "where's"],
    "payment": ["payment", "pay", "checkout", "cash on delivery", "cod"],
    "discount": ["discount", "sale", "offer", "deal"],
}


def _match_keyword_boundary(query: str, kw: str) -> bool:
    # match whole word or phrase using word boundaries
    pattern = r"\b" + re.escape(kw.lower()) + r"\b"
    return re.search(pattern, query.lower()) is not None


def fuzzy_keyword_route(query: str, routes: list[Route]) -> Optional[Route]:
    """Looser matching: looks for keywords or synonyms as whole words in query."""
    q = query.lower()
    best = None
    best_count = 0
    for r in routes:
        kws = []
        if getattr(r, "metadata", None) and isinstance(r.metadata, dict):
            kws = r.metadata.get("keywords", [])
        count = 0
        for kw in kws:
            if _match_keyword_boundary(q, kw):
                count += 2
            else:
                # try synonyms
                for syns in SYNONYMS.values():
                    for syn in syns:
                        if _match_keyword_boundary(q, syn):
                            count += 1
                            break
        if count > best_count:
            best_count = count
            best = r
    return best if best_count > 0 else None


def route_query(query: str, routes: list[Route]):
    """Return best Route for a natural-language `query`.

    Strategy:
    1. Exact keyword metadata match (strong signal)
    2. SemanticRouter (embedding-based)
    3. Fuzzy keyword + synonyms (fallback)
    """
    # 1) exact metadata keywords
    k = keyword_route(query, routes)
    if k:
        return k

    # 2) semantic router
    try:
        sem = router(query)
        if sem is not None:
            return sem
    except Exception:
        # if index isn't ready or other issue, fall through
        pass

    # 3) fuzzy keyword match
    return fuzzy_keyword_route(query, routes)

# Ensure the index is initialized and routes are added before performing inference
try:
    router._init_index_state()
except Exception:
    # fallback to async init if required by the index implementation
    import asyncio

    asyncio.run(router._async_init_index_state())

# Add routes to the index (if not already present)
router.add([faq, sql])

if __name__ == "__main__":
    print(router("How can I track my order?").name)
    print(router("Pink Puma shoes in price range 5000 to 1000").name)