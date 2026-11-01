"""Benchmark-specific source adapters."""

from .hotpotqa import adapt as adapt_hotpotqa
from .nq_wiki18 import adapt as adapt_nq_wiki18
from .twowiki import adapt as adapt_twowiki

ADAPTERS = {
    "hotpotqa_abstracts_2017": adapt_hotpotqa,
    "2wiki_ircot": adapt_twowiki,
    "wiki18": adapt_nq_wiki18,
}
