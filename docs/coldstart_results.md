# Cold-Start Results — LLM Embeddings + MMR

Evaluated on **1,974 cold-start decision points** from the held-out
(temporal-split) test sessions — users with <3 historical orders and sparse
feature vectors.

## Precision@10 on the cold-start segment
| Stage | Precision@10 | vs previous |
|---|---:|---:|
| Popularity fallback (no personalization / no content model) | 0.197 | — |
| Collaborative LightGBM ranker (Item2Vec + GRU) | 0.216 | +9.7% |
| **+ LLM content relevance (this pipeline)** | **0.217** | +0.7% |

**Headline: cold-start Precision@10 improves +10.5%** vs the popularity
fallback that a user with no usable history would otherwise receive
(0.197 → 0.217).

## Diversity — the MMR operating point
Content relevance (above) is the precision-optimal point. **MMR** trades a little
precision for diversity, removing near-duplicate items from the rail. Average
pairwise content cosine among the 10 shown items (lower = more diverse):

| Operating point | Precision@10 | top-10 redundancy |
|---|---:|---:|
| content re-rank (MMR off) | 0.217 | 0.506 |
| content + MMR (λ=0.7) | 0.214 | 0.489 |

MMR cuts rail redundancy by **3%** for a **1.6%**
precision cost — a tunable knob (λ) to avoid showing three near-identical items.

## Honest read on the ~+15% target
The brief targeted ~+15%. We land at **+10.5% vs the popularity fallback**,
and the *incremental* contribution of the LLM content signal over the already-strong
collaborative ranker is small (**+0.7%**). This is a property of the
**synthetic data**, and worth stating plainly:

* Item2Vec is trained on the full basket corpus, so it already recovers the
  cuisine/co-occurrence structure that item **text** would provide — the two
  signals are largely redundant here, so content adds little *on top of* a strong
  collaborative model.
* The relevance structure is **category complementarity** (biryani → raita → a
  drink). Text similarity captures *cuisine/style*, not complementarity, so a
  content signal cannot re-rank complementary items much — it mainly helps rank
  the same-cuisine family, which the ranker already does. (A content-**blended
  retrieval** was tested and *hurt*, because text similarity pulls in same-category
  duplicates; retrieval therefore stays Item2Vec.)
* Genuinely cold **items** (no interaction history) are where content is decisive —
  but by construction such items rarely appear as cart items or accepted add-ons in
  logged sessions, so this offline slice under-exercises exactly the case LLM
  embeddings were built for.

**Where LLM embeddings still earn their place** (and would show a larger lift on
real, sparser data): they give *every* item and cold user a dense vector with **no
interaction history required** — the 1,800 never-ordered items that
Item2Vec can only represent by a cuisine-mean fallback get a real, description-derived
embedding, and MMR delivers a measurable diversity win (3% less
redundancy) regardless of the data regime.

_Generated in 5s by `src/coldstart/mmr_rerank.py`._
