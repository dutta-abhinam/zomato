# Cart Super Add-On (CSAO) Rail Recommendation System

A real-time recommendation engine that suggests complementary add-on items as a
customer builds their cart on a food-delivery app. Given the **current cart
state** (an ordered sequence of items) plus **context** (time of day, city,
restaurant, user), the system returns a ranked rail of **N = 8–10** add-ons the
user is most likely to accept — updating live as items are added or removed
(e.g. *Biryani → recommend Salan → add Salan → recommend Gulab Jamun → add →
recommend a drink*).

Built for **Zomathon Problem Statement 2 (CSAO Rail)**. Since no dataset is
provided, we generate our own realistic food-delivery data and document the full
pipeline end-to-end.

---

## Target outcomes

- **Data:** ~1.2M+ candidate interactions across **50K users** & **15K items**,
  simulating city-wise behaviour, **3× peak-hour** spikes, and **30% cold-start**
  users.
- **Model:** Item2Vec + FAISS retrieval → LightGBM **LambdaRank** with a
  **GRU-encoded cart state**. Targets **AUC 0.85** and **NDCG@10 ≈ 0.61**.
- **Cold-start:** LLM/content embeddings for the 30% sparse-history users,
  **+15% Precision@10** via **MMR** re-ranking.
- **Serving:** **~180 ms** end-to-end via a Redis feature store + FAISS
  sharding; projected **+14% AOV lift** and **>40% acceptance rate** against an
  **18%** popularity baseline.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   Cart state +          │                 SERVING LAYER                 │
   context (user,        │        (Redis feature store, ~180 ms)         │
   city, time, rest.)    └──────────────────────────────────────────────┘
        │                                    │
        ▼                                    ▼
┌───────────────────┐   warm users    ┌───────────────────┐
│  STAGE 1          │────────────────▶│  STAGE 2           │
│  RETRIEVAL        │  ~120 candidates│  RANKING           │
│                   │  from 15K items │                    │
│  Item2Vec vectors │                 │  GRU cart encoder  │
│  + FAISS ANN      │                 │  → cart-state emb  │
│  (sharded by city)│                 │  + Item2Vec + ctx  │
└───────────────────┘                 │  → LightGBM        │
        │                             │    LambdaRank      │
        │ cold-start / sparse users   └───────────────────┘
        ▼                                    │
┌───────────────────┐                        ▼
│  COLD-START PATH  │                 ┌───────────────────┐
│                   │                 │  RE-RANK           │
│  LLM (MiniLM)     │────────────────▶│  MMR diversity     │
│  content emb +    │                 │  → Top-N (8–10)    │
│  FAISS content ANN│                 └───────────────────┘
└───────────────────┘                        │
                                             ▼
                                     Ranked add-on rail
```

### Two-stage design

1. **Retrieval (candidate generation).** **Item2Vec** (word2vec trained on order
   baskets) learns item co-occurrence embeddings. A **FAISS** index (sharded by
   city for scale) returns ~120 nearest neighbours to the pooled cart vector,
   narrowing 15K items to a cheap candidate set.

2. **Ranking.** A **GRU** encodes the *ordered* cart into a cart-state embedding
   (capturing "what's already here and in what order"). This embedding is
   concatenated with Item2Vec candidate vectors, complementarity/meal-completion
   features, user affinity, price-fit and temporal/geo context, then scored by a
   **LightGBM LambdaRank** model optimised for NDCG.

3. **Cold-start fallback.** For the 30% of users with sparse/no history,
   collaborative Item2Vec signal is weak, so we retrieve on **LLM content
   embeddings** (`all-MiniLM-L6-v2` over item name/metadata). **MMR** re-ranking
   trades relevance for diversity to lift Precision@10 and avoid rail fatigue.

4. **Serving.** Precomputed features/embeddings live in **Redis**; FAISS shards
   keep ANN latency flat under load. End-to-end request budget: **200–300 ms**.

---

## Repository layout

```
.
├── data/                 # generated datasets (parquet / npz) — git-ignored
├── notebooks/            # exploratory analysis & result walkthroughs
├── outputs/              # metrics, figures, trained artifacts, submission PDF
├── docs/                 # design notes, architecture, submission write-up
├── src/
│   ├── data/             # synthetic data generator + choice model
│   ├── features/         # feature engineering / offline+online feature build
│   ├── retrieval/        # Item2Vec training + FAISS index (sharded)
│   ├── ranking/          # GRU cart encoder + LightGBM LambdaRank
│   ├── coldstart/        # LLM content embeddings + MMR re-ranking
│   ├── serving/          # Redis feature store + end-to-end latency benchmark
│   └── eval/             # offline metrics + business-impact simulation
├── config.py             # single source of truth for paths / sizes / hparams
├── requirements.txt
└── README.md
```

---

## Pipeline stages

| Stage | Module | Output |
|-------|--------|--------|
| 1. Generate data | `src/data/generate.py` | users, items, restaurants, orders, impressions, candidates |
| 2. Item2Vec + FAISS | `src/retrieval/` | item embeddings, ANN index (sharded) |
| 3. Feature build | `src/features/` | training feature matrix + group index |
| 4. GRU + LambdaRank | `src/ranking/` | cart encoder, ranker, offline metrics |
| 5. Cold-start + MMR | `src/coldstart/` | content index, MMR re-ranker |
| 6. Serving | `src/serving/` | Redis feature store, latency p50/p95 |
| 7. Eval + business | `src/eval/` | AUC/NDCG/P@K/R@K, AOV lift, A/B design |

---

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.generate            # 1. synthesize dataset
python -m src.retrieval.item2vec       # 2. train Item2Vec + build FAISS
python -m src.features.build_features   # 3. feature engineering
python -m src.ranking.train_ranker      # 4. GRU + LambdaRank
python -m src.coldstart.build           # 5. LLM embeddings + MMR
python -m src.serving.benchmark         # 6. latency benchmark
python -m src.eval.report               # 7. metrics + business + PDF
```

---

## Evaluation

- **Offline (temporal split** — last 20% of time held out to prevent leakage):
  AUC, Precision@K, Recall@K, NDCG@10, with **per-segment** breakdowns
  (cold vs warm, city, meal-time).
- **Business impact:** projected AOV lift and acceptance rate via a counterfactual
  simulation against the popularity baseline, plus a proposed **A/B testing**
  framework with guardrail metrics (cart abandonment, C2O).

> **Note:** all data is synthetic and self-generated; the generative "choice
> model" encodes realistic complementarity, personalization and context so that
> the learned models recover genuine signal. See `docs/` for assumptions.
