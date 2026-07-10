"""
Real-time CSAO serving service (FastAPI).
=========================================

On every **cart-update event** the service returns a freshly-ranked rail of the
top 8–10 add-on items, within the 200–300 ms SLA. Request path:

  1. **Feature store (Redis)** — pull cached user/session features by `user_id`.
  2. **Sharded FAISS retrieval** — route to the user's **city shard** and pull
     ~100 candidate items (Item2Vec ANN). Sharding keeps each index small so
     latency stays flat as the catalogue grows to millions of items.
  3. **GRU encoder + LightGBM ranker** — encode the ordered cart into the
     cart-state vector, build the feature row per candidate, score with the
     LambdaRank model.
  4. **MMR re-rank (cold-start)** — for cold-start users, recover newly-listed
     items via the LLM content signal and diversify with MMR.
  5. Return the **top-N** ranked items.

Everything (embeddings, FAISS shards, GRU, ranker, LLM vectors) is loaded once at
startup; only lightweight user/session features live in Redis.

Run the server:   python -m src.serving.serve
                  # or: uvicorn src.serving.serve:app --port 8000
Load test:        python -m src.serving.load_test
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
import src.ranking.train_ranker as RK  # noqa: E402
from src.coldstart.llm_embeddings import ContentSignals  # noqa: E402
from src.coldstart import mmr_rerank as MM  # noqa: E402

SERVE_RETRIEVE_K = 100          # candidates pulled from the shard per request
TOP_N = C.TOP_N_DISPLAY         # rail size returned (8–10)
LUNCH = set(C.LUNCH_HOURS); DINNER = set(C.DINNER_HOURS)


# --------------------------------------------------------------------------- #
# Sharded FAISS index (shard by city; IVF within shard for scale)
# --------------------------------------------------------------------------- #
class ShardedFaiss:
    """One ANN index per city. Each shard holds only that city's items, so search
    cost scales with items-per-city, not the whole catalogue. Within a shard an
    IVF index gives sub-linear search once a city grows large; for the current
    ~1.9K items/city we keep exact FlatIP (already sub-millisecond)."""

    def __init__(self, emb: np.ndarray, item_city: np.ndarray, n_city: int, use_ivf_threshold=20000):
        self.dim = emb.shape[1]
        self.shards, self.ids = {}, {}
        for c in range(n_city):
            ids = np.where(item_city == c)[0].astype(np.int64)
            if len(ids) == 0:
                continue
            if len(ids) >= use_ivf_threshold:
                nlist = max(16, int(np.sqrt(len(ids))))
                quant = faiss.IndexFlatIP(self.dim)
                idx = faiss.IndexIVFFlat(quant, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                idx.train(emb[ids]); idx.add(emb[ids]); idx.nprobe = 8
            else:
                idx = faiss.IndexFlatIP(self.dim); idx.add(emb[ids])
            self.shards[c], self.ids[c] = idx, ids

    def retrieve(self, city: int, qvec: np.ndarray, cart_set: set, k: int):
        idx, ids = self.shards[city], self.ids[city]
        n = min(idx.ntotal, k + len(cart_set) + 5)
        _, I = idx.search(qvec.reshape(1, -1).astype(np.float32), n)
        out = [int(ids[i]) for i in I[0] if i >= 0 and int(ids[i]) not in cart_set]
        return out[:k]

    def stats(self):
        return {c: int(idx.ntotal) for c, idx in self.shards.items()}


# --------------------------------------------------------------------------- #
# Serving engine — loads everything once, serves single requests
# --------------------------------------------------------------------------- #
class ServingEngine:
    def __init__(self):
        import pickle
        t0 = time.time()
        self.T = RK.Tables()
        with open(ROOT / "outputs" / "lgbm_ranker.pkl", "rb") as f:
            self.ranker = pickle.load(f)["model"]
        self.emb = self.T.emb                                   # Item2Vec (normalised)
        self.llm_item = np.load(C.ARTIFACT_DIR / "llm_item_emb.npy")
        self.llm_user = np.load(C.ARTIFACT_DIR / "llm_user_emb.npy")
        self.cs = ContentSignals(self.llm_item, self.llm_user)

        items = pd.read_parquet(C.DATA_DIR / "items.parquet").sort_values("item_id").reset_index(drop=True)
        self.item_name = items["name"].to_numpy()
        self.is_new = items["is_new_item"].to_numpy()
        self.item_cat = self.T.item_cat
        city_map = {c: i for i, c in enumerate(RK.CITIES)}
        self.item_city = items["restaurant_city"].map(city_map).to_numpy()
        self.protos = MM.category_prototypes(self.llm_item, self.item_cat, self.is_new)

        faiss.omp_set_num_threads(1)                            # per-request latency realism
        self.index = ShardedFaiss(self.emb, self.item_city, len(RK.CITIES))

        self.redis = self._build_feature_store()
        self.load_sec = time.time() - t0

    # ---- Redis feature store (fakeredis; drop-in for real Redis) ----
    def _build_feature_store(self):
        import fakeredis, json
        r = fakeredis.FakeStrictRedis()
        pipe = r.pipeline()
        u_city = self.T.u_city; u_cold = self.T.u_cold; u_seg = self.T.u_seg
        for uid in range(len(u_city)):
            pipe.set(f"user:{uid}", json.dumps(dict(
                city=int(u_city[uid]), is_cold=int(u_cold[uid]), seg=int(u_seg[uid]))))
            if uid % 10000 == 0:
                pipe.execute()
        pipe.execute()
        return r

    def fetch_user(self, user_id: int):
        import json
        raw = self.redis.get(f"user:{user_id}")
        if raw is None:
            return dict(city=0, is_cold=1, seg=1)               # unknown -> treat as cold
        return json.loads(raw)

    # ---- main request handler ----
    def recommend(self, user_id: int, cart_item_ids, timestamps=None, hour=13, top_n=TOP_N):
        uf = self.fetch_user(user_id)                           # (1) Redis features
        city, is_cold = uf["city"], uf["is_cold"]
        cart = [int(x) for x in cart_item_ids][:RK.MAX_ITEMS]
        if not cart:
            return []
        meal = 1 if 11 <= hour <= 14 else 3 if 19 <= hour <= 22 else 2
        is_peak = int(hour in LUNCH or hour in DINNER)
        ts = timestamps or list(range(0, len(cart) * 60, 60))
        group = dict(cart=cart, acc=[0] * len(cart), ts=ts, nxt=-1, future=set(),
                     uid=int(user_id), city=int(city), hour=int(hour), meal=int(meal), peak=is_peak)

        # (2) sharded FAISS retrieval
        cvec = self.emb[np.asarray(cart)].mean(0)
        cvec = cvec / (np.linalg.norm(cvec) + 1e-8)
        cand = self.index.retrieve(city, cvec, set(cart), SERVE_RETRIEVE_K)
        if not cand:
            return []
        cands = np.asarray(cand, dtype=np.int64).reshape(1, -1)

        # (3) GRU encode + LightGBM rank
        cart_emb, cart_query = RK.gru_encode(self.T, [group])
        X, _, _, names = RK.build_features(self.T, [group], cart_emb, cart_query, cands,
                                           cvec.reshape(1, -1))
        if is_cold:
            X = MM.mask_new_item_features(X, names, cands.reshape(-1), self.is_new)
        scores = self.ranker.predict(X)

        # (4) cold-start: LLM content recovery + MMR
        if is_cold:
            cart_hist = np.bincount(self.item_cat[np.asarray(cart)], minlength=RK.N_CAT).astype(np.float32)
            ccomp = MM.content_complementarity(cand, cart_hist, self.llm_item, self.protos)
            cc = self.cs.cart_centroid(cart)
            crel = 0.5 * self.cs.sim_to_vec(cand, cc) + 0.3 * (self.llm_item[cand] @ self.llm_user[user_id])
            sig = np.maximum(0.0, MM._rank_norm(ccomp) - 0.55) + 0.15 * MM._rank_norm(crel)
            new_mask = (self.is_new[cand] == 1).astype(np.float32)
            blend = MM._rank_norm(scores) + MM.CONTENT_BOOST * sig * new_mask
            ranked = MM.mmr_rerank(cand, blend, self.llm_item, lam=MM.MMR_LAMBDA, top_n=top_n)
            order_scores = {it: float(blend[cand.index(it)]) for it in ranked}
        else:
            top_idx = np.argsort(-scores)[:top_n]
            ranked = [int(cand[j]) for j in top_idx]
            order_scores = {int(cand[j]): float(scores[j]) for j in top_idx}

        # (5) return top-N
        return [dict(item_id=it, name=str(self.item_name[it]),
                     score=round(order_scores.get(it, 0.0), 4),
                     is_new_item=int(self.is_new[it]))
                for it in ranked[:top_n]]


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class CartUpdate(BaseModel):
        user_id: int
        cart_item_ids: list[int]
        timestamps: list[int] | None = None
        hour: int = 13
        top_n: int = TOP_N

    app = FastAPI(title="CSAO Rail Recommender", version="1.0")
    _engine: ServingEngine | None = None

    @app.on_event("startup")
    def _startup():
        global _engine
        _engine = ServingEngine()
        print(f"[serve] engine ready in {_engine.load_sec:.1f}s | shard sizes {_engine.index.stats()}", flush=True)

    @app.get("/health")
    def health():
        return {"status": "ok", "shards": _engine.index.stats() if _engine else None}

    @app.post("/recommend")
    def recommend(req: CartUpdate):
        t0 = time.perf_counter()
        rail = _engine.recommend(req.user_id, req.cart_item_ids, req.timestamps, req.hour, req.top_n)
        return {"rail": rail, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
except ImportError:  # FastAPI not installed — engine still usable directly
    app = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.serve:app", host="127.0.0.1", port=8000, workers=1, log_level="warning")
