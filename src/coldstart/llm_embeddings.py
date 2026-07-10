"""
LLM / content embeddings for the cold-start segment.
====================================================

Collaborative signal (Item2Vec, Phase 2) is only as good as a user's / item's
interaction history. For the **30% cold-start** users and the ~1.8K never-ordered
items it is sparse or absent. We substitute a **content** signal derived from
free text with a sentence-transformer (`all-MiniLM-L6-v2`, 384-d):

  * **Item content embedding** — encode each item's free-text menu description
    ("Chicken Biryani: aromatic Biryani non-vegetarian rice dish, great with
    raita and a soft drink."). Every item, cold or warm, gets a dense vector — no
    interaction history required.
  * **User content embedding** — encode a short profile sentence built from the
    *available* cold-start context (city, segment, veg preference, price band).
    Profile text depends only on a handful of buckets, so we encode the unique
    combinations and map them back to users (fast).

These plug into the cold-start pipeline two ways:
  1. **Blended retrieval** — candidate score = α·Item2Vec-sim + (1-α)·content-sim,
     so semantically-relevant items surface even with no collaborative signal.
  2. **Content relevance features** for the re-ranker (see `mmr_rerank.py`).

Outputs (artifacts/):
    llm_item_emb.npy   (N_ITEMS, 384) float32, L2-normalised
    llm_user_emb.npy   (N_USERS, 384) float32, L2-normalised

Run:  python -m src.coldstart.llm_embeddings
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

MODEL_NAME = C.LLM_MODEL_NAME       # all-MiniLM-L6-v2
EMB_DIM = C.LLM_EMB_DIM             # 384


# --------------------------------------------------------------------------- #
# User profile text (from available cold-start context only)
# --------------------------------------------------------------------------- #
def _veg_bucket(v):
    return "vegetarian" if v > 0.6 else "non-vegetarian" if v < 0.4 else "both vegetarian and non-vegetarian"


def _price_bucket(p):
    return "budget-friendly" if p > 0.6 else "premium" if p < 0.35 else "mid-priced"


def user_profile_text(city, segment, veg, price):
    return (f"A {segment} customer in {city} who orders food online, prefers "
            f"{_veg_bucket(veg)} dishes and {_price_bucket(price)} options.")


def build_user_texts(users: pd.DataFrame):
    """Return (texts_per_user, unique_texts, inverse_index) — dedup for speed."""
    veg_b = np.where(users["veg_pref"].to_numpy() > 0.6, "veg",
                     np.where(users["veg_pref"].to_numpy() < 0.4, "non", "both"))
    pr_b = np.where(users["price_sensitivity"].to_numpy() > 0.6, "budget",
                    np.where(users["price_sensitivity"].to_numpy() < 0.35, "premium", "mid"))
    keys = (users["city"].astype(str) + "|" + users["segment"].astype(str) + "|" + veg_b + "|" + pr_b)
    texts = [user_profile_text(c, s, v, p) for c, s, v, p in zip(
        users["city"], users["segment"], users["veg_pref"], users["price_sensitivity"])]
    uniq, inv = np.unique(keys.to_numpy(), return_inverse=True)
    # one representative text per unique key
    rep = {}
    for t, k in zip(texts, keys.to_numpy()):
        rep.setdefault(k, t)
    uniq_texts = [rep[k] for k in uniq]
    return uniq_texts, inv


# --------------------------------------------------------------------------- #
# Content signal helper (used by retrieval + re-ranking)
# --------------------------------------------------------------------------- #
class ContentSignals:
    def __init__(self, item_emb: np.ndarray, user_emb: np.ndarray | None = None):
        self.item_emb = item_emb                        # (N_ITEMS, D) normalised
        self.user_emb = user_emb                        # (N_USERS, D) normalised

    def cart_centroid(self, cart_ids) -> np.ndarray:
        v = self.item_emb[np.asarray(cart_ids)].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    def sim_to_vec(self, cand_ids, vec) -> np.ndarray:
        return self.item_emb[np.asarray(cand_ids)] @ vec

    def blended_retrieve(self, cart_ids, item2vec_emb, alpha, k=100, exclude=None):
        """Candidate score = α·Item2Vec-sim + (1-α)·content-sim to the cart."""
        cart = np.asarray(cart_ids)
        cf = item2vec_emb[cart].mean(0); cf /= (np.linalg.norm(cf) + 1e-8)
        ct = self.cart_centroid(cart)
        cf_sim = item2vec_emb @ cf
        ct_sim = self.item_emb @ ct
        score = alpha * cf_sim + (1 - alpha) * ct_sim
        drop = set(int(x) for x in cart) | set(exclude or [])
        order = np.argsort(-score)
        out = [int(i) for i in order if int(i) not in drop][:k]
        return out


# --------------------------------------------------------------------------- #
def encode_texts(model, texts, batch=256, desc=""):
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    print(f"  encoded {len(texts):,} {desc} in {time.time()-t0:.0f}s", flush=True)
    return emb.astype(np.float32)


def main():
    from sentence_transformers import SentenceTransformer
    items = pd.read_parquet(C.DATA_DIR / "items.parquet").sort_values("item_id").reset_index(drop=True)
    users = pd.read_parquet(C.DATA_DIR / "users.parquet").sort_values("user_id").reset_index(drop=True)

    print(f"Loading sentence-transformer '{MODEL_NAME}' ...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding item descriptions ...", flush=True)
    item_emb = encode_texts(model, items["description"].tolist(), desc="items")

    print("Encoding user profiles (deduplicated) ...", flush=True)
    uniq_texts, inv = build_user_texts(users)
    uniq_emb = encode_texts(model, uniq_texts, desc=f"unique profiles (of {len(users):,} users)")
    user_emb = uniq_emb[inv]

    np.save(C.ARTIFACT_DIR / "llm_item_emb.npy", item_emb)
    np.save(C.ARTIFACT_DIR / "llm_user_emb.npy", user_emb)

    # quick content-neighbour sanity check
    cs = ContentSignals(item_emb)
    id2name = items["name"].to_numpy(); id2cat = items["category"].to_numpy()
    probe = int(items.index[items["category"] == "dessert"][0])
    sims = item_emb @ item_emb[probe]
    nbrs = np.argsort(-sims)[1:6]
    print(f"content neighbours of [{id2name[probe]} | {id2cat[probe]}]:")
    for j in nbrs:
        print(f"    {id2name[j]} | {id2cat[j]}  (cos={sims[j]:.3f})")

    meta = dict(model=MODEL_NAME, dim=EMB_DIM, n_items=int(len(items)),
                n_users=int(len(users)), n_unique_profiles=int(len(uniq_texts)))
    (C.RESULTS_DIR / "llm_embeddings_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved llm_item_emb.npy, llm_user_emb.npy")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
