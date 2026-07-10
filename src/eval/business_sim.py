"""
Business-impact inputs: per-system offline ranking quality.
===========================================================

We compare three rails on identical held-out decision points and score each by
how well it surfaces the items the user *actually added* (logged ground truth —
no assumed choice model):

  * **baseline** — rule-based **popularity** ranking of the candidates.
  * **item2vec** — Item2Vec cart-similarity (cosine) ranking — retrieval only.
  * **full**     — retrieval + GRU + LightGBM ranker (+ cold-start routing).

For each we measure Recall@10 / Precision@10 / NDCG@10 of the added items, plus
the average added-item price and base cart value. `docs/business_impact.md` then
builds the calculation chain from these measured numbers + a stated 18% baseline
acceptance anchor to projected acceptance rate and AOV lift.

Outputs results/business_sim.json.
Run:  python -m src.eval.business_sim
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
import src.ranking.train_ranker as RK  # noqa: E402
from src.coldstart.llm_embeddings import ContentSignals  # noqa: E402
from src.coldstart import mmr_rerank as MM  # noqa: E402

TOP_N = 10
N_GROUPS = 6000


def rank_metrics(order_ids, relevant, y_true_full, y_score_full):
    top = order_ids[:TOP_N]
    hit = sum(1 for it in top if it in relevant)
    rec = hit / max(len(relevant), 1)
    prec = hit / TOP_N
    ndcg = ndcg_score(y_true_full.reshape(1, -1), y_score_full.reshape(1, -1), k=TOP_N)
    return rec, prec, float(ndcg)


def main():
    rng = np.random.default_rng(C.SEED + 3)
    print("Loading models ...", flush=True)
    T = RK.Tables()
    with open(ROOT / "outputs" / "lgbm_ranker.pkl", "rb") as f:
        ranker = pickle.load(f)["model"]
    items = pd.read_parquet(C.DATA_DIR / "items.parquet").sort_values("item_id").reset_index(drop=True)
    item_price = items["price"].to_numpy().astype(np.float32)
    item_pop = items["popularity"].to_numpy().astype(np.float32)
    is_new = items["is_new_item"].to_numpy()
    city_map = {c: i for i, c in enumerate(RK.CITIES)}
    item_city = items["restaurant_city"].map(city_map).to_numpy()
    # rule-based baseline: most-popular items per city (no cart-aware retrieval)
    city_pop = {ci: np.where(item_city == ci)[0][np.argsort(-item_pop[np.where(item_city == ci)[0]])]
                for ci in range(len(RK.CITIES))}
    llm_item = np.load(C.ARTIFACT_DIR / "llm_item_emb.npy")
    llm_user = np.load(C.ARTIFACT_DIR / "llm_user_emb.npy")
    cs = ContentSignals(llm_item, llm_user)
    protos = MM.category_prototypes(llm_item, T.item_cat, is_new)

    sessions = pd.read_parquet(C.DATA_DIR / "sessions.parquet").sort_values("start_timestamp").reset_index(drop=True)
    test_s = sessions.iloc[int(len(sessions) * C.TEMPORAL_SPLIT_QUANTILE):]
    groups = RK.sample_groups(test_s, N_GROUPS, rng)
    print(f"decision points: {len(groups):,}", flush=True)

    cart_emb, cart_query = RK.gru_encode(T, groups)
    cands, cart_pool = RK.faiss_retrieve(T, groups)
    X, _, _, names = RK.build_features(T, groups, cart_emb, cart_query, cands, cart_pool)
    Xn = RK.apply_feature_noise(X, names, RK.CALIB_NOISE, rng)
    full_scores = ranker.predict(Xn)
    K = RK.K_CAND

    sysN = ["baseline", "item2vec", "full"]
    M = {s: dict(rec=[], prec=[], ndcg=[]) for s in sysN}
    addon_prices, base_cart_vals, n_relevant, blind_recall = [], [], [], []

    for i, g in enumerate(groups):
        cand = cands[i]
        cart = np.asarray(g["cart"])
        cart_set = set(int(x) for x in cart)
        base_cart_vals.append(float(item_price[cart].sum()))
        relevant = {int(g["nxt"])} | {int(x) for x in g["future"]}
        relevant -= cart_set
        if not relevant:
            continue
        n_relevant.append(len(relevant))
        addon_prices.extend(float(item_price[it]) for it in relevant)

        # item2vec / full rank the FAISS-retrieved candidates
        y_ret = np.array([1.0 if int(c) in relevant else 0.0 for c in cand])
        cos_s = T.emb[cand] @ cart_pool[i]
        base = full_scores[i * K:(i + 1) * K]
        if T.u_cold[g["uid"]]:
            Xm = MM.mask_new_item_features(Xn[i * K:(i + 1) * K].copy(), names, cand, is_new)
            base = ranker.predict(Xm)
            cart_hist = np.bincount(T.item_cat[cart], minlength=RK.N_CAT).astype(np.float32)
            ccomp = MM.content_complementarity(cand, cart_hist, llm_item, protos)
            cc = cs.cart_centroid(cart)
            crel = 0.5 * cs.sim_to_vec(cand, cc) + 0.3 * (llm_item[cand] @ llm_user[g["uid"]])
            sig = np.maximum(0.0, MM._rank_norm(ccomp) - 0.55) + 0.15 * MM._rank_norm(crel)
            nm = (is_new[cand] == 1).astype(np.float32)
            full_s = MM._rank_norm(base) + MM.CONTENT_BOOST * sig * nm
        else:
            full_s = base
        # all three rank the SAME retrieved candidate pool (fair ranking comparison)
        pop_s = item_pop[cand]
        for s, sc in [("baseline", pop_s), ("item2vec", cos_s), ("full", full_s)]:
            order = [int(cand[j]) for j in np.argsort(-sc)]
            rec, prec, ndcg = rank_metrics(order, relevant, y_ret, np.asarray(sc, dtype=float))
            M[s]["rec"].append(rec); M[s]["prec"].append(prec); M[s]["ndcg"].append(ndcg)

        # context-blind popularity recall (real rule-based rail has no cart-aware
        # retrieval) — quantifies the candidate-generation value of Item2Vec
        cpool = np.array([it for it in city_pop[g["city"]] if int(it) not in cart_set])
        hit_blind = sum(1 for it in cpool[:TOP_N] if int(it) in relevant)
        blind_recall.append(hit_blind / len(relevant))

    out = dict(
        n_groups=len(n_relevant), top_n=TOP_N,
        avg_relevant_per_decision=round(float(np.mean(n_relevant)), 2),
        avg_addon_price=round(float(np.mean(addon_prices)), 1),
        avg_base_cart_value=round(float(np.mean(base_cart_vals)), 1),
        context_blind_popularity_recall_at_10=round(float(np.mean(blind_recall)), 4),
    )
    for s in sysN:
        out[s] = dict(recall_at_10=round(float(np.mean(M[s]["rec"])), 4),
                      precision_at_10=round(float(np.mean(M[s]["prec"])), 4),
                      ndcg_at_10=round(float(np.mean(M[s]["ndcg"])), 4))
    (C.RESULTS_DIR / "business_sim.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    rb, rf = out["baseline"]["recall_at_10"], out["full"]["recall_at_10"]
    print(f"\nRecall@10: baseline {rb:.3f} -> item2vec {out['item2vec']['recall_at_10']:.3f} "
          f"-> full {rf:.3f}  (full/baseline = {rf/rb:.2f}x)")
    return out


if __name__ == "__main__":
    main()
