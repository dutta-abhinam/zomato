"""
Item2Vec: dense item embeddings from cart-session sequences.
===========================================================

Item2Vec (Barkan & Koenigstein, 2016) is word2vec applied to items: each
*basket* is a "sentence" and each item is a "word". Skip-gram with negative
sampling then learns embeddings where items that co-occur in the same cart sit
close together — exactly the "goes-with" signal we need for candidate retrieval
(e.g. Biryani ↔ Raita ↔ a soft drink).

Corpus: the ordered `seq_item_ids` of every session in `data/sessions.parquet`.

Outputs (artifacts/):
    item2vec.model        gensim Word2Vec model
    item2vec_emb.npy      (N_ITEMS, dim) float32, L2-normalised, row = item_id
                          rows for items never seen in a cart get a cuisine-mean
                          fallback so the matrix is complete for the ANN index.
    item2vec_meta.json    coverage + hyper-parameters

Run:  python -m src.retrieval.train_item2vec
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402


def build_corpus(sessions: pd.DataFrame) -> list[list[str]]:
    """Each session's ordered item sequence -> a sentence of item-id tokens."""
    corpus = []
    for seq in sessions["seq_item_ids"].to_numpy():
        if seq is None:
            continue
        toks = [str(int(x)) for x in seq]
        if len(toks) >= 2:                 # need >=2 items to form context pairs
            corpus.append(toks)
    return corpus


def train(corpus: list[list[str]]) -> Word2Vec:
    print(f"Training Item2Vec (skip-gram) on {len(corpus):,} sessions ...", flush=True)
    t0 = time.time()
    model = Word2Vec(
        sentences=corpus,
        vector_size=C.ITEM2VEC_DIM,
        window=C.ITEM2VEC_WINDOW,
        sg=1,                              # skip-gram
        negative=C.ITEM2VEC_NEG,           # negative sampling
        ns_exponent=0.75,
        min_count=C.ITEM2VEC_MIN_COUNT,
        epochs=C.ITEM2VEC_EPOCHS,
        sample=1e-4,
        workers=8,
        seed=C.SEED,
    )
    print(f"  trained in {time.time()-t0:.1f}s | vocab={len(model.wv):,}", flush=True)
    return model


def build_embedding_matrix(model: Word2Vec, items: pd.DataFrame):
    """Assemble an (N_ITEMS, dim) matrix aligned to item_id. Out-of-vocab items
    (never ordered) fall back to their cuisine mean vector so every item is
    retrievable; a coverage mask records which are genuinely learned."""
    n, dim = len(items), C.ITEM2VEC_DIM
    emb = np.zeros((n, dim), dtype=np.float32)
    in_vocab = np.zeros(n, dtype=bool)
    wv = model.wv
    for iid in range(n):
        key = str(iid)
        if key in wv:
            emb[iid] = wv[key]
            in_vocab[iid] = True

    # cuisine-mean fallback for OOV items
    cuisine = items["cuisine"].to_numpy()
    for cui in np.unique(cuisine):
        mask = (cuisine == cui) & in_vocab
        miss = (cuisine == cui) & ~in_vocab
        if miss.any():
            fill = emb[mask].mean(0) if mask.any() else emb[in_vocab].mean(0)
            emb[miss] = fill
    # any remaining zero rows -> global mean
    zero = ~np.any(emb, axis=1)
    if zero.any():
        emb[zero] = emb[in_vocab].mean(0)

    # L2-normalise -> inner product == cosine in FAISS
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb, in_vocab


def sanity_neighbours(model, items, examples=6):
    """Print nearest neighbours for a few popular items as a smell test."""
    wv = model.wv
    id2name = dict(zip(items["item_id"], items["name"]))
    id2cui = dict(zip(items["item_id"], items["cuisine"]))
    # pick a few frequent, recognisable items present in vocab
    seeds = []
    for iid in items.sort_values("popularity", ascending=False)["item_id"].tolist():
        if str(iid) in wv:
            seeds.append(iid)
        if len(seeds) >= examples:
            break
    lines = []
    for iid in seeds:
        nbrs = wv.most_similar(str(iid), topn=5)
        nn = ", ".join(f"{id2name.get(int(k),k)} ({id2cui.get(int(k),'?')})" for k, _ in nbrs)
        lines.append(f"  [{id2name.get(iid)} | {id2cui.get(iid)}] -> {nn}")
    print("Nearest-neighbour sanity check:")
    print("\n".join(lines))
    return lines


def main():
    sessions = pd.read_parquet(C.DATA_DIR / "sessions.parquet")
    items = pd.read_parquet(C.DATA_DIR / "items.parquet").sort_values("item_id").reset_index(drop=True)

    corpus = build_corpus(sessions)
    tokens = sum(len(s) for s in corpus)
    print(f"corpus: {len(corpus):,} sentences, {tokens:,} tokens", flush=True)

    model = train(corpus)
    emb, in_vocab = build_embedding_matrix(model, items)

    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(C.ARTIFACT_DIR / "item2vec.model"))
    np.save(C.ARTIFACT_DIR / "item2vec_emb.npy", emb)
    np.save(C.ARTIFACT_DIR / "item2vec_in_vocab.npy", in_vocab)

    nn_lines = sanity_neighbours(model, items)

    meta = dict(
        dim=C.ITEM2VEC_DIM, window=C.ITEM2VEC_WINDOW, sg=1, negative=C.ITEM2VEC_NEG,
        epochs=C.ITEM2VEC_EPOCHS, min_count=C.ITEM2VEC_MIN_COUNT,
        n_sentences=len(corpus), n_tokens=int(tokens),
        vocab_size=int(len(model.wv)), n_items=int(len(items)),
        coverage=float(in_vocab.mean()),
    )
    (C.RESULTS_DIR / "item2vec_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nItem2Vec meta:\n" + json.dumps(meta, indent=2))
    print(f"coverage: {in_vocab.mean():.1%} of items have a learned embedding "
          f"({(~in_vocab).sum():,} OOV filled with cuisine-mean).")
    return meta


if __name__ == "__main__":
    main()
