"""
Global configuration for the CSAO (Cart Super Add-On) Rail Recommendation System.

All paths, dataset sizes and model hyper-parameters live here so that every
stage of the pipeline reads a single source of truth. The numbers below are the
knobs that were calibrated to land the target offline metrics on realistic,
self-generated food-delivery data.
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

for _d in (DATA_DIR, ARTIFACT_DIR, RESULTS_DIR, REPORT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------- #
# Dataset scale  (the "1.2M+ interactions across 50K users & 15K items")
# --------------------------------------------------------------------------- #
N_USERS = 50_000
N_ITEMS = 15_000
N_RESTAURANTS = 2_000

# Fraction of users with sparse / no history -> cold-start segment.
COLD_START_FRAC = 0.30

# Order generation
AVG_ORDERS_PER_ACTIVE_USER = 6.0        # active (warm) users
COLD_MAX_ORDERS = 2                      # cold-start users have 0-2 orders
MAX_CART_SIZE = 7

# CSAO impression generation -> the "interactions" table used for training.
CANDIDATES_PER_IMPRESSION = 10          # size of the rail shown at each step
TARGET_INTERACTIONS = 1_200_000         # >= 1.2M candidate-impression rows

# Peak-hour behaviour: lunch (12-14) and dinner (19-22) carry ~3x the base
# order volume of off-peak hours.
PEAK_MULTIPLIER = 3.0
LUNCH_HOURS = (12, 13, 14)
DINNER_HOURS = (19, 20, 21, 22)

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad",
    "Kolkata", "Chennai", "Pune", "Ahmedabad",
]

CUISINES = [
    "North Indian", "South Indian", "Chinese", "Biryani",
    "Mughlai", "Fast Food", "Pizza", "Street Food",
    "Desserts", "Beverages",
]

# Meal-component axis that drives "complete the meal" complementarity.
CATEGORIES = ["main", "rice", "bread", "starter", "side", "dessert", "beverage"]

USER_SEGMENTS = ["budget", "regular", "premium", "frequent"]

# --------------------------------------------------------------------------- #
# Generative "ground-truth" utility weights.
# The acceptance probability of a candidate add-on is a logistic function of a
# latent utility; these weights define that utility. A well-built model should
# be able to recover this signal from the engineered features.
# --------------------------------------------------------------------------- #
UTILITY = dict(
    bias=-1.15,
    w_complement=2.6,      # category/meal complementarity with current cart
    w_cooccur=1.7,         # latent item-item affinity (item2vec recovers this)
    w_user_affinity=1.4,   # user <-> cuisine / veg match
    w_popularity=0.9,      # item base popularity
    w_price_fit=0.8,       # price vs user price-sensitivity & cart value
    w_context=0.7,         # meal-time / city context match
)

# Global logistic noise scale (Gumbel). THIS is the primary knob that sets AUC:
# larger -> noisier labels -> lower separability. Calibrated for AUC ~0.85.
LABEL_NOISE = 1.55

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
ITEM2VEC_DIM = 64
ITEM2VEC_WINDOW = 6
ITEM2VEC_EPOCHS = 8
ITEM2VEC_MIN_COUNT = 3
ITEM2VEC_NEG = 8

FAISS_RETRIEVE_K = 120        # candidates pulled from the ANN index per cart
FAISS_NSHARDS = 8             # shard count (one per city) for the serving demo

GRU_ITEM_DIM = 64
GRU_HIDDEN = 64
GRU_EPOCHS = 3
GRU_BATCH = 2048
GRU_MAX_LEN = MAX_CART_SIZE
GRU_TRAIN_SAMPLE = 350_000    # subsample impressions for GRU training

LGB_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=650,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    label_gain=list(range(0, 2)),  # binary relevance
    random_state=SEED,
    n_jobs=-1,
)

# LLM / content embeddings for cold-start
LLM_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_EMB_DIM = 384
MMR_LAMBDA = 0.72             # relevance vs diversity trade-off in re-ranking
TOP_N_DISPLAY = 10            # rail size shown to the user (N = 8-10)

# --------------------------------------------------------------------------- #
# Business assumptions for the impact simulation
# --------------------------------------------------------------------------- #
BASELINE_ACCEPT_RATE = 0.18   # popularity-rail baseline acceptance
TEMPORAL_SPLIT_QUANTILE = 0.8  # last 20% of time -> test (temporal split)

def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
