"""
Event-level synthetic data generator for the CSAO Rail Recommendation System.
======================================================================

We have no access to real Zomato logs, so we simulate a realistic food-delivery
world and, crucially, the *event stream* of cart building rather than only final
carts. Every session emits an ordered sequence of events:

    add     - user organically adds an item (search / browse)
    accept  - user accepts a CSAO rail recommendation (item enters the cart)
    reject  - a recommended item was shown but not added
    remove  - user removes an item already in the cart

This event granularity is what lets a GRU consume the cart as a *sequence* and
lets us later train an accept/reject ranker.

Realism baked in
----------------
* 50,000 users, 15,000 items, ~1.2M interaction events.
* 8 cities, each with its own cuisine mix, AOV level and peak intensity.
* 3x order-volume spikes at lunch (12-2pm) and dinner (7-9pm) vs off-peak.
* 30% cold-start users (<3 historical orders, sparse feature vectors),
  STRATIFIED within every city (not clustered in a few cities).
* Free-text menu descriptions on every item (for later LLM embeddings).
* Latent item taste vectors so co-occurrence is genuinely learnable (Item2Vec).

Outputs (parquet) -> data/
    users.parquet          one row per user (+ sparse cold-start features)
    items.parquet          one row per item (+ restaurant metadata + description)
    interactions.parquet   one row per event  (~1.2M)
    sessions.parquet       one row per session with ordered event sequences

Also writes docs/data_dictionary.md.

Run:  python -m src.data.generate_data
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project importable whether run as module or script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

RNG = np.random.default_rng(C.SEED)
DATA_DIR = C.DATA_DIR
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Scale knobs
# --------------------------------------------------------------------------- #
N_USERS = C.N_USERS                 # 50,000
N_ITEMS = C.N_ITEMS                 # 15,000
N_RESTAURANTS = 800                 # ~19 items/menu -> varied, multi-category carts
COLD_FRAC = C.COLD_START_FRAC       # 0.30
TARGET_EVENTS = 1_200_000
SIM_DAYS = 90
BASE_DATE = pd.Timestamp("2025-01-01")

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
CUISINES = [
    "North Indian", "South Indian", "Chinese", "Biryani", "Mughlai",
    "Fast Food", "Pizza", "Street Food", "Desserts", "Beverages",
]
CI = {c: i for i, c in enumerate(CUISINES)}
CATEGORIES = ["main", "rice", "bread", "starter", "side", "dessert", "beverage"]
CAT = {c: i for i, c in enumerate(CATEGORIES)}
MEAL_PERIODS = ["breakfast", "lunch", "snacks", "dinner", "late_night"]

# Meal windows used for the 3x peak spikes.
LUNCH_HOURS = (12, 13)      # 12-2pm
DINNER_HOURS = (19, 20)     # 7-9pm

# --------------------------------------------------------------------------- #
# City profiles: cuisine mix (bump over uniform), AOV multiplier, peak intensity
# --------------------------------------------------------------------------- #
CITY_PROFILES = {
    "Mumbai":    dict(aov=1.20, peak=3.6, bumps={"Street Food": 1.4, "Pizza": 0.6, "Chinese": 0.5}),
    "Delhi":     dict(aov=1.15, peak=3.3, bumps={"North Indian": 1.6, "Mughlai": 1.0, "Street Food": 0.6}),
    "Bangalore": dict(aov=1.28, peak=3.7, bumps={"South Indian": 1.0, "Chinese": 0.9, "Fast Food": 0.8}),
    "Hyderabad": dict(aov=1.05, peak=3.1, bumps={"Biryani": 1.9, "Mughlai": 0.8}),
    "Kolkata":   dict(aov=0.92, peak=2.6, bumps={"Desserts": 1.3, "Mughlai": 1.0, "Chinese": 0.6}),
    "Chennai":   dict(aov=0.96, peak=2.8, bumps={"South Indian": 1.9, "Beverages": 0.5}),
    "Pune":      dict(aov=1.06, peak=3.0, bumps={"Fast Food": 0.9, "North Indian": 0.7, "Pizza": 0.6}),
    "Ahmedabad": dict(aov=0.90, peak=2.5, bumps={"Street Food": 1.2, "Desserts": 0.8, "Beverages": 0.4}),
}
CITIES = list(CITY_PROFILES.keys())
CITY_IDX = {c: i for i, c in enumerate(CITIES)}


def city_cuisine_weights(city: str) -> np.ndarray:
    w = np.ones(len(CUISINES), dtype=np.float64)
    for cui, b in CITY_PROFILES[city]["bumps"].items():
        w[CI[cui]] += b
    return w / w.sum()


def city_hour_distribution(city: str) -> np.ndarray:
    """Hour-of-day pmf with ~3x lunch & dinner spikes, scaled by city peak."""
    peak = CITY_PROFILES[city]["peak"]
    w = np.full(24, 1.0)
    w[0:6] = 0.15          # deep night
    w[6:11] = 0.55         # breakfast ramp
    w[15:18] = 0.8         # snacks
    w[23] = 0.4
    for h in LUNCH_HOURS:
        w[h] = peak
    for h in DINNER_HOURS:
        w[h] = peak
    # shoulders of the peaks
    w[11] = max(w[11], peak * 0.55)
    w[14] = max(w[14], peak * 0.5)
    w[18] = max(w[18], peak * 0.55)
    w[21] = max(w[21], peak * 0.55)
    return w / w.sum()


def hour_to_meal(h: int) -> int:
    if 6 <= h <= 10:
        return 0   # breakfast
    if 11 <= h <= 14:
        return 1   # lunch
    if 15 <= h <= 18:
        return 2   # snacks
    if 19 <= h <= 22:
        return 3   # dinner
    return 4       # late_night


# --------------------------------------------------------------------------- #
# Category priors per cuisine + complementarity matrix
# --------------------------------------------------------------------------- #
CAT_PRIOR = np.array([
    # main rice bread start side dess  bev
    [0.34, 0.10, 0.18, 0.14, 0.14, 0.05, 0.05],  # North Indian
    [0.40, 0.14, 0.02, 0.16, 0.16, 0.04, 0.08],  # South Indian
    [0.42, 0.16, 0.00, 0.24, 0.14, 0.00, 0.04],  # Chinese
    [0.06, 0.62, 0.00, 0.18, 0.12, 0.02, 0.00],  # Biryani
    [0.44, 0.04, 0.14, 0.30, 0.06, 0.00, 0.02],  # Mughlai
    [0.46, 0.02, 0.02, 0.34, 0.12, 0.02, 0.02],  # Fast Food
    [0.50, 0.00, 0.02, 0.30, 0.16, 0.02, 0.00],  # Pizza
    [0.40, 0.04, 0.06, 0.30, 0.16, 0.02, 0.02],  # Street Food
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],  # Desserts
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # Beverages
], dtype=np.float64)


def complement_matrix() -> np.ndarray:
    """M[a, b] = affinity of adding category b to a cart containing category a."""
    M = np.full((len(CATEGORIES), len(CATEGORIES)), 0.15, dtype=np.float32)

    def s(a, b, v):
        M[CAT[a], CAT[b]] = v
    for anchor in ("main", "rice"):
        s(anchor, "beverage", 0.95); s(anchor, "bread", 0.85); s(anchor, "side", 0.80)
        s(anchor, "dessert", 0.70); s(anchor, "starter", 0.45)
    s("rice", "side", 0.95)
    s("bread", "main", 0.90); s("bread", "side", 0.55)
    s("starter", "main", 0.75); s("starter", "beverage", 0.70)
    s("side", "beverage", 0.55); s("side", "dessert", 0.50)
    s("dessert", "beverage", 0.60); s("beverage", "dessert", 0.45)
    return M


COMPLEMENT = complement_matrix()

# --------------------------------------------------------------------------- #
# Item names + free-text menu descriptions (for LLM embeddings)
# --------------------------------------------------------------------------- #
NAME_BANK = {
    "North Indian": {"main": ["Paneer Butter Masala", "Dal Makhani", "Kadai Paneer", "Chole", "Shahi Paneer", "Rajma Masala"],
                     "bread": ["Butter Naan", "Garlic Naan", "Tandoori Roti", "Lachha Paratha", "Missi Roti"],
                     "side": ["Jeera Rice", "Boondi Raita", "Green Salad", "Masala Papad"],
                     "starter": ["Paneer Tikka", "Hara Bhara Kabab", "Aloo Tikki"],
                     "rice": ["Veg Pulao", "Jeera Rice"]},
    "South Indian": {"main": ["Masala Dosa", "Idli Sambar", "Uttapam", "Rava Dosa", "Ven Pongal"],
                     "side": ["Coconut Chutney", "Medu Vada", "Extra Sambar"],
                     "starter": ["Mysore Bonda", "Rava Idli"],
                     "beverage": ["Filter Coffee", "Masala Buttermilk"],
                     "rice": ["Lemon Rice", "Curd Rice"]},
    "Chinese": {"main": ["Veg Hakka Noodles", "Chilli Paneer", "Schezwan Fried Rice", "Veg Manchurian", "Kung Pao Chicken"],
                "starter": ["Veg Spring Roll", "Chilli Chicken", "Crispy Corn"],
                "side": ["Steamed Momos", "Honey Chilli Potato"],
                "rice": ["Veg Fried Rice", "Schezwan Rice"]},
    "Biryani": {"rice": ["Chicken Dum Biryani", "Veg Dum Biryani", "Hyderabadi Biryani", "Mutton Biryani", "Egg Biryani"],
                "side": ["Mirchi Ka Salan", "Raita", "Boiled Egg"],
                "starter": ["Chicken 65", "Chicken Lollipop"]},
    "Mughlai": {"main": ["Butter Chicken", "Chicken Korma", "Mutton Rogan Josh", "Chicken Changezi"],
                "bread": ["Rumali Roti", "Sheermal"],
                "starter": ["Seekh Kebab", "Chicken Tikka", "Galouti Kebab"]},
    "Fast Food": {"main": ["Veg Burger", "Crispy Chicken Burger", "Grilled Sandwich", "Loaded Fries Bowl"],
                  "starter": ["Salted Fries", "Peri Peri Fries", "Chicken Nuggets"],
                  "side": ["Cheese Dip", "Coleslaw"]},
    "Pizza": {"main": ["Margherita Pizza", "Farmhouse Pizza", "Peppy Paneer Pizza", "Chicken Supreme Pizza"],
              "starter": ["Garlic Bread", "Cheesy Dip Sticks", "Stuffed Garlic Bread"],
              "side": ["Choco Lava Cake", "Potato Wedges"]},
    "Street Food": {"main": ["Pav Bhaji", "Vada Pav", "Chole Bhature", "Ragda Pattice"],
                    "starter": ["Samosa", "Kachori", "Dahi Puri"],
                    "side": ["Sev Puri", "Bhel Puri"]},
    "Desserts": {"dessert": ["Gulab Jamun", "Rasmalai", "Choco Lava Cake", "Gajar Halwa", "Vanilla Ice Cream", "Walnut Brownie"]},
    "Beverages": {"beverage": ["Masala Coke", "Sweet Lassi", "Cold Coffee", "Fresh Lime Soda", "Mango Shake", "Iced Tea", "Coke", "Sprite"]},
}
DESC_ADJ = ["authentic", "freshly prepared", "chef-special", "homestyle", "rich and flavourful",
            "slow-cooked", "aromatic", "indulgent", "classic", "bestselling"]
DESC_PAIR = {
    "main": "pairs perfectly with breads, rice or a chilled beverage",
    "rice": "a complete meal on its own, great with raita and a soft drink",
    "bread": "best enjoyed hot with rich curries and gravies",
    "starter": "an ideal starter to share before the main course",
    "side": "the perfect accompaniment to complete your meal",
    "dessert": "the sweet ending your meal deserves",
    "beverage": "a refreshing drink to go with any dish",
}


def make_name(cuisine: str, category: str, rng) -> str:
    bank = NAME_BANK.get(cuisine, {})
    opts = bank.get(category) or [n for v in bank.values() for n in v] or [f"{cuisine} {category}"]
    return str(rng.choice(opts))


def make_description(name: str, cuisine: str, category: str, is_veg: int, rng) -> str:
    adj = rng.choice(DESC_ADJ)
    diet = "vegetarian" if is_veg else "non-vegetarian"
    pair = DESC_PAIR[category]
    return f"{name}: {adj} {cuisine} {diet} {category} dish, {pair}."


# --------------------------------------------------------------------------- #
# Item + restaurant generation
# --------------------------------------------------------------------------- #
def gen_restaurants(rng):
    city = rng.integers(0, len(CITIES), size=N_RESTAURANTS)
    # restaurants have a FOOD primary cuisine (exclude Desserts/Beverages, which
    # are distributed as add-on items across every menu).
    cuisine = rng.integers(0, len(CUISINES) - 2, size=N_RESTAURANTS)
    price_tier = rng.choice([1, 2, 3], size=N_RESTAURANTS, p=[0.45, 0.40, 0.15])
    rating = np.clip(rng.normal(4.0, 0.33, N_RESTAURANTS), 2.8, 5.0).round(1)
    names = [f"{CUISINES[cuisine[i]].split()[0]} {sfx} {i}"
             for i, sfx in enumerate(rng.choice(
                 ["House", "Kitchen", "Hub", "Corner", "Express", "Darbar", "Junction", "Bistro"],
                 size=N_RESTAURANTS))]
    return pd.DataFrame(dict(
        restaurant_id=np.arange(N_RESTAURANTS, dtype=np.int32),
        restaurant_name=names, city=city.astype(np.int16),
        rest_cuisine=cuisine.astype(np.int16), rest_price_tier=price_tier.astype(np.int8),
        rest_rating=rating.astype(np.float32),
    ))


def gen_items(rng, rest_df):
    cui_w = np.array([1.4, 1.1, 1.2, 1.0, 0.9, 1.1, 0.9, 0.8, 0.9, 1.0])
    cuisine = rng.choice(len(CUISINES), size=N_ITEMS, p=cui_w / cui_w.sum())
    category = np.array([rng.choice(len(CATEGORIES), p=CAT_PRIOR[c]) for c in cuisine], dtype=np.int8)

    cat_price_mu = np.array([5.3, 5.2, 4.7, 4.9, 4.8, 4.4, 4.2], dtype=np.float32)  # log INR by category
    price = np.exp(rng.normal(cat_price_mu[category], 0.42)).astype(np.float32)
    price = np.clip(price, 25, 900).round(0)
    price_tier = np.select([price <= 120, price <= 300], [1, 2], default=3).astype(np.int8)

    base_veg = np.array([0.62, 0.75, 0.60, 0.30, 0.20, 0.45, 0.70, 0.75, 0.95, 0.98])[cuisine]
    is_veg = (rng.random(N_ITEMS) < base_veg).astype(np.int8)

    pop = rng.pareto(1.6, size=N_ITEMS).astype(np.float32) + 0.05
    pop = pop / pop.mean()

    # latent taste vector clustered by cuisine -> Item2Vec-recoverable structure
    zdim = 8
    zc = rng.normal(0, 1, size=(len(CUISINES), zdim)).astype(np.float32)
    z = zc[cuisine] + rng.normal(0, 0.55, size=(N_ITEMS, zdim)).astype(np.float32)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    # assign each item to a restaurant of the same cuisine (fallback: any)
    rest_cui = rest_df["rest_cuisine"].to_numpy()
    rest_ids = rest_df["restaurant_id"].to_numpy()
    rest_by_cui = {c: rest_ids[rest_cui == c] for c in range(len(CUISINES))}
    bev, des = CI["Beverages"], CI["Desserts"]
    item_rest = np.empty(N_ITEMS, dtype=np.int32)
    for i in range(N_ITEMS):
        c = cuisine[i]
        pool = rest_by_cui.get(c)
        if c in (bev, des) or pool is None or len(pool) == 0:
            item_rest[i] = rest_ids[rng.integers(0, len(rest_ids))]
        else:
            item_rest[i] = pool[rng.integers(0, len(pool))]

    names, descs = [], []
    for i in range(N_ITEMS):
        nm = make_name(CUISINES[cuisine[i]], CATEGORIES[category[i]], rng)
        names.append(nm)
        descs.append(make_description(nm, CUISINES[cuisine[i]], CATEGORIES[category[i]], int(is_veg[i]), rng))

    rest_lookup = rest_df.set_index("restaurant_id")
    df = pd.DataFrame(dict(
        item_id=np.arange(N_ITEMS, dtype=np.int32),
        name=names,
        cuisine=[CUISINES[c] for c in cuisine],
        category=[CATEGORIES[c] for c in category],
        price=price,
        price_tier=price_tier,
        is_veg=is_veg,
        popularity=pop.round(4),
        restaurant_id=item_rest,
        description=descs,
    ))
    df["restaurant_name"] = rest_lookup.loc[df["restaurant_id"], "restaurant_name"].to_numpy()
    df["restaurant_city"] = [CITIES[c] for c in rest_lookup.loc[df["restaurant_id"], "city"].to_numpy()]
    # keep integer codes for the simulator
    meta = dict(cuisine=cuisine.astype(np.int16), category=category, price=price.astype(np.float32),
                is_veg=is_veg, pop=pop, z=z, rest=item_rest,
                rest_city=rest_lookup.loc[df["restaurant_id"], "city"].to_numpy().astype(np.int16))
    return df, meta


# --------------------------------------------------------------------------- #
# User generation (cold-start stratified per city)
# --------------------------------------------------------------------------- #
def gen_users(rng):
    # assign cities weighted by "market size"
    city_size = np.array([1.5, 1.5, 1.4, 1.0, 1.0, 0.9, 0.8, 0.7])
    city = rng.choice(len(CITIES), size=N_USERS, p=city_size / city_size.sum()).astype(np.int16)

    segment = rng.choice(4, size=N_USERS, p=[0.30, 0.40, 0.15, 0.15]).astype(np.int8)
    seg_names = np.array(["budget", "regular", "premium", "frequent"])

    # cuisine affinity = city taste + personal noise (softmax)
    city_pref = np.stack([np.log(city_cuisine_weights(CITIES[ci]) + 1e-6) for ci in range(len(CITIES))])
    personal = rng.normal(0, 1.1, size=(N_USERS, len(CUISINES))).astype(np.float32)
    logits = city_pref[city] + personal
    aff = np.exp(logits - logits.max(1, keepdims=True))
    aff /= aff.sum(1, keepdims=True)

    veg_pref = np.clip(rng.beta(2.2, 2.0, N_USERS), 0.02, 0.98).astype(np.float32)
    seg_ps = np.array([0.85, 0.55, 0.20, 0.45], dtype=np.float32)[segment]
    price_sens = np.clip(seg_ps + rng.normal(0, 0.10, N_USERS), 0.02, 0.98).astype(np.float32)

    # --- cold-start: stratified 30% WITHIN each city ---
    is_cold = np.zeros(N_USERS, dtype=np.int8)
    for ci in range(len(CITIES)):
        idx = np.where(city == ci)[0]
        k = int(round(COLD_FRAC * len(idx)))
        chosen = rng.choice(idx, size=k, replace=False)
        is_cold[chosen] = 1

    # order counts: warm >=3, cold in {0,1,2}
    seg_activity = np.array([3.2, 4.2, 5.0, 7.5], dtype=np.float32)[segment]
    order_count = (rng.poisson(seg_activity) + 3).astype(np.int32)
    order_count = np.clip(order_count, 3, 45)
    cold_mask = is_cold == 1
    order_count[cold_mask] = rng.integers(0, 3, size=int(cold_mask.sum()))

    tenure = np.where(cold_mask, rng.integers(1, 45, N_USERS), rng.integers(30, 720, N_USERS)).astype(np.int32)

    df = pd.DataFrame(dict(
        user_id=np.arange(N_USERS, dtype=np.int32),
        city=[CITIES[c] for c in city],
        segment=seg_names[segment],
        is_cold_start=is_cold,
        order_count=order_count,
        tenure_days=tenure,
        veg_pref=veg_pref.round(3),
        price_sensitivity=price_sens.round(3),
    ))
    # historical AOV: NaN (unknown) for cold users with 0 orders -> sparse features
    base_aov = np.array([260, 360, 620, 480], dtype=np.float32)[segment]
    city_aov_mult = np.array([CITY_PROFILES[CITIES[c]]["aov"] for c in city], dtype=np.float32)
    hist_aov = (base_aov * city_aov_mult * (1 + rng.normal(0, 0.12, N_USERS))).round(0)
    hist_aov[order_count == 0] = np.nan          # truly unknown
    df["hist_avg_order_value"] = hist_aov

    # favourite cuisine known only for warm users (sparse for cold)
    fav = np.array(CUISINES)[aff.argmax(1)]
    fav_obj = fav.astype(object)
    fav_obj[cold_mask] = None                    # sparse feature vector for cold users
    df["fav_cuisine"] = fav_obj

    # store per-cuisine affinity columns; blank them out for cold users (sparsity)
    aff_cold = aff.copy()
    aff_cold[cold_mask] = np.nan
    for j, cu in enumerate(CUISINES):
        df[f"aff_{cu.replace(' ', '_').lower()}"] = aff_cold[:, j].round(4)

    meta = dict(city=city, aff=aff, veg_pref=veg_pref, price_sens=price_sens,
                order_count=order_count, is_cold=is_cold, segment=segment)
    return df, meta


# --------------------------------------------------------------------------- #
# Session / event simulation
# --------------------------------------------------------------------------- #
def utility(cand, cart, cart_cats, uaff, uveg, ups, meta, city_cui, meal,
            item_z_cart_mean, city_aov=1.0):
    """Latent acceptance utility of each candidate given the current cart."""
    cc = meta["category"][cand]
    # complementarity vs cart categories + meal-completion bonus
    comp = COMPLEMENT[cart_cats][:, cc].mean(0)
    bev, des = CAT["beverage"], CAT["dessert"]
    if bev not in cart_cats:
        comp = comp + 0.55 * (cc == bev)
    if des not in cart_cats:
        comp = comp + 0.3 * (cc == des)
    cooc = meta["z"][cand] @ item_z_cart_mean
    aff = (uaff[meta["cuisine"][cand]] - 0.1) * 3.0
    veg = np.where(meta["is_veg"][cand] == 1, uveg - 0.5, (0.5 - uveg) - 0.4 * (uveg > 0.8))
    aff = aff + 1.2 * veg
    pop = meta["pop_z"][cand]
    cart_price = meta["price"][cart].mean()
    relp = (np.log(meta["price"][cand]) - np.log(cart_price)) / 0.6
    price_fit = -ups * np.clip(relp, -2, 4)
    ctx = 0.7 * city_cui[meta["cuisine"][cand]] - 0.25
    # city AOV effect: high-AOV cities lean towards pricier add-ons
    aov_pull = (city_aov - 1.0) * meta["price_z"][cand]
    U = (C.UTILITY["bias"]
         + C.UTILITY["w_complement"] * (comp - 0.4)
         + C.UTILITY["w_cooccur"] * cooc
         + C.UTILITY["w_user_affinity"] * aff
         + C.UTILITY["w_popularity"] * pop
         + C.UTILITY["w_price_fit"] * price_fit
         + C.UTILITY["w_context"] * ctx
         + 2.2 * aov_pull)
    return U.astype(np.float32)


def simulate(users_meta, items_meta, rest_df, city_cui_mat, rng):
    meta = items_meta
    lp = np.log(meta["price"])
    meta["pop_z"] = ((np.log1p(meta["pop"]) - np.log1p(meta["pop"]).mean()) /
                     (np.log1p(meta["pop"]).std() + 1e-8)).astype(np.float32)
    meta["price_z"] = ((lp - lp.mean()) / (lp.std() + 1e-8)).astype(np.float32)

    # menus grouped by restaurant
    menus = {}
    order = np.argsort(meta["rest"])
    rs = meta["rest"][order]
    bounds = np.searchsorted(rs, np.arange(rs.max() + 2))
    for r in range(len(bounds) - 1):
        s, e = bounds[r], bounds[r + 1]
        if e > s:
            menus[r] = order[s:e]

    rest_city = rest_df["city"].to_numpy()
    rest_ids = rest_df["restaurant_id"].to_numpy()
    rest_rating = rest_df["rest_rating"].to_numpy()
    rest_by_city = {ci: rest_ids[rest_city == ci] for ci in range(len(CITIES))}

    anchor_cats = np.array([CAT["main"], CAT["rice"]])
    u_city = users_meta["city"]; u_aff = users_meta["aff"]
    u_veg = users_meta["veg_pref"]; u_ps = users_meta["price_sens"]
    u_orders = users_meta["order_count"]

    # precompute per-city hour pmfs
    hour_pmf = {ci: city_hour_distribution(CITIES[ci]) for ci in range(len(CITIES))}

    inter_cols = []   # event rows
    sess_rows = []
    event_id = 0
    session_id = 0
    rec_counter = 0
    t0 = time.time()

    for uid in range(N_USERS):
        k = int(u_orders[uid])
        if k == 0:
            continue
        ci = int(u_city[uid])
        pool_rest = rest_by_city[ci]
        if len(pool_rest) == 0:
            continue
        rc = rest_df["rest_cuisine"].to_numpy()[pool_rest]
        wsel = u_aff[uid][rc] * rest_rating[pool_rest]
        wsel = wsel / wsel.sum()
        city_cui = city_cui_mat[ci]
        city_aov = CITY_PROFILES[CITIES[ci]]["aov"]

        for _ in range(k):
            rid = int(rng.choice(pool_rest, p=wsel))
            menu = menus.get(rid)
            if menu is None or len(menu) < 4:
                continue

            hour = int(rng.choice(24, p=hour_pmf[ci]))
            meal = hour_to_meal(hour)
            day = int(rng.integers(0, SIM_DAYS))
            start = BASE_DATE + pd.Timedelta(days=day, hours=hour,
                                             minutes=int(rng.integers(0, 60)),
                                             seconds=int(rng.integers(0, 60)))
            start_epoch = int(start.value // 10**9)
            is_peak = int(hour in (LUNCH_HOURS + DINNER_HOURS))

            # --- seed: 1-2 anchor items added organically ---
            mcat = meta["category"][menu]
            amask = np.isin(mcat, anchor_cats)
            seed_pool = menu[amask] if amask.any() else menu
            sw = meta["pop"][seed_pool] * (1 + u_aff[uid][meta["cuisine"][seed_pool]])
            n_seed = 1 + int(rng.random() < 0.45)
            seeds = rng.choice(seed_pool, size=min(n_seed, len(seed_pool)),
                               replace=False, p=sw / sw.sum())
            cart = list(dict.fromkeys(int(s) for s in seeds))

            t = start_epoch
            step = 0
            seq_items, seq_ts, seq_recid = [], [], []
            for it in cart:
                inter_cols.append((event_id, session_id, uid, it, rid, ci, t, step,
                                   "add", -1, -1, -1, len(seq_items), meal, is_peak))
                seq_items.append(it); seq_ts.append(t); seq_recid.append(-1)
                event_id += 1; step += 1; t += int(rng.integers(20, 90))

            # --- recommendation rounds (accept / reject events) ---
            n_rounds = int(rng.choice([1, 2, 3], p=[0.42, 0.40, 0.18]))
            max_cart = min(len(menu), 2 + int(rng.integers(2, C.MAX_CART_SIZE))
                           + int(round((city_aov - 1.0) * 3)))
            for _r in range(n_rounds):
                if len(cart) >= max_cart:
                    break
                cand_pool = np.setdiff1d(menu, np.array(cart), assume_unique=False)
                if len(cand_pool) == 0:
                    break
                cart_cats = meta["category"][np.array(cart)]
                z_mean = meta["z"][np.array(cart)].mean(0)
                U = utility(cand_pool, np.array(cart), cart_cats, u_aff[uid],
                            u_veg[uid], u_ps[uid], meta, city_cui, meal, z_mean,
                            city_aov=city_aov)
                # rail = top complementary candidates + a couple popular fillers
                topn = np.argsort(-U)[:8]
                pop_fill = np.argsort(-meta["pop"][cand_pool])[:4]
                rail_local = list(dict.fromkeys(list(topn[:4]) + list(pop_fill) + list(topn[4:8])))
                rail_size = min(len(rail_local), int(rng.choice([3, 4, 5], p=[0.35, 0.4, 0.25])))
                rail_local = rail_local[:rail_size]
                rail_items = cand_pool[rail_local]
                rail_U = U[rail_local]
                rail_p = 1.0 / (1.0 + np.exp(-rail_U / C.LABEL_NOISE))
                accepts = rng.random(len(rail_items)) < rail_p

                for j, loc in enumerate(rail_local):
                    it = int(rail_items[j])
                    rid_rec = rec_counter; rec_counter += 1
                    if accepts[j] and len(cart) < max_cart:
                        inter_cols.append((event_id, session_id, uid, it, rid, ci, t, step,
                                           "accept", rid_rec, j + 1, 1, len(cart), meal, is_peak))
                        cart.append(it)
                        seq_items.append(it); seq_ts.append(t); seq_recid.append(rid_rec)
                    else:
                        inter_cols.append((event_id, session_id, uid, it, rid, ci, t, step,
                                           "reject", rid_rec, j + 1, 0, len(cart), meal, is_peak))
                    event_id += 1; step += 1; t += int(rng.integers(5, 40))

                # occasional removal of a non-seed item
                if len(cart) > len(seeds) and rng.random() < 0.06:
                    rem = int(rng.choice(cart[len(seeds):]))
                    cart.remove(rem)
                    inter_cols.append((event_id, session_id, uid, rem, rid, ci, t, step,
                                       "remove", -1, -1, -1, len(cart), meal, is_peak))
                    event_id += 1; step += 1; t += int(rng.integers(5, 40))

            cart_value = float(meta["price"][np.array(cart)].sum()) if cart else 0.0
            sess_rows.append((session_id, uid, rid, ci, start_epoch, hour, meal, is_peak,
                              step, len(cart), round(cart_value, 1),
                              seq_items, seq_ts, seq_recid))
            session_id += 1

        if uid % 5000 == 0:
            print(f"  users {uid}/{N_USERS}  sessions={session_id}  events={event_id}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    interactions = pd.DataFrame(inter_cols, columns=[
        "event_id", "session_id", "user_id", "item_id", "restaurant_id", "city_id",
        "timestamp", "event_seq", "event_type", "recommendation_id", "rec_rank",
        "label", "cart_size_before", "meal_period_id", "is_peak"])
    interactions["city"] = np.array(CITIES)[interactions["city_id"].to_numpy()]
    interactions["meal_period"] = np.array(MEAL_PERIODS)[interactions["meal_period_id"].to_numpy()]
    # genuine nulls for organic add/remove
    for col in ("recommendation_id", "rec_rank", "label"):
        interactions[col] = interactions[col].replace(-1, pd.NA).astype("Int64")

    sessions = pd.DataFrame(sess_rows, columns=[
        "session_id", "user_id", "restaurant_id", "city_id", "start_timestamp",
        "hour", "meal_period_id", "is_peak", "n_events", "final_cart_size",
        "final_cart_value", "seq_item_ids", "seq_timestamps", "seq_accepted_rec_id"])
    sessions["city"] = np.array(CITIES)[sessions["city_id"].to_numpy()]
    sessions["meal_period"] = np.array(MEAL_PERIODS)[sessions["meal_period_id"].to_numpy()]
    return interactions, sessions


# --------------------------------------------------------------------------- #
def write_data_dictionary(stats):
    md = f"""# Data Dictionary — CSAO Rail Recommendation System (synthetic)

Generated by `src/data/generate_data.py`. All data is synthetic. Cities: {', '.join(CITIES)}.
Cuisines: {', '.join(CUISINES)}. Meal-component categories: {', '.join(CATEGORIES)}.
Meal periods: {', '.join(MEAL_PERIODS)}.

## Dataset summary
| Metric | Value |
|---|---|
| Users | {stats['n_users']:,} |
| Cold-start users (<3 orders) | {stats['n_cold']:,} ({stats['cold_frac']:.1%}) |
| Items | {stats['n_items']:,} |
| Restaurants | {stats['n_restaurants']:,} |
| Sessions | {stats['n_sessions']:,} |
| Interaction events | {stats['n_events']:,} |
| Accept events | {stats['n_accept']:,} |
| Reject events | {stats['n_reject']:,} |
| Add events (organic) | {stats['n_add']:,} |
| Remove events | {stats['n_remove']:,} |
| Overall accept rate (accept/(accept+reject)) | {stats['accept_rate']:.3f} |
| Peak-hour event share (lunch+dinner) | {stats['peak_share']:.3f} |

## `users.parquet` — one row per user
| Column | Type | Description |
|---|---|---|
| user_id | int32 | Unique user id. |
| city | str | Home city (one of 8). |
| segment | str | budget / regular / premium / frequent. |
| is_cold_start | int8 | 1 if user has <3 historical orders. **Stratified ~30% within each city.** |
| order_count | int32 | Historical order count (cold users: 0–2). |
| tenure_days | int32 | Days since signup. |
| veg_pref | float | P(user prefers veg), 0–1. |
| price_sensitivity | float | 0 (insensitive) – 1 (very price sensitive). |
| hist_avg_order_value | float | Historical AOV in INR. **NaN for users with 0 orders (sparse).** |
| fav_cuisine | str | Argmax cuisine affinity. **None for cold users (sparse feature).** |
| aff_<cuisine> (×10) | float | Per-cuisine affinity (sums to 1 for warm users). **NaN for cold users (sparse vector).** |

## `items.parquet` — one row per item (+ restaurant metadata)
| Column | Type | Description |
|---|---|---|
| item_id | int32 | Unique item id. |
| name | str | Dish name. |
| cuisine | str | Cuisine. |
| category | str | Meal component (main/rice/bread/starter/side/dessert/beverage). |
| price | float | Price in INR. |
| price_tier | int8 | 1 budget (≤120), 2 mid (≤300), 3 premium. |
| is_veg | int8 | 1 vegetarian, 0 non-veg. |
| popularity | float | Relative popularity (mean 1, heavy-tailed). |
| restaurant_id | int32 | Owning restaurant. |
| restaurant_name | str | Restaurant name. |
| restaurant_city | str | Restaurant city. |
| description | str | **Free-text menu description** (for LLM/content embeddings). |

## `interactions.parquet` — one row per event (~{stats['n_events']:,})
| Column | Type | Description |
|---|---|---|
| event_id | int64 | Global unique event id. |
| session_id | int64 | Session the event belongs to. |
| user_id | int32 | User. |
| item_id | int32 | Item acted on. |
| restaurant_id | int32 | Restaurant of the session. |
| city / city_id | str/int | City. |
| timestamp | int64 | Unix epoch seconds. |
| event_seq | int32 | 0-indexed position of the event within the session. |
| event_type | str | `add` (organic) / `accept` / `reject` / `remove`. |
| recommendation_id | Int64 | Unique id of the shown recommendation (accept/reject only; **null** for add/remove). |
| rec_rank | Int64 | Slot of the item in the shown rail 1..N (accept/reject only; null otherwise). |
| label | Int64 | 1 accept, 0 reject; **null** for organic add/remove (the ranker's training target). |
| cart_size_before | int32 | Cart size immediately before the event. |
| meal_period / meal_period_id | str/int | breakfast/lunch/snacks/dinner/late_night. |
| is_peak | int8 | 1 if hour ∈ lunch(12–13) ∪ dinner(19–20). |

## `sessions.parquet` — one row per session (ordered sequences for the GRU)
| Column | Type | Description |
|---|---|---|
| session_id | int64 | Unique session id. |
| user_id | int32 | User. |
| restaurant_id | int32 | Restaurant. |
| city / city_id | str/int | City. |
| start_timestamp | int64 | Session start (epoch seconds). |
| hour | int8 | Start hour of day. |
| meal_period / meal_period_id | str/int | Meal period. |
| is_peak | int8 | Peak-hour flag. |
| n_events | int32 | Total events in the session. |
| final_cart_size | int32 | Items in cart at checkout. |
| final_cart_value | float | Cart value (INR) — session AOV. |
| **seq_item_ids** | list<int> | **Ordered** items as they entered the cart. |
| **seq_timestamps** | list<int> | Epoch seconds aligned to `seq_item_ids`. |
| **seq_accepted_rec_id** | list<int> | Per added item: the `recommendation_id` it came from, or **-1** if organic (null). |

> The `(seq_item_ids, seq_timestamps, seq_accepted_rec_id)` triple is the ordered
> cart-state sequence a GRU consumes. `seq_accepted_rec_id == -1` marks an
> organically-added item; a non-negative value marks a CSAO-accepted item.

## Realism notes
* **City-wise behaviour:** each city has its own cuisine mix, AOV multiplier and
  peak intensity (see `CITY_PROFILES`).
* **Peak spikes:** hour-of-day pmf places ~3× mass on lunch (12–2pm) and dinner
  (7–9pm) vs off-peak; peak share observed = {stats['peak_share']:.1%}.
* **Cold-start:** exactly ~30% of each city's users are flagged cold with sparse
  (NaN) affinity vectors / AOV / fav-cuisine — stratified, not clustered.
* **Learnable signal:** accept/reject labels come from a latent-utility choice
  model (complementarity + co-occurrence + affinity + popularity + price-fit +
  context) with tunable noise, so downstream models recover genuine structure.
"""
    (DOCS_DIR / "data_dictionary.md").write_text(md, encoding="utf-8")


def main():
    rng = np.random.default_rng(C.SEED)
    print("Generating restaurants + items ...", flush=True)
    rest_df = gen_restaurants(rng)
    items_df, items_meta = gen_items(rng, rest_df)

    print("Generating users (city-stratified cold-start) ...", flush=True)
    users_df, users_meta = gen_users(rng)

    # city -> cuisine popularity matrix (avg affinity of that city's users)
    city_cui_mat = np.zeros((len(CITIES), len(CUISINES)), dtype=np.float32)
    for ci in range(len(CITIES)):
        m = users_meta["city"] == ci
        city_cui_mat[ci] = users_meta["aff"][m].mean(0)

    print("Simulating sessions + events ...", flush=True)
    interactions, sessions = simulate(users_meta, items_meta, rest_df, city_cui_mat, rng)

    print("Writing parquet ...", flush=True)
    users_df.to_parquet(DATA_DIR / "users.parquet", index=False)
    items_df.to_parquet(DATA_DIR / "items.parquet", index=False)
    interactions.to_parquet(DATA_DIR / "interactions.parquet", index=False)
    sessions.to_parquet(DATA_DIR / "sessions.parquet", index=False)

    et = interactions["event_type"]
    stats = dict(
        n_users=int(len(users_df)), n_cold=int(users_df["is_cold_start"].sum()),
        cold_frac=float(users_df["is_cold_start"].mean()),
        n_items=int(len(items_df)), n_restaurants=int(len(rest_df)),
        n_sessions=int(len(sessions)), n_events=int(len(interactions)),
        n_accept=int((et == "accept").sum()), n_reject=int((et == "reject").sum()),
        n_add=int((et == "add").sum()), n_remove=int((et == "remove").sum()),
        accept_rate=float((et == "accept").sum() / max(1, (et.isin(["accept", "reject"])).sum())),
        peak_share=float(interactions["is_peak"].mean()),
    )
    # per-city sanity: cold fraction + cuisine share
    city_cold = users_df.groupby("city")["is_cold_start"].mean().round(3).to_dict()
    stats["cold_frac_by_city"] = city_cold
    (C.RESULTS_DIR / "data_gen_stats.json").write_text(json.dumps(stats, indent=2))
    write_data_dictionary(stats)

    print("\n==== DATA GENERATION COMPLETE ====")
    print(json.dumps({k: v for k, v in stats.items() if k != "cold_frac_by_city"}, indent=2))
    print("cold fraction by city:", json.dumps(city_cold))
    return stats


if __name__ == "__main__":
    main()
