"""
Synthetic data generation for the CSAO Rail Recommendation System.

We do NOT have access to Zomato's data, so we build a *generative simulator*
that reproduces the messy structure of real food-delivery behaviour:

  * 50K users across 8 cities, with segment, cuisine affinity, veg preference
    and price sensitivity. 30% are cold-start (0-2 orders).
  * 15K items with cuisine, meal-component category, price, veg flag, a latent
    taste vector (so co-occurrence is learnable by Item2Vec) and a realistic
    generated name (so content/LLM embeddings are meaningful).
  * 2K restaurants, city-localised, each exposing a coherent menu.
  * Orders built *sequentially* (seed dish -> complementary add-ons) so the cart
    is an ordered sequence -> this is what the GRU encodes.
  * A ground-truth choice model assigns each candidate add-on an acceptance
    probability from a latent utility (complementarity + co-occurrence + user
    affinity + popularity + price-fit + context). Labels are Bernoulli draws;
    the LABEL_NOISE temperature controls separability -> calibrates AUC.

Outputs (parquet / npz) in data/:
  users, items, restaurants, orders, impressions, candidates
The ground-truth acceptance prob is stored on candidates as `gt_prob` and is
used ONLY by the offline business simulator (never as a model feature).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as C

RNG = np.random.default_rng(C.SEED)

MEAL_BUCKETS = ["breakfast", "lunch", "snacks", "dinner", "latenight"]

# --------------------------------------------------------------------------- #
# Taxonomy helpers
# --------------------------------------------------------------------------- #
def _hour_meal_bucket(hours: np.ndarray) -> np.ndarray:
    b = np.full(hours.shape, 2, dtype=np.int8)          # default snacks
    b[(hours >= 6) & (hours <= 10)] = 0                 # breakfast
    b[(hours >= 11) & (hours <= 15)] = 1                 # lunch
    b[(hours >= 16) & (hours <= 18)] = 2                 # snacks
    b[(hours >= 19) & (hours <= 23)] = 3                 # dinner
    b[(hours <= 5)] = 4                                  # latenight
    return b


# Category complementarity matrix C[a, b] = affinity of adding category b to a
# cart that already contains category a.  Encodes "complete the meal" logic.
def _category_complement_matrix() -> np.ndarray:
    cats = C.CATEGORIES
    idx = {c: i for i, c in enumerate(cats)}
    M = np.full((len(cats), len(cats)), 0.15, dtype=np.float32)
    def s(a, b, v):
        M[idx[a], idx[b]] = v
    for anchor in ("main", "rice"):
        s(anchor, "beverage", 0.95); s(anchor, "bread", 0.85); s(anchor, "side", 0.80)
        s(anchor, "dessert", 0.70); s(anchor, "starter", 0.45)
    s("rice", "side", 0.95)          # biryani -> raita / salan
    s("bread", "main", 0.90); s("bread", "side", 0.55)
    s("starter", "main", 0.75); s("starter", "beverage", 0.70)
    s("side", "beverage", 0.55); s("side", "dessert", 0.50)
    s("dessert", "beverage", 0.60)
    s("beverage", "dessert", 0.45)
    return M


# Meal-time propensity per category (rows=category, cols=meal bucket).
def _category_mealtime() -> np.ndarray:
    cats = C.CATEGORIES
    P = np.ones((len(cats), len(MEAL_BUCKETS)), dtype=np.float32) * 0.5
    m = {c: i for i, c in enumerate(cats)}
    b = {x: i for i, x in enumerate(MEAL_BUCKETS)}
    P[m["main"]]    = [0.4, 1.0, 0.5, 1.0, 0.7]
    P[m["rice"]]    = [0.2, 1.0, 0.4, 1.0, 0.6]
    P[m["bread"]]   = [0.3, 0.9, 0.4, 1.0, 0.5]
    P[m["starter"]] = [0.2, 0.7, 0.9, 0.9, 0.8]
    P[m["side"]]    = [0.3, 0.8, 0.5, 0.8, 0.5]
    P[m["dessert"]] = [0.3, 0.6, 0.8, 0.8, 0.7]
    P[m["beverage"]]= [0.9, 0.9, 1.0, 0.8, 0.9]
    return P


# --------------------------------------------------------------------------- #
# Name generation (so LLM/content embeddings carry real semantics)
# --------------------------------------------------------------------------- #
_NAME_BANK = {
    "North Indian": {
        "main": ["Paneer Butter Masala", "Dal Makhani", "Kadai Paneer", "Chole", "Shahi Paneer", "Rajma"],
        "bread": ["Butter Naan", "Garlic Naan", "Tandoori Roti", "Lachha Paratha", "Missi Roti"],
        "side": ["Jeera Rice", "Raita", "Green Salad", "Papad"],
        "starter": ["Paneer Tikka", "Hara Bhara Kabab", "Aloo Tikki"],
        "rice": ["Veg Pulao", "Jeera Rice"],
    },
    "South Indian": {
        "main": ["Masala Dosa", "Idli Sambar", "Uttapam", "Rava Dosa", "Pongal"],
        "side": ["Coconut Chutney", "Medu Vada", "Sambar Bowl"],
        "starter": ["Mysore Bonda", "Rava Idli"],
        "beverage": ["Filter Coffee", "Buttermilk"],
        "rice": ["Lemon Rice", "Curd Rice"],
    },
    "Chinese": {
        "main": ["Veg Hakka Noodles", "Chilli Paneer", "Schezwan Fried Rice", "Manchurian Gravy", "Kung Pao"],
        "starter": ["Veg Spring Roll", "Chilli Chicken Dry", "Crispy Corn"],
        "side": ["Momos", "Honey Chilli Potato"],
        "rice": ["Veg Fried Rice", "Schezwan Rice"],
    },
    "Biryani": {
        "rice": ["Chicken Biryani", "Veg Dum Biryani", "Hyderabadi Biryani", "Mutton Biryani", "Egg Biryani"],
        "side": ["Mirchi Ka Salan", "Raita", "Boiled Egg"],
        "starter": ["Chicken 65", "Chicken Lollipop"],
    },
    "Mughlai": {
        "main": ["Butter Chicken", "Chicken Korma", "Mutton Rogan Josh", "Chicken Changezi"],
        "bread": ["Rumali Roti", "Sheermal"],
        "starter": ["Seekh Kebab", "Chicken Tikka", "Galouti Kebab"],
    },
    "Fast Food": {
        "main": ["Veg Burger", "Chicken Burger", "Grilled Sandwich", "Loaded Fries Bowl"],
        "starter": ["French Fries", "Peri Peri Fries", "Nuggets"],
        "side": ["Cheese Dip", "Coleslaw"],
    },
    "Pizza": {
        "main": ["Margherita Pizza", "Farmhouse Pizza", "Peppy Paneer Pizza", "Chicken Supreme Pizza"],
        "starter": ["Garlic Bread", "Cheesy Dip Sticks", "Stuffed Garlic Bread"],
        "side": ["Choco Lava Cake", "Potato Wedges"],
    },
    "Street Food": {
        "main": ["Pav Bhaji", "Vada Pav", "Chole Bhature", "Pani Puri Plate"],
        "starter": ["Samosa", "Kachori", "Dahi Puri"],
        "side": ["Sev Puri", "Bhel Puri"],
    },
    "Desserts": {
        "dessert": ["Gulab Jamun", "Rasmalai", "Choco Lava Cake", "Gajar Halwa", "Ice Cream Tub", "Brownie"],
    },
    "Beverages": {
        "beverage": ["Masala Coke", "Sweet Lassi", "Cold Coffee", "Fresh Lime Soda", "Mango Shake", "Iced Tea", "Coke", "Sprite"],
    },
}
_PREFIX = ["", "Special ", "Classic ", "Signature ", "Homestyle ", "Chef's ", "Family ", "Regular "]


def _make_name(cuisine, category, rng):
    bank = _NAME_BANK.get(cuisine, {})
    opts = bank.get(category)
    if not opts:
        # fall back to any category available for that cuisine
        flat = [n for v in bank.values() for n in v]
        base = rng.choice(flat) if flat else f"{cuisine} {category}"
    else:
        base = rng.choice(opts)
    return (rng.choice(_PREFIX) + base).strip()


# --------------------------------------------------------------------------- #
# Entity generation
# --------------------------------------------------------------------------- #
def gen_items(rng):
    n = C.N_ITEMS
    n_cui, n_cat = len(C.CUISINES), len(C.CATEGORIES)

    # Cuisine mix: some cuisines (beverages/desserts) are pure-category.
    cui_weights = np.array([1.4, 1.1, 1.2, 1.0, 0.9, 1.1, 0.9, 0.8, 0.9, 1.0])
    cuisine = rng.choice(n_cui, size=n, p=cui_weights / cui_weights.sum())

    # Category depends on cuisine (Beverages->beverage only, Desserts->dessert).
    cat_prior = np.array([
        # main rice bread starter side dessert beverage
        [0.34,0.10,0.18,0.14,0.14,0.05,0.05],   # North Indian
        [0.40,0.14,0.02,0.16,0.16,0.04,0.08],   # South Indian
        [0.42,0.16,0.00,0.24,0.14,0.00,0.04],   # Chinese
        [0.06,0.62,0.00,0.18,0.12,0.02,0.00],   # Biryani
        [0.44,0.04,0.14,0.30,0.06,0.00,0.02],   # Mughlai
        [0.46,0.02,0.02,0.34,0.12,0.02,0.02],   # Fast Food
        [0.50,0.00,0.02,0.30,0.16,0.02,0.00],   # Pizza
        [0.40,0.04,0.06,0.30,0.16,0.02,0.02],   # Street Food
        [0.00,0.00,0.00,0.00,0.00,1.00,0.00],   # Desserts
        [0.00,0.00,0.00,0.00,0.00,0.00,1.00],   # Beverages
    ], dtype=np.float64)
    category = np.array([rng.choice(n_cat, p=cat_prior[c]) for c in cuisine], dtype=np.int8)

    # Price: log-normal, shifted by category (beverages cheap, mains dearer).
    cat_price_mu = np.array([5.3,5.2,4.7,4.9,4.8,4.4,4.2], dtype=np.float32)  # log INR
    price = np.exp(rng.normal(cat_price_mu[category], 0.45)).astype(np.float32)
    price = np.clip(price, 25, 900)

    # Veg flag: cuisine + category dependent.
    base_veg = np.array([0.62,0.75,0.60,0.30,0.20,0.45,0.70,0.75,0.95,0.98])[cuisine]
    is_veg = (rng.random(n) < base_veg).astype(np.int8)

    # Base popularity: heavy-tailed (Zipf-ish) so a few items dominate.
    pop = rng.pareto(1.6, size=n).astype(np.float32) + 0.05
    pop = pop / pop.mean()

    # Latent taste vector: cluster by cuisine so Item2Vec can recover structure.
    zc = rng.normal(0, 1, size=(n_cui, C.ITEM2VEC_DIM // 8)).astype(np.float32)
    z = zc[cuisine] + rng.normal(0, 0.55, size=(n, C.ITEM2VEC_DIM // 8)).astype(np.float32)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    names = [_make_name(C.CUISINES[cuisine[i]], C.CATEGORIES[category[i]], rng) for i in range(n)]

    df = pd.DataFrame(dict(
        item_id=np.arange(n, dtype=np.int32),
        name=names,
        cuisine=cuisine.astype(np.int8),
        category=category,
        price=price.round(0),
        is_veg=is_veg,
        popularity=pop,
    ))
    return df, z


def gen_restaurants(rng, items_df):
    n = C.N_RESTAURANTS
    city = rng.integers(0, len(C.CITIES), size=n).astype(np.int8)
    # restaurant primary cuisine biased by nothing special; menus assembled later
    cuisine = rng.integers(0, len(C.CUISINES), size=n).astype(np.int8)
    price_tier = rng.choice([1, 2, 3], size=n, p=[0.45, 0.4, 0.15]).astype(np.int8)
    rating = np.clip(rng.normal(4.0, 0.35, size=n), 2.8, 5.0).astype(np.float32)
    df = pd.DataFrame(dict(
        restaurant_id=np.arange(n, dtype=np.int32),
        city=city, cuisine=cuisine, price_tier=price_tier, rating=rating.round(1),
    ))
    return df


def assign_menus(rng, items_df, rest_df):
    """Assign each item to a restaurant; build per-restaurant menus that are
    coherent (mostly the restaurant's cuisine + universal beverages/desserts)."""
    n_items = len(items_df)
    item_cui = items_df["cuisine"].to_numpy()
    bev_id = C.CUISINES.index("Beverages")
    des_id = C.CUISINES.index("Desserts")

    # Group restaurants by cuisine for fast matching.
    rest_by_cui = {c: rest_df.index[rest_df["cuisine"] == c].to_numpy()
                   for c in range(len(C.CUISINES))}
    all_rest = rest_df["restaurant_id"].to_numpy()

    item_restaurant = np.empty(n_items, dtype=np.int32)
    for i in range(n_items):
        c = item_cui[i]
        if c in (bev_id, des_id):
            item_restaurant[i] = all_rest[rng.integers(0, len(all_rest))]
        else:
            pool = rest_by_cui.get(c)
            if pool is None or len(pool) == 0:
                item_restaurant[i] = all_rest[rng.integers(0, len(all_rest))]
            else:
                item_restaurant[i] = pool[rng.integers(0, len(pool))]
    items_df = items_df.copy()
    items_df["restaurant_id"] = item_restaurant

    # Menus: restaurant items + a pooled set of beverages/desserts available to all.
    menus = {}
    for rid, grp in items_df.groupby("restaurant_id"):
        menus[int(rid)] = grp["item_id"].to_numpy()
    return items_df, menus


def gen_users(rng):
    n = C.N_USERS
    n_cui = len(C.CUISINES)
    city = rng.integers(0, len(C.CITIES), size=n).astype(np.int8)
    segment = rng.choice(len(C.USER_SEGMENTS), size=n, p=[0.30, 0.40, 0.15, 0.15]).astype(np.int8)

    # City cuisine preference (rows city, cols cuisine) -> regional taste.
    city_pref = rng.normal(0, 0.8, size=(len(C.CITIES), n_cui)).astype(np.float32)
    # Hyderabad loves Biryani, Kolkata sweets, Chennai South Indian, etc.
    def bump(cty, cui, v):
        city_pref[C.CITIES.index(cty), C.CUISINES.index(cui)] += v
    bump("Hyderabad", "Biryani", 1.6); bump("Chennai", "South Indian", 1.6)
    bump("Kolkata", "Desserts", 1.2); bump("Kolkata", "Mughlai", 0.9)
    bump("Mumbai", "Street Food", 1.3); bump("Delhi", "North Indian", 1.4)
    bump("Bangalore", "Chinese", 0.8); bump("Ahmedabad", "Street Food", 1.0)
    bump("Pune", "Fast Food", 0.8)

    personal = rng.normal(0, 1.1, size=(n, n_cui)).astype(np.float32)
    logits = city_pref[city] + personal
    cuisine_aff = np.exp(logits - logits.max(axis=1, keepdims=True))
    cuisine_aff /= cuisine_aff.sum(axis=1, keepdims=True)

    veg_pref = np.clip(rng.beta(2.2, 2.0, size=n).astype(np.float32), 0.02, 0.98)
    # price sensitivity by segment: budget high, premium low
    seg_ps = np.array([0.85, 0.55, 0.20, 0.45], dtype=np.float32)[segment]
    price_sensitivity = np.clip(seg_ps + rng.normal(0, 0.12, n), 0.02, 0.98).astype(np.float32)

    # Cold-start assignment
    is_cold = np.zeros(n, dtype=np.int8)
    cold_idx = rng.choice(n, size=int(C.COLD_START_FRAC * n), replace=False)
    is_cold[cold_idx] = 1

    # target order counts
    seg_activity = np.array([0.8, 1.0, 1.3, 1.9], dtype=np.float32)[segment]
    n_orders = rng.poisson(C.AVG_ORDERS_PER_ACTIVE_USER * seg_activity).astype(np.int32)
    n_orders[is_cold == 1] = rng.integers(0, C.COLD_MAX_ORDERS + 1, size=int(is_cold.sum()))
    n_orders = np.clip(n_orders, 0, 40)

    df = pd.DataFrame(dict(
        user_id=np.arange(n, dtype=np.int32),
        city=city, segment=segment,
        veg_pref=veg_pref, price_sensitivity=price_sensitivity,
        is_cold_start=is_cold, n_orders=n_orders,
    ))
    return df, cuisine_aff


# --------------------------------------------------------------------------- #
# Ground-truth utility  (vectorised over a candidate set)
# --------------------------------------------------------------------------- #
class ChoiceModel:
    """Latent-utility choice model reused by generation, eval and business sim."""

    def __init__(self, items_df, z, cuisine_aff, city_cuisine):
        self.item_cui = items_df["cuisine"].to_numpy()
        self.item_cat = items_df["category"].to_numpy()
        self.item_price = items_df["price"].to_numpy().astype(np.float32)
        self.item_veg = items_df["is_veg"].to_numpy()
        pop = items_df["popularity"].to_numpy().astype(np.float32)
        self.item_pop_z = ((np.log1p(pop) - np.log1p(pop).mean()) /
                           (np.log1p(pop).std() + 1e-8)).astype(np.float32)
        self.z = z
        self.cuisine_aff = cuisine_aff
        self.city_cuisine = city_cuisine
        self.Ccat = _category_complement_matrix()
        self.cat_meal = _category_mealtime()
        lp = np.log(self.item_price)
        self.price_logmu, self.price_logsd = lp.mean(), lp.std() + 1e-8
        self.bev_cat = C.CATEGORIES.index("beverage")
        self.des_cat = C.CATEGORIES.index("dessert")

    def utility(self, cand_ids, cart_ids, user_id, meal_bucket, city):
        """Utility of each candidate given the current cart & context."""
        cand = np.asarray(cand_ids)
        cart = np.asarray(cart_ids)
        cc = self.item_cat[cand]
        # --- complementarity vs cart categories ---
        if len(cart):
            cart_cats = self.item_cat[cart]
            comp = self.Ccat[cart_cats][:, cc].mean(axis=0)          # (n_cand,)
            has_bev = np.any(self.item_cat[cart] == self.bev_cat)
            has_des = np.any(self.item_cat[cart] == self.des_cat)
            comp = comp + (~has_bev) * (cc == self.bev_cat) * 0.6
            comp = comp + (~has_des) * (cc == self.des_cat) * 0.35
            # --- co-occurrence via latent taste vectors ---
            cart_z = self.z[cart].mean(axis=0)
            cooc = self.z[cand] @ cart_z
            cart_price = self.item_price[cart].mean()
        else:
            comp = np.full(len(cand), 0.3, dtype=np.float32)
            cooc = np.zeros(len(cand), dtype=np.float32)
            cart_price = self.item_price[cand].mean()

        # --- user affinity ---
        ua = self.cuisine_aff[user_id][self.item_cui[cand]]
        ua = (ua - 0.1) * 3.0
        veg_pref = self._veg_pref_cache[user_id]
        veg_match = np.where(self.item_veg[cand] == 1,
                             (veg_pref - 0.5),
                             (0.5 - veg_pref) - 0.4 * (veg_pref > 0.8))
        ua = ua + 1.2 * veg_match

        # --- popularity ---
        pop = self.item_pop_z[cand]

        # --- price fit ---
        ps = self._ps_cache[user_id]
        relprice = (np.log(self.item_price[cand]) - np.log(cart_price)) / 0.6
        price_fit = -ps * np.clip(relprice, -2, 4)

        # --- context: meal-time + city cuisine ---
        ctx = self.cat_meal[cc, meal_bucket] + 0.7 * self.city_cuisine[city][self.item_cui[cand]]
        ctx = (ctx - 0.8)

        U = (C.UTILITY["bias"]
             + C.UTILITY["w_complement"] * (comp - 0.4)
             + C.UTILITY["w_cooccur"] * cooc
             + C.UTILITY["w_user_affinity"] * ua
             + C.UTILITY["w_popularity"] * pop
             + C.UTILITY["w_price_fit"] * price_fit
             + C.UTILITY["w_context"] * ctx)
        return U.astype(np.float32)

    def prob(self, *a, **k):
        return 1.0 / (1.0 + np.exp(-self.utility(*a, **k) / C.LABEL_NOISE))


# --------------------------------------------------------------------------- #
# Order + impression simulation
# --------------------------------------------------------------------------- #
def _sample_hours(n, rng):
    """Hour-of-day distribution with ~3x peak mass at lunch & dinner."""
    base = np.ones(24, dtype=np.float64) * 0.35
    base[6:11] = 0.7
    for h in C.LUNCH_HOURS:
        base[h] = C.PEAK_MULTIPLIER
    for h in C.DINNER_HOURS:
        base[h] = C.PEAK_MULTIPLIER
    base[0:5] = 0.25
    base /= base.sum()
    return rng.choice(24, size=n, p=base).astype(np.int8)


def simulate(items_df, menus, users_df, rest_df, cuisine_aff, z, city_cuisine, rng):
    cm = ChoiceModel(items_df, z, cuisine_aff, city_cuisine)
    cm._veg_pref_cache = users_df["veg_pref"].to_numpy()
    cm._ps_cache = users_df["price_sensitivity"].to_numpy()

    item_cat = items_df["category"].to_numpy()
    item_cui = items_df["cuisine"].to_numpy()
    item_pop = items_df["popularity"].to_numpy().astype(np.float64)
    anchor_cats = {C.CATEGORIES.index(c) for c in ("main", "rice")}

    user_city = users_df["city"].to_numpy()
    n_orders_arr = users_df["n_orders"].to_numpy()
    rest_city = rest_df["city"].to_numpy()
    rest_ids = rest_df["restaurant_id"].to_numpy()
    rest_cui = rest_df["cuisine"].to_numpy()
    rest_by_city = {ci: rest_ids[rest_city == ci] for ci in range(len(C.CITIES))}

    # rolling order timestamp over a 90-day window
    order_rows, order_items_seq = [], []
    imp_rows, cand_rows = [], []
    oid = 0
    imp_id = 0
    target_imps = C.TARGET_INTERACTIONS // C.CANDIDATES_PER_IMPRESSION
    # probability of emitting a training impression at a decision step
    # (tuned after we know rough #steps; start high, will be capped by target)
    day = np.zeros(1)

    t0 = time.time()
    total_users = len(users_df)
    keep_imp_prob = 0.28   # subsample decision steps toward target

    for uid in range(total_users):
        k = n_orders_arr[uid]
        if k == 0:
            continue
        ci = user_city[uid]
        cand_rest = rest_by_city[ci]
        if len(cand_rest) == 0:
            continue
        # restaurant preference by user cuisine affinity
        rc = rest_cui[cand_rest]
        w = cuisine_aff[uid][rc] * rest_df["rating"].to_numpy()[cand_rest]
        w = w / w.sum()

        hours = _sample_hours(k, rng)
        meal = _hour_meal_bucket(hours)
        days = rng.integers(0, 90, size=k)

        for j in range(k):
            rid = int(rng.choice(cand_rest, p=w))
            menu = menus.get(rid)
            if menu is None or len(menu) < 3:
                continue
            mb = int(meal[j])
            # ---- seed anchor item ----
            menu_cat = item_cat[menu]
            anchor_mask = np.isin(menu_cat, list(anchor_cats))
            seed_pool = menu[anchor_mask] if anchor_mask.any() else menu
            seed_w = item_pop[seed_pool] * (1 + cuisine_aff[uid][item_cui[seed_pool]])
            seed = int(rng.choice(seed_pool, p=seed_w / seed_w.sum()))
            cart = [seed]

            target_size = min(len(menu), 1 + rng.integers(1, C.MAX_CART_SIZE))
            # ---- sequential additions ----
            while len(cart) < target_size:
                pool = np.setdiff1d(menu, np.array(cart), assume_unique=False)
                if len(pool) == 0:
                    break
                U = cm.utility(pool, np.array(cart), uid, mb, ci)
                p = 1.0 / (1.0 + np.exp(-U / C.LABEL_NOISE))
                # organic next-item choice ~ softmax of utility
                logits = U - U.max()
                probs = np.exp(logits * 1.3)
                probs /= probs.sum()

                # ---- emit a CSAO training impression at this decision step ----
                if rng.random() < keep_imp_prob and imp_id < target_imps:
                    n_cand = C.CANDIDATES_PER_IMPRESSION
                    # rail = a "retrieved" candidate set: top-utility distractors
                    # + popular + random, always includes the organically chosen
                    top_idx = np.argsort(-U)[:max(6, n_cand)]
                    chosen_local = int(np.argmax(np.cumsum(probs) > rng.random()))
                    picks = set(top_idx[:4].tolist()) | {chosen_local}
                    # popular fillers
                    pop_local = np.argsort(-item_pop[pool])[:6]
                    for pl in pop_local:
                        if len(picks) >= n_cand:
                            break
                        picks.add(int(pl))
                    while len(picks) < n_cand and len(picks) < len(pool):
                        picks.add(int(rng.integers(0, len(pool))))
                    picks = list(picks)[:n_cand]
                    rail_ids = pool[picks]
                    rail_U = U[picks]
                    rail_p = 1.0 / (1.0 + np.exp(-rail_U / C.LABEL_NOISE))
                    labels = (rng.random(len(rail_ids)) < rail_p).astype(np.int8)
                    # guarantee the organically-chosen item is a positive
                    chosen_item = int(pool[chosen_local])
                    for r, it in enumerate(rail_ids):
                        if int(it) == chosen_item:
                            labels[r] = 1
                    if labels.sum() == 0:
                        labels[int(np.argmax(rail_p))] = 1
                    imp_rows.append((imp_id, uid, rid, ci, int(days[j]), int(hours[j]),
                                     mb, len(cart), " ".join(map(str, cart))))
                    for r, it in enumerate(rail_ids):
                        cand_rows.append((imp_id, int(it), int(labels[r]), float(rail_p[r])))
                    imp_id += 1

                # commit organic choice
                nxt = int(pool[int(np.argmax(np.cumsum(probs) > rng.random()))])
                cart.append(nxt)

            order_rows.append((oid, uid, rid, ci, int(days[j]), int(hours[j]), mb, len(cart)))
            order_items_seq.append((oid, " ".join(map(str, cart))))
            oid += 1

        if uid % 5000 == 0:
            el = time.time() - t0
            print(f"  users {uid}/{total_users}  orders={oid}  imps={imp_id}  ({el:.0f}s)", flush=True)
        if imp_id >= target_imps and oid > 120_000:
            # enough impressions AND a healthy Item2Vec corpus
            pass

    orders_df = pd.DataFrame(order_rows, columns=[
        "order_id", "user_id", "restaurant_id", "city", "day", "hour", "meal_bucket", "cart_size"])
    order_items_df = pd.DataFrame(order_items_seq, columns=["order_id", "items"])
    imp_df = pd.DataFrame(imp_rows, columns=[
        "impression_id", "user_id", "restaurant_id", "city", "day", "hour",
        "meal_bucket", "step", "cart_items"])
    cand_df = pd.DataFrame(cand_rows, columns=["impression_id", "item_id", "label", "gt_prob"])
    return orders_df, order_items_df, imp_df, cand_df


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(C.SEED)
    print("Generating items / restaurants / users ...", flush=True)
    items_df, z = gen_items(rng)
    rest_df = gen_restaurants(rng, items_df)
    items_df, menus = assign_menus(rng, items_df, rest_df)
    users_df, cuisine_aff = gen_users(rng)

    # city -> cuisine popularity (softmax of avg user affinity in that city)
    n_cui = len(C.CUISINES)
    city_cuisine = np.zeros((len(C.CITIES), n_cui), dtype=np.float32)
    for ci in range(len(C.CITIES)):
        m = users_df["city"].to_numpy() == ci
        city_cuisine[ci] = cuisine_aff[m].mean(axis=0)

    print("Simulating orders + CSAO impressions ...", flush=True)
    orders_df, order_items_df, imp_df, cand_df = simulate(
        items_df, menus, users_df, rest_df, cuisine_aff, z, city_cuisine, rng)

    # --- persist ---
    np.save(C.ARTIFACT_DIR / "item_latent_z.npy", z)
    np.save(C.ARTIFACT_DIR / "cuisine_aff.npy", cuisine_aff)
    np.save(C.ARTIFACT_DIR / "city_cuisine.npy", city_cuisine)
    items_df.to_parquet(C.DATA_DIR / "items.parquet", index=False)
    rest_df.to_parquet(C.DATA_DIR / "restaurants.parquet", index=False)
    users_df.to_parquet(C.DATA_DIR / "users.parquet", index=False)
    orders_df.to_parquet(C.DATA_DIR / "orders.parquet", index=False)
    order_items_df.to_parquet(C.DATA_DIR / "order_items.parquet", index=False)
    imp_df.to_parquet(C.DATA_DIR / "impressions.parquet", index=False)
    cand_df.to_parquet(C.DATA_DIR / "candidates.parquet", index=False)

    stats = dict(
        n_users=int(len(users_df)),
        n_cold_users=int(users_df["is_cold_start"].sum()),
        n_items=int(len(items_df)),
        n_restaurants=int(len(rest_df)),
        n_orders=int(len(orders_df)),
        n_impressions=int(len(imp_df)),
        n_interactions=int(len(cand_df)),
        pos_rate=float(cand_df["label"].mean()),
        avg_cart_size=float(orders_df["cart_size"].mean()),
        peak_hour_share=float(orders_df["hour"].isin(
            list(C.LUNCH_HOURS) + list(C.DINNER_HOURS)).mean()),
    )
    (C.RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nDATA STATS:\n" + json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
