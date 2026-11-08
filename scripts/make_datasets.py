"""Generate the synthetic, seeded datasets that the examples run against.

Everything here is FICTIONAL. The universe is "Lumen Home", a smart-home hardware
company, plus eight made-up listed companies for the market examples. All files are
written under ``data/`` and committed, so you only re-run this to change the data:

    uv run python scripts/make_datasets.py

Outputs
-------
data/support_tickets.csv      240 labelled customer tickets (category/priority/sentiment)
data/product_reviews.csv      320 product reviews (some contain safety/legal risk language)
data/news_headlines.csv       180 labelled market headlines (fictional tickers)
data/prices.csv               160 trading days of OHLCV for 8 fictional tickers
data/sales.csv                denormalised order lines (for pandas / tool examples)
data/store.sqlite             the same shop as a relational DB (for text-to-SQL)
data/loan_applications.csv    150 loan applications with free-text underwriter notes
data/refund_requests.jsonl    45 refund requests whose order_ids exist in store.sqlite
data/kb/**                    markdown knowledge bases (see scripts/kb_content.py)
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from kb_content import LANGGRAPH_KB, SUPPORT_KB

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42

rng = random.Random(SEED)
nprng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# The Lumen Home universe
# ---------------------------------------------------------------------------
PRODUCTS = [
    # product_id, name, category, price, unit_cost, popularity
    (1, "Lumen Hub", "hubs", 129.0, 52.0, 10),
    (2, "Lumen Cam Pro", "cameras", 89.0, 38.0, 14),
    (3, "Lumen Therm", "climate", 149.0, 71.0, 8),
    (4, "Lumen Bell", "doorbells", 119.0, 49.0, 9),
    (5, "Lumen Plug 4-pack", "plugs", 39.0, 14.0, 16),
    (6, "Lumen Sensor Kit", "sensors", 59.0, 22.0, 9),
    (7, "Lumen Lock", "locks", 179.0, 84.0, 5),
    (8, "Lumen Bulb 6-pack", "lighting", 49.0, 17.0, 12),
]
PRODUCT_NAMES = [p[1] for p in PRODUCTS]
REGIONS = ["North America", "Europe", "UK & Ireland", "APAC"]
REGION_W = [0.46, 0.27, 0.15, 0.12]
SEGMENTS = ["consumer", "consumer", "consumer", "prosumer", "small_business"]
CHANNELS = ["web", "app", "retail_partner", "marketplace"]
CHANNEL_W = [0.55, 0.25, 0.15, 0.05]

FIRST = ["Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Ines", "Omar", "Priya", "Lucas", "Hana", "Mateo",
         "Chloe", "Arjun", "Sofia", "Kenji", "Amara", "Felix", "Nora", "Diego", "Leila", "Tomas", "Yara", "Ivan"]
LAST = ["Alvarez", "Brennan", "Chen", "Dubois", "Eriksen", "Fofana", "Garcia", "Haddad", "Iyer", "Jensen",
        "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quinn", "Rossi", "Silva", "Tanaka"]

END_DATE = date(2026, 10, 30)  # last trading day in the market data
SALES_START = date(2025, 11, 1)
SALES_END = date(2026, 10, 31)


def _d(d: date) -> str:
    return d.isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Customers, orders, order items  ->  sales.csv + store.sqlite
# ---------------------------------------------------------------------------
def build_store() -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    customers = []
    for cid in range(1, 181):
        region = rng.choices(REGIONS, REGION_W)[0]
        first, last = rng.choice(FIRST), rng.choice(LAST)
        customers.append(
            {
                "customer_id": cid,
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{cid}@example.com",
                "region": region,
                "segment": rng.choice(SEGMENTS),
                "signup_date": _d(SALES_START - timedelta(days=rng.randint(10, 900))),
            }
        )

    # seasonality: Nov/Dec spike, Jan dip, gentle upward trend
    month_factor = {11: 1.5, 12: 1.9, 1: 0.8, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.0, 7: 0.95, 8: 1.0, 9: 1.1, 10: 1.2}
    orders, items = [], []
    order_id, item_id = 1000, 1
    cur = SALES_START
    while cur <= SALES_END:
        n_orders = np.clip(int(nprng.poisson(1.9 * month_factor[cur.month] * (1 + 0.0006 * (cur - SALES_START).days))), 0, 9)
        for _ in range(n_orders):
            order_id += 1
            cust = rng.choice(customers)
            channel = rng.choices(CHANNELS, CHANNEL_W)[0]
            discount = rng.choices([0, 5, 10, 15, 20], [55, 10, 17, 10, 8])[0]
            status = rng.choices(["delivered", "returned", "cancelled"], [90, 6, 4])[0]
            n_lines = rng.choices([1, 2, 3], [60, 30, 10])[0]
            chosen = set()
            for _ in range(n_lines):
                pid, name, cat, price, cost, pop = rng.choices(PRODUCTS, [p[5] for p in PRODUCTS])[0]
                if pid in chosen:
                    continue
                chosen.add(pid)
                base_q = rng.choices([1, 2, 3], [70, 22, 8])[0]
                qty = max(1, round(base_q * (1 + 0.035 * discount)))  # discounts lift units
                unit_price = price
                if pid == 3 and cur < date(2026, 5, 1):  # Therm price cut on 1 May 2026
                    unit_price = 159.0
                # Cam Pro recall window: fewer sales, many returns
                line_status = status
                if pid == 2 and date(2026, 4, 1) <= cur <= date(2026, 6, 30):
                    qty = max(1, qty - 1) if rng.random() < 0.5 else qty
                    if rng.random() < 0.3:
                        line_status = "returned"
                items.append(
                    {
                        "item_id": item_id,
                        "order_id": order_id,
                        "product_id": pid,
                        "quantity": qty,
                        "unit_price": unit_price,
                        "_status": line_status,
                    }
                )
                item_id += 1
            # an order is "returned" if any line was returned
            order_status = "returned" if any(i["_status"] == "returned" for i in items if i["order_id"] == order_id) else status
            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "order_date": _d(cur),
                    "channel": channel,
                    "status": order_status,
                    "discount_pct": discount,
                }
            )
        cur += timedelta(days=1)

    for i in items:
        i.pop("_status")

    products = [
        {"product_id": p[0], "name": p[1], "category": p[2], "unit_price": p[3], "unit_cost": p[4]} for p in PRODUCTS
    ]

    cdf, odf, idf, pdf = map(pd.DataFrame, (customers, orders, items, products))
    sales = (
        idf.merge(odf, on="order_id")
        .merge(cdf[["customer_id", "region", "segment"]], on="customer_id")
        .merge(pdf[["product_id", "name", "category", "unit_cost"]], on="product_id")
        .rename(columns={"name": "product"})
    )
    sales["revenue"] = (sales.quantity * sales.unit_price * (1 - sales.discount_pct / 100)).round(2)
    sales["cost"] = (sales.quantity * sales.unit_cost).round(2)
    sales["margin"] = (sales.revenue - sales.cost).round(2)
    sales = sales[
        ["order_id", "order_date", "customer_id", "region", "segment", "channel", "status", "product", "category",
         "quantity", "unit_price", "discount_pct", "revenue", "cost", "margin"]
    ].sort_values(["order_date", "order_id"])
    return sales, customers, orders, items


def write_store(sales: pd.DataFrame, customers, orders, items) -> None:
    sales.to_csv(DATA / "sales.csv", index=False)
    print(f"  wrote data/sales.csv  ({len(sales)} rows)")

    db_path = DATA / "store.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, email TEXT, region TEXT,
                                segment TEXT, signup_date TEXT);
        CREATE TABLE products  (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT,
                                unit_price REAL, unit_cost REAL);
        CREATE TABLE orders    (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers,
                                order_date TEXT, channel TEXT, status TEXT, discount_pct INTEGER);
        CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders,
                                  product_id INTEGER REFERENCES products, quantity INTEGER, unit_price REAL);
        """
    )
    con.executemany("INSERT INTO customers VALUES (:customer_id,:name,:email,:region,:segment,:signup_date)", customers)
    con.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?)", [(p[0], p[1], p[2], p[3], p[4]) for p in PRODUCTS]
    )
    con.executemany(
        "INSERT INTO orders VALUES (:order_id,:customer_id,:order_date,:channel,:status,:discount_pct)", orders
    )
    con.executemany(
        "INSERT INTO order_items VALUES (:item_id,:order_id,:product_id,:quantity,:unit_price)", items
    )
    con.commit()
    con.close()
    print(f"  wrote data/store.sqlite  ({len(customers)} customers, {len(orders)} orders, {len(items)} items)")


# ---------------------------------------------------------------------------
# Support tickets
# ---------------------------------------------------------------------------
def build_tickets() -> list[dict]:
    codes = ["PROMO10", "WELCOME15", "HOLIDAY20", "SPRING5"]
    errs = ["E-102", "E-317", "E-504", "WIFI-7", "AUTH-12"]
    fws = ["2.4.1", "2.5.0", "3.0.2"]
    feats = ["geofencing for the thermostat", "a dark mode in the app", "Matter support on the Lock",
             "longer clip history on the Cam Pro", "multi-home support"]
    T = {
        "billing": [
            ("Charged twice for order {order}", "Hi, I was charged ${amt} twice for order {order} on {date}. Please refund the duplicate charge."),
            ("Subscription renewed unexpectedly", "My Lumen Care plan renewed at ${amt} and I'm sure I cancelled it back in {month}. Can you reverse it?"),
            ("Need an invoice with VAT number", "Could you send an invoice with our company VAT number for order {order}? Our accounting closes Friday."),
            ("Promo code {code} not applied", "The code {code} did not take anything off my {product}. Please apply the discount to order {order}."),
            ("Refund still not received", "It has been {n} days since you approved the refund for order {order}. Where is my money?"),
        ],
        "technical": [
            ("{product} keeps going offline", "My {product} drops off wifi every few hours. I rebooted the router twice and the app shows error {err}."),
            ("Firmware update stuck", "The update to firmware {fw} has been stuck at {pct}% for hours on my {product}. Is it safe to unplug it?"),
            ("Cannot pair {product}", "I can't pair my new {product} with the Hub. The light blinks amber and the app says {err}."),
            ("Video is choppy", "Live view on the Cam Pro buffers constantly in the evening although my internet is fast. Error {err} sometimes."),
            ("Hub shows a red light", "The Lumen Hub has had a solid red light since the power cut yesterday and none of my devices respond."),
        ],
        "account": [
            ("Cannot log in after password reset", "I reset my password but the app still says invalid credentials. I'm using {email}."),
            ("Change email address on account", "Please change the email on my account from {email} to a new address, I no longer have access to it."),
            ("Delete my account and data", "I want my account and all camera recordings deleted under GDPR. How do I confirm this?"),
            ("Two-factor codes not arriving", "I'm not receiving the SMS codes for two-factor login, so I'm locked out of the app."),
            ("Add a family member to my home", "How do I give my partner access to our Hub and cameras without sharing my password?"),
        ],
        "shipping": [
            ("Where is order {order}?", "Order {order} shipped {n} days ago and the tracking hasn't changed. When will it arrive?"),
            ("Marked delivered but not received", "Tracking says order {order} was delivered yesterday but nothing is here. Neighbours haven't seen it either."),
            ("Wrong item in the box", "I ordered a {product} but received a different product in order {order}."),
            ("Box arrived damaged", "The box for order {order} was crushed and the {product} has a cracked case."),
            ("Change delivery address", "I need to change the delivery address on order {order} before it ships. I've moved house."),
        ],
        "product_feedback": [
            ("Feature request", "Would love to see {feat}. Otherwise I'm really happy with the {product}."),
            ("Love the {product}", "Just wanted to say the {product} setup took five minutes. Great job, the team should be proud."),
            ("App redesign is confusing", "The new app layout buries the automations tab. Please bring back the old navigation."),
            ("Suggestion for the {product}", "It would help if the {product} had a quieter chime option for night time."),
        ],
    }
    special = [
        ("URGENT: plug smells of burning", "technical", "urgent", "angry",
         "One of the plugs in my Lumen Plug 4-pack got extremely hot and there's a burning plastic smell. I've unplugged it. This is a fire risk!!"),
        ("Locked out of my own house", "technical", "urgent", "angry",
         "My Lumen Lock stopped responding and I'm locked out with my kids outside. I need help RIGHT NOW."),
        ("Cam Pro showing someone else's video", "account", "urgent", "angry",
         "I opened the app and the Cam Pro feed showed a stranger's living room. This is a serious privacy breach and I'm contacting a regulator."),
        ("Therm heating stuck on", "technical", "urgent", "negative",
         "The Lumen Therm won't turn the heating off and the house is 29C. Manual override doesn't work."),
    ]
    sentiment_by_cat = {
        "billing": (["negative", "angry", "neutral"], [50, 20, 30]),
        "technical": (["negative", "neutral", "angry"], [45, 35, 20]),
        "account": (["neutral", "negative", "angry"], [50, 35, 15]),
        "shipping": (["negative", "neutral", "angry"], [45, 30, 25]),
        "product_feedback": (["positive", "neutral", "negative"], [55, 30, 15]),
    }
    rows = []
    start = date(2026, 8, 1)
    for i in range(1, 237):
        cat = rng.choices(list(T), [22, 28, 18, 20, 12])[0]
        subj_t, body_t = rng.choice(T[cat])
        slots = dict(
            order=f"#{rng.randint(1001, 1700)}", amt=rng.choice([19, 29, 39, 49, 89, 129, 149]),
            date=_d(start + timedelta(days=rng.randint(0, 85))), month=rng.choice(["March", "June", "August"]),
            code=rng.choice(codes), product=rng.choice(PRODUCT_NAMES), n=rng.randint(3, 21), err=rng.choice(errs),
            fw=rng.choice(fws), pct=rng.choice([12, 47, 63, 88, 99]), feat=rng.choice(feats),
            email=f"user{rng.randint(10, 99)}@example.com",
        )
        sent_opts, sent_w = sentiment_by_cat[cat]
        sentiment = rng.choices(sent_opts, sent_w)[0]
        priority = {"billing": "medium", "technical": "medium", "account": "medium", "shipping": "medium", "product_feedback": "low"}[cat]
        if sentiment == "angry":
            priority = "high"
        body = body_t.format(**slots)
        if sentiment == "angry":
            body += rng.choice([" This is the third time I'm writing!", " I'm extremely frustrated.", " Fix this today or I'm disputing the charge."])
        rows.append(
            {
                "ticket_id": f"T-{2000 + i}", "customer_id": rng.randint(1, 180), "channel": rng.choice(["email", "chat", "web_form"]),
                "created_at": _d(start + timedelta(days=rng.randint(0, 90))), "subject": subj_t.format(**slots), "body": body,
                "category": cat, "priority": priority, "sentiment": sentiment,
            }
        )
    for j, (subj, cat, pri, sent, body) in enumerate(special, start=1):
        rows.append(
            {"ticket_id": f"T-{2300 + j}", "customer_id": rng.randint(1, 180), "channel": "chat",
             "created_at": _d(START_TICKETS_END), "subject": subj, "body": body, "category": cat, "priority": pri, "sentiment": sent}
        )
    rng.shuffle(rows)
    return rows


START_TICKETS_END = date(2026, 10, 28)

# ---------------------------------------------------------------------------
# Product reviews
# ---------------------------------------------------------------------------
def build_reviews() -> list[dict]:
    pos = {
        "Lumen Hub": ["Setup took five minutes and it found all my devices.", "Rock solid. Automations just work.", "Responds instantly, even with 40 devices paired."],
        "Lumen Cam Pro": ["Picture quality at night is excellent.", "Motion alerts are accurate, no more false alarms from trees.", "Clip history is easy to scrub through."],
        "Lumen Therm": ["Cut my heating bill noticeably this winter.", "The schedule learning actually learned our routine.", "Looks great on the wall and the app is clear."],
        "Lumen Bell": ["Package alerts are brilliant.", "Two-way audio is crisp.", "Installed it myself in ten minutes."],
        "Lumen Plug 4-pack": ["Cheap, simple and reliable.", "Energy monitoring is a nice touch for the price.", "Great for scheduling lamps and the coffee maker."],
        "Lumen Sensor Kit": ["Door sensors are tiny and the batteries last.", "Caught a leak under the sink early, paid for itself.", "Easy to place and pair."],
        "Lumen Lock": ["Auto-lock gives me peace of mind.", "Guest codes are super convenient.", "Feels solid and the keypad is responsive."],
        "Lumen Bulb 6-pack": ["Colours are vibrant and dimming is smooth.", "Great value for six bulbs.", "Pairs with the Hub instantly."],
    }
    neg = {
        "Lumen Hub": ["Drops offline after firmware updates.", "App is slow to show device status.", "Needed three resets to pair my sensors."],
        "Lumen Cam Pro": ["Live view buffers a lot on my wifi.", "Subscription is required for clip history.", "Night mode washes out faces."],
        "Lumen Therm": ["Scheduling is confusing and it ignored my away mode.", "Display ghosting after a month.", "Wiring guide was unclear."],
        "Lumen Bell": ["Notifications arrive 30 seconds late.", "Battery drains in three weeks in the cold.", "Chime is too quiet."],
        "Lumen Plug 4-pack": ["Two of four lost connection within a month.", "Too bulky, blocks the neighbouring socket.", "App crashes when I rename a plug."],
        "Lumen Sensor Kit": ["Adhesive failed and one fell off.", "False tamper alerts.", "Hub range is poor through brick walls."],
        "Lumen Lock": ["Keypad stops responding when it's cold.", "Auto-lock fired while I was outside and I got locked out.", "Pricey for what it does."],
        "Lumen Bulb 6-pack": ["One bulb flickered from day one.", "Warm white is too yellow.", "Disconnects from the Hub every few days."],
    }
    risk = [
        ("Lumen Plug 4-pack", 1, "Plug got hot and smelled like burning plastic", "One of the plugs became very hot and left a scorch mark on my socket. This is a fire hazard, I'm reporting it."),
        ("Lumen Therm", 1, "Overheats and trips the breaker", "The Therm got hot to the touch and tripped our breaker twice. I'm talking to a lawyer about this."),
        ("Lumen Cam Pro", 2, "Camera exposes my feed", "I could see another household's feed after the last update. Serious privacy issue and possibly illegal."),
        ("Lumen Bulb 6-pack", 1, "Bulb shattered", "A bulb cracked and sparked when I switched it on. Could have injured my child."),
        ("Lumen Lock", 1, "Lock jammed with me inside", "The Lock jammed and I couldn't get out until the fire brigade arrived. Dangerous product."),
        ("Lumen Hub", 2, "Looks like a counterfeit", "The box and logo look off. I think this is a counterfeit listing on the marketplace, I'm filing a chargeback."),
    ]
    rows, n = [], 0
    start = date(2026, 1, 1)
    for _ in range(314):
        n += 1
        pname = rng.choices(PRODUCT_NAMES, [p[5] for p in PRODUCTS])[0]
        rating = rng.choices([1, 2, 3, 4, 5], [6, 8, 14, 32, 40])[0]
        if pname in ("Lumen Plug 4-pack", "Lumen Bulb 6-pack"):
            rating = rng.choices([1, 2, 3, 4, 5], [9, 11, 16, 30, 34])[0]
        if rating >= 4:
            text = " ".join(rng.sample(pos[pname], 2 if rating == 5 else 1)) + (" Highly recommend." if rating == 5 else "")
            title = rng.choice(["Does what it says", "Great buy", "Very happy", "Solid"])
        elif rating == 3:
            text = rng.choice(pos[pname]) + " But " + rng.choice(neg[pname]).lower()
            title = rng.choice(["Mixed feelings", "Okay overall", "Good, not great"])
        else:
            text = " ".join(rng.sample(neg[pname], 2 if rating == 1 else 1)) + (" Returning it." if rating == 1 else "")
            title = rng.choice(["Disappointed", "Not reliable", "Wouldn't buy again"])
        rows.append(
            {"review_id": f"R-{n:04d}", "product": pname, "rating": rating, "title": title, "text": text,
             "date": _d(start + timedelta(days=rng.randint(0, 300))), "verified_purchase": rng.random() < 0.88,
             "helpful_votes": int(nprng.poisson(2.5))}
        )
    for pname, rating, title, text in risk:
        n += 1
        rows.append(
            {"review_id": f"R-{n:04d}", "product": pname, "rating": rating, "title": title, "text": text,
             "date": _d(start + timedelta(days=rng.randint(180, 300))), "verified_purchase": True, "helpful_votes": rng.randint(8, 40)}
        )
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Fictional market: companies, headlines, prices
# ---------------------------------------------------------------------------
COMPANIES = [
    # ticker, name, sector, drift (annual), vol (annual), market beta
    ("ACMR", "Acme Robotics", "industrial tech", 0.14, 0.38, 1.25),
    ("BORE", "Borealis Energy", "energy", 0.06, 0.32, 0.85),
    ("CBLT", "Cobalt Semiconductor", "semiconductors", 0.22, 0.46, 1.45),
    ("DLPH", "Delphi Health", "healthcare", 0.08, 0.28, 0.70),
    ("EMBR", "Ember Foods", "consumer staples", 0.04, 0.20, 0.55),
    ("FTHM", "Fathom Cloud", "software", 0.18, 0.41, 1.30),
    ("GRNT", "Granite Bank", "financials", 0.07, 0.27, 1.05),
    ("HELO", "Helio Solar", "renewables", 0.12, 0.52, 1.35),
]


def build_prices() -> pd.DataFrame:
    days = pd.bdate_range(end=pd.Timestamp(END_DATE), periods=160)
    n = len(days)
    market = nprng.normal(0.0003, 0.009, n)
    frames = []
    for tk, _name, _sector, mu, vol, beta in COMPANIES:
        idio_vol = max(vol / math.sqrt(252) - abs(beta) * 0.003, 0.004)
        r = beta * market + nprng.normal(mu / 252, idio_vol, n)
        # a couple of event-driven jumps per name
        for k in nprng.choice(np.arange(10, n - 5), 2, replace=False):
            r[k] += nprng.choice([-1, 1]) * nprng.uniform(0.04, 0.09)
        close = 100 * nprng.uniform(0.4, 2.2) * np.exp(np.cumsum(r))
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * (1 + nprng.normal(0, 0.004, n))
        hi = np.maximum(open_, close) * (1 + np.abs(nprng.normal(0.004, 0.003, n)))
        lo = np.minimum(open_, close) * (1 - np.abs(nprng.normal(0.004, 0.003, n)))
        vol_sh = (nprng.lognormal(14.2, 0.35, n) * (1 + 6 * np.abs(r))).astype(int)
        frames.append(
            pd.DataFrame(
                {"date": days.strftime("%Y-%m-%d"), "ticker": tk, "open": open_.round(2), "high": hi.round(2),
                 "low": lo.round(2), "close": close.round(2), "volume": vol_sh}
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_headlines(prices: pd.DataFrame) -> list[dict]:
    pos = [
        "{n} beats quarterly revenue estimates as demand accelerates",
        "{n} raises full-year guidance after record bookings",
        "{n} wins multi-year contract worth ${m}M",
        "Analysts upgrade {n} citing margin expansion",
        "{n} announces ${m}M buyback and a higher dividend",
        "{n} unveils product line that outpaces rivals in early tests",
    ]
    neg = [
        "{n} misses earnings estimates and cuts guidance",
        "Regulators open probe into {n} accounting practices",
        "{n} warns of supply shortages hitting second-half sales",
        "Analysts downgrade {n} on weakening demand",
        "{n} CFO resigns abruptly ahead of results",
        "{n} recalls flagship product after safety complaints",
    ]
    neu = [
        "{n} to present at industry conference next week",
        "{n} schedules earnings call for the 14th",
        "{n} names new head of investor relations",
        "{n} completes routine refinancing of existing notes",
        "{n} opens regional office in Singapore",
    ]
    hard = [  # mixed or tricky; ground-truth label chosen by net effect
        ("{n} beats revenue estimates but slashes full-year outlook", "negative"),
        ("{n} shares fall despite record bookings as one-off charge weighs", "neutral"),
        ("{n} misses on EPS yet announces ${m}M buyback and strong order backlog", "neutral"),
        ("{n} faces probe, but analysts say exposure is immaterial", "neutral"),
        ("{n} lifts guidance only after accounting restatement", "negative"),
        ("{n} cuts jobs as part of efficiency plan; margins to improve", "positive"),
    ]
    rows = []
    days = sorted(prices.date.unique())
    for i in range(1, 181):
        tk, name, *_ = rng.choice(COMPANIES)
        m = rng.choice([40, 75, 120, 250, 600])
        r = rng.random()
        if r < 0.32:
            label, tmpl = "positive", rng.choice(pos)
        elif r < 0.62:
            label, tmpl = "negative", rng.choice(neg)
        elif r < 0.84:
            label, tmpl = "neutral", rng.choice(neu)
        else:
            tmpl, label = rng.choice(hard)
        rows.append({"headline_id": f"H-{i:04d}", "date": rng.choice(days[10:]), "ticker": tk,
                     "headline": tmpl.format(n=name, m=m), "label": label})
    rows.sort(key=lambda x: (x["date"], x["ticker"]))
    return rows


# ---------------------------------------------------------------------------
# Loan applications
# ---------------------------------------------------------------------------
def build_loans() -> list[dict]:
    purposes = ["debt_consolidation", "home_improvement", "small_business", "vehicle", "medical", "education"]
    good = ["Long-tenured employee; stable salary and 12 months of savings.", "Co-signer has excellent credit.", "Existing customer, no late payments in 5 years.", "Pays rent on time for 6+ years."]
    bad = ["Two late payments in 2024.", "Income is mostly seasonal gig work.", "Recently changed jobs; still on probation.", "Applied for three other loans this month.", "Owes back taxes on a payment plan."]
    neutral = ["First-time borrower, thin credit file.", "Self-employed; two years of tax returns provided.", "Loan will fund a home renovation with contractor quotes attached.", "Says income will rise soon after a promotion (unverified)."]
    rows = []
    for i in range(1, 151):
        income = int(np.clip(nprng.lognormal(11.0, 0.45), 22000, 260000))
        score = int(np.clip(nprng.normal(675, 70), 480, 830))
        debt = int(income / 12 * np.clip(nprng.normal(0.28, 0.14), 0.02, 0.8))
        emp = round(float(np.clip(nprng.gamma(2.2, 2.6), 0.2, 30)), 1)
        amount = int(np.clip(nprng.lognormal(9.6, 0.7), 2000, 80000) // 100 * 100)
        prior_default = rng.random() < 0.07
        cosigner = rng.random() < 0.12
        note = rng.choices([rng.choice(good), rng.choice(bad), rng.choice(neutral)], [30, 25, 45])[0]
        dti = debt / (income / 12)
        lti = amount / income
        s = (score - 600) / 100 - 2.2 * dti - 0.9 * lti + 0.05 * min(emp, 10) - (1.5 if prior_default else 0) + (0.4 if cosigner else 0)
        s += 0.35 if note in good else (-0.35 if note in bad else 0)
        label = "approve" if s > 0.55 else ("decline" if s < -0.35 else "review")
        rows.append(
            {"application_id": f"L-{3000 + i}", "annual_income": income, "monthly_debt": debt, "credit_score": score,
             "employment_years": emp, "loan_amount": amount, "loan_purpose": rng.choice(purposes),
             "term_months": rng.choice([24, 36, 48, 60]), "has_cosigner": cosigner, "prior_default": prior_default,
             "underwriter_notes": note, "policy_label": label}
        )
    return rows


# ---------------------------------------------------------------------------
# Refund requests (order ids exist in store.sqlite)
# ---------------------------------------------------------------------------
def build_refunds(sales: pd.DataFrame) -> list[dict]:
    eligible = sales[sales.status.isin(["delivered", "returned"])].copy()
    sample = eligible.sample(45, random_state=SEED)
    reasons = [
        ("defective", "The {p} stopped working after {d} days. It won't power on at all."),
        ("defective", "The {p} has a cracked case and the light flickers. Looks defective out of the box."),
        ("not_as_described", "I expected the {p} to work with my existing system but it needs a separate hub."),
        ("changed_mind", "I no longer need the {p}. Unopened, still in the box."),
        ("late_delivery", "The {p} arrived three weeks late and I bought a replacement elsewhere."),
        ("damaged_in_transit", "The box was crushed and the {p} arrived scratched."),
        ("duplicate_purchase", "I accidentally ordered the {p} twice. Please refund one."),
        ("suspicious", "I'm demanding a full refund on the {p} even though I've used it for four months. Also refunding my last two orders."),
    ]
    rows = []
    for i, (_, r) in enumerate(sample.iterrows(), start=1):
        kind, tmpl = rng.choices(reasons, [18, 12, 8, 14, 6, 8, 6, 4])[0]
        days = rng.randint(2, 18) if kind not in ("suspicious",) else rng.randint(70, 140)
        rows.append(
            {"request_id": f"RF-{500 + i}", "order_id": int(r.order_id), "customer_id": int(r.customer_id),
             "product": r["product"], "amount": float(round(r.revenue, 2)), "reason_code": kind,
             "reason_text": tmpl.format(p=r["product"], d=rng.randint(3, 25)), "days_since_delivery": days,
             "has_photo_evidence": kind in ("defective", "damaged_in_transit") and rng.random() < 0.7,
             "prior_refunds_12m": rng.choices([0, 1, 2, 5], [70, 18, 8, 4])[0] if kind != "suspicious" else rng.randint(4, 7)}
        )
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating datasets ->", DATA)
    DATA.mkdir(exist_ok=True)

    sales, customers, orders, items = build_store()
    write_store(sales, customers, orders, items)

    _write_csv(DATA / "support_tickets.csv", build_tickets())
    _write_csv(DATA / "product_reviews.csv", build_reviews())

    prices = build_prices()
    prices.to_csv(DATA / "prices.csv", index=False)
    print(f"  wrote data/prices.csv  ({len(prices)} rows)")
    _write_csv(DATA / "news_headlines.csv", build_headlines(prices))
    pd.DataFrame(
        [{"ticker": t, "company": n, "sector": s, "beta": b} for t, n, s, _m, _v, b in COMPANIES]
    ).to_csv(DATA / "companies.csv", index=False)
    print("  wrote data/companies.csv  (8 rows)")

    _write_csv(DATA / "loan_applications.csv", build_loans())

    refunds = build_refunds(sales)
    with (DATA / "refund_requests.jsonl").open("w", encoding="utf-8") as f:
        for r in refunds:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote data/refund_requests.jsonl  ({len(refunds)} rows)")

    for corpus, docs in (("langgraph", LANGGRAPH_KB), ("support", SUPPORT_KB)):
        out = DATA / "kb" / corpus
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.md"):
            old.unlink()
        for slug, (title, body) in docs.items():
            (out / f"{slug}.md").write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
        print(f"  wrote data/kb/{corpus}/  ({len(docs)} docs)")


if __name__ == "__main__":
    main()
