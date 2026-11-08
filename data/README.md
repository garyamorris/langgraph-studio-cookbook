# Bundled datasets

All data is **synthetic and fictional**, generated with a fixed seed by `scripts/make_datasets.py` (re-run it to
regenerate; the committed files are the source of truth for the examples).

The universe is **Lumen Home**, a made-up smart-home hardware company. Tickets, reviews, orders, the SQL database, the
refund requests and the support knowledge base all describe the same eight products, so examples can cross-reference
each other. A second made-up universe of eight listed companies (`ACMR`, `BORE`, `CBLT`, `DLPH`, `EMBR`, `FTHM`,
`GRNT`, `HELO`) backs the market examples.

| File | Rows | What it is | Used for |
|---|---|---|---|
| `support_tickets.csv` | 240 | `ticket_id, customer_id, channel, created_at, subject, body, category, priority, sentiment` - labelled. `T-2301..T-2304` are urgent safety/privacy tickets. | routing, triage, evaluation |
| `product_reviews.csv` | 320 | `review_id, product, rating, title, text, date, verified_purchase, helpful_votes`. Six reviews contain fire/legal/privacy risk language. | parallel analysis, moderation, summarisation |
| `sales.csv` | 1,210 | Order lines, 2025-11 -> 2026-10: `order_id, order_date, customer_id, region, segment, channel, status, product, category, quantity, unit_price, discount_pct, revenue, cost, margin`. Built-in stories: Nov/Dec seasonality, a *Lumen Therm* price cut on 2026-05-01 ($159 -> $149), a *Cam Pro* recall window (Apr-Jun 2026) with many returns, discounts that lift units. | analytics agents, report writing, elasticity |
| `store.sqlite` | 180 customers, 845 orders, 1,210 items | The same shop, relational: `customers, products, orders, order_items`. | text-to-SQL |
| `refund_requests.jsonl` | 45 | `request_id, order_id, customer_id, product, amount, reason_code, reason_text, days_since_delivery, has_photo_evidence, prior_refunds_12m`. `order_id` exists in `store.sqlite`. | refund approval |
| `loan_applications.csv` | 150 | Applicant financials + free-text `underwriter_notes` + a hidden-policy `policy_label` (approve/review/decline). | probabilistic decision gates |
| `news_headlines.csv` | 180 | `headline_id, date, ticker, headline, label` (positive/negative/neutral, ~15% deliberately tricky/mixed). | classifier comparison, research |
| `prices.csv` | 1,280 | 160 trading days of OHLCV for 8 fictional tickers (2026-03-23 -> 2026-10-30). | market examples |
| `companies.csv` | 8 | `ticker, company, sector, beta` | market examples |
| `kb/support/*.md` | 10 | Lumen Home policies (returns, refund approval matrix [internal], shipping, warranty, wifi, firmware, account, billing, privacy, safety). Articles titled "(internal)" are excluded from the default retriever. | RAG, specialists, refund policy |
| `kb/langgraph/*.md` | 15 | A compact LangGraph handbook (state, routing, Send, subgraphs, checkpointers, interrupts, time travel, store, streaming, functional API, retries/caching/errors, Studio, multi-agent, prebuilt, Decisions API). | "ask the handbook" RAG |

Access them through `deepdive.common.data` (`read_csv`, `read_jsonl`, `db_connect`, `db_schema`, `kb_docs`) and
`deepdive.common.retrieval.get_retriever`.
