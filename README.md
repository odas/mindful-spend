# mindful-spend
**Mindful expense tracking for India's fragmented payment reality.**

---
india-ocr-expense-tracker, Python · Streamlit · Gemini Vision API · pandas · Plotly · google-genai SDK

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Gemini Vision](https://img.shields.io/badge/Gemini-Vision%20API-orange)
![pandas](https://img.shields.io/badge/pandas-2.x-green)
---

## The problem with expense tracking in India

You pay for groceries on Amazon using an ICICI credit card via GPay. You top up your Paytm wallet from your SBI account and use it for a Zomato order. You split a cab fare via PhonePe from a different bank. At the end of the month, your bank statement says `UPI-ZOMATO-PAY-98765@HDFC` and `UPI-VPA-KUMAR88@OKHDFCBANK` — was that the Ola driver or your regular delivery guy?

This is the reality of Indian spending: fragmented across apps, UPI handles, wallets, and cards — often four payment methods for the same app in the same month.

Most expense trackers handle this badly, and in ways that quietly compromise your privacy:

- **SMS scrapers** (like Axio/Walnut) read your entire inbox to catch bank alerts — which means they technically have access to your OTPs, personal messages, and private conversations. They also double-count wallet top-ups as expenses, and they see `Zomato - ₹450`, not what you ordered.
- **Account Aggregator apps** (like Fold) are more private — RBI-regulated, consent-based — but they still only see the bank debit. The line items inside your Zepto or HUFT order are invisible to them.
- **Manual apps** (like Money Manager) are accurate but brutal. Logging every ₹10 chai is a full-time job. Most people stop after two weeks.

None of them tell you what you actually bought.

---

## A different philosophy

Most tracking apps want you to stop thinking about your money. They automate everything and hand you a pie chart at the end of the month.

ReceiptIQ takes the opposite view: **the act of reviewing what you spent is the point.**

When you upload a screenshot and confirm each line item, you're making a deliberate choice to engage with your spending. That ₹3,200 midnight air fryer from Zepto isn't just a data point — it's a decision you're now conscious of. The mild friction is intentional.

The LLM handles the OCR grunt work so you're not fighting with a spreadsheet. You stay in the loop.

---

## What ReceiptIQ actually does

Upload a payment screenshot. Gemini Vision reads it and extracts line items with amounts, quantities, and platform context — pre-filling an editable form. You review, correct, categorise, and save. Your data lives in a local CSV, clean and ready for analysis.

**For grocery and product orders:** one row per line item — so a ₹680 HUFT order becomes three rows: dry food, wet food, litter.

**For restaurant orders:** one row per order — because `Chicken Biryani, Coke, Raita` from Zomato is one eating-out event, not three separate data points.

This granularity is what makes ReceiptIQ useful for tracking niches that actually matter to you.

---

## Custom categories: useful data, not guilt data

Most apps give you fixed categories you cannot change. ReceiptIQ is built around a taxonomy you own.

The default setup has 8 categories with sub-categories:

| Category | Sub-categories |
|---|---|
| Big Purchases | Electronics, Tax, Vet, Travel |
| Cat | Dry Food, Wet Food, Snacks, Litter, Cat Toys |
| Eating Out | *(restaurant name as item)* |
| Grocery & Home | Snacks, Healthy, Beverages, Home, Self-care, Stationery, Electronics, Hobby |
| Health | Medicine, Self-care |
| Shopping | Clothes |
| Utility | Phone, Cloud Storage, News, Hobby, Travel |
| Gift & Entertainment | Gift, Show |

The `Cat` category exists because knowing you spent ₹4,200 on your cat this month — broken down by dry food vs wet food vs vet vs toys — is genuinely useful information. Knowing you ordered Swiggy three times is mostly just guilt.

The `notes` column takes niche tracking further. Add brand, unit weight, unit price per item. Over a year, you can run a simple pandas query to compare the average per-packet price of Whiskas vs Temptations vs Royal Canin across Zepto, Amazon, and HUFT. No SMS scraper or bank-linked app can do this — the data simply does not exist in any transaction record.

---

## Privacy: honest trade-offs

ReceiptIQ doesn't read your SMS. It doesn't link your bank account. It doesn't run in the background. You choose what to upload and when.

The caveat: screenshots you upload are processed by Google's Gemini API. Google sees your line items and spending patterns — arguably more structured information than an SMS scraper collects, because you are sending itemised data. You are trading blanket background access for selective, intentional sharing.

For most users this is a better deal. For privacy absolutists, it isn't — the v2 backlog includes a local LLM option (Ollama + LLaVA) for fully offline, zero-API processing.

---

## Features (v1)

- Screenshot uploader with Gemini Vision OCR — multiple files, cached per session
- Editable form per item: name, amount, platform, category, sub-category, notes
- Sidebar platform checklist — tick off sources as you work through the month
- Sidebar sub-category reference — collapsible, always accessible
- Month selector as the only time dimension
- Manual entry for foreign accounts, cheques, cash
- Data tab: filter by month/category/platform, running total, CSV export
- Graphs: category bar chart, monthly trend, category × month heatmap, configurable custom chart

---

## Stack

- **App:** Streamlit
- **OCR:** Google Gemini Vision (`gemini-2.0-flash`, `google-genai` SDK)
- **Data:** pandas, CSV on disk
- **Charts:** Plotly
- **Config:** python-dotenv

---

## Getting started

```bash
git clone https://github.com/yourusername/receiptiq
cd receiptiq
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

---

## Backlog (v2)

1. **Auto-categorize** — taxonomy + platform rules in Gemini system instruction; model attempts category + sub-category; user corrects exceptions
2. **Month from current date** — auto-derive default month, allow override
3. **Summary tab** — one row per month: category totals, monthly total, cumulative annual, subscription count, cat sub-breakdown
4. **Graph improvements** — grouped bar chart, categories as series, months on X-axis
5. **Google Sheets sync** — write directly to existing sheet structure
6. **Local LLM option** — Ollama + LLaVA for fully offline, zero-API processing

---

## About

Built as a personal tool and portfolio project by a data engineer with a background in genomics and cloud infrastructure (Snowflake, GCP). Uses AI extensively — Gemini Vision for OCR, structured prompting with response schema for consistent field extraction. The goal was a working tool that fits an existing analysis workflow, not a polished demo.
