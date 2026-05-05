# mindful-spend — AI-Powered Expense Tracker
#
# Stack:  Streamlit · Google Gemini Vision API · Pandas · Plotly
# Data:   Local CSV — transparent, portable, pandas-friendly
# Model:  gemini-flash-latest — free tier, strong vision, fast
#
# Design decisions are documented in README.md

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import pandas as pd
import plotly.express as px

load_dotenv()

# ── User config — edit these to match your setup ──────────────────────────────

CURRENCY = "₹"   # change to "$", "€", "CAD $" etc.

CATEGORIES = [
    "", "Big Purchases", "Cat", "Eating Out",
    "Grocery & Home", "Health", "Shopping",
    "Utility", "Gift & Entertainment",
]

# Consistent color per category — used in all charts and badges
CAT_COLORS = {
    "Eating Out":           "#F97316",
    "Grocery & Home":       "#22C55E",
    "Cat":                  "#EAB308",
    "Health":               "#06B6D4",
    "Shopping":             "#A855F7",
    "Utility":              "#64748B",
    "Big Purchases":        "#EF4444",
    "Gift & Entertainment": "#EC4899",
}

DATA_PATH = Path("data/expenses.csv")
COLUMNS   = ["month", "item_name", "amount", "category", "sub_category", "platform", "notes"]

PLATFORM_CHECKLIST = [
    "Swiggy / Zomato",
    "Amazon",
    "Bigbasket / Zepto / Blinkit / Instamart / Fresh",
    "HUFT",
    "Myntra",
    "Urbancompany",
    "Foreign accounts  ← manual entry",
    "Cheque payments   ← manual entry",
    "Credit card 1     ← catch-all",
    "Credit card 2     ← catch-all",
    "Debit / wallet    ← catch-all",
]

SUBCATEGORY_HELP = """
| Category | Sub-categories |
|---|---|
| Big Purchases | Electronics · Tax · Vet · Travel |
| Cat | Dry Food · Wet Food · Snacks · Litter · Cat Toys |
| Eating Out | *(leave blank)* |
| Grocery & Home | Snacks · Healthy · Beverages · Home · Self-care · Stationery · Electronics · Hobby |
| Health | Medicine · Self-care |
| Shopping | Clothes |
| Utility | Phone · Cloud Storage · News · Hobby · Travel |
| Gift & Entertainment | Gift · Show |

**Tips:**
- `Self-care / G&H` = everyday grooming (soap, shampoo). `Self-care / Health` = glasses, salon, prescriptions.
- `Hobby / G&H` = one-off physical purchase (paint kit). `Hobby / Utility` = recurring subscription (Prime).
- `Eating Out` sub-category = always leave blank.
"""

# ── OCR — the core AI function ────────────────────────────────────────────────

def extract_spends(img_bytes: bytes, mime_type: str = "image/jpeg") -> list[dict]:
    """
    Send a payment screenshot to Gemini Vision. Returns a list of spend dicts.

    Key design choices:
    - system_instruction holds all rules; user prompt is just the task.
      This separation makes each easier to iterate independently.
    - response_schema enforces exact field names — no JSON string cleaning needed.
    - Platform rules are baked in to reduce manual categorization.
    - The model handles two screenshot types automatically:
        restaurant order lists → one row per restaurant
        grocery/product screens → one row per line item
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    sys_instruct = (
        "You are a receipt extraction specialist. Extract all spending items from the screenshot. "
        "Rules: "
        "1. Crossed-out prices are original prices — use the active final price only. "
        "2. If quantity > 1, append 'x2', 'x5' etc to item_name; amount = unit price × quantity. "
        "3. Restaurant order lists (Swiggy / Zomato style): one row per restaurant, "
        "   item_name = restaurant name, amount = order total. "
        "4. Grocery / product screens: one row per line item. "
        "5. Platform: extract only if clearly visible in screenshot, else return null. "
        "6. Category: infer using these rules — "
        "   Swiggy or Zomato → 'Eating Out'; "
        "   HUFT → 'Cat'; "
        "   Myntra → 'Shopping'; "
        "   otherwise use item context: "
        "   Big Purchases (major electronics, vet bills, tax, advance travel), "
        "   Cat (pet food and supplies), "
        "   Eating Out (restaurants, cafes, food delivery), "
        "   Grocery & Home (food, household items, stationery), "
        "   Health (medicine, doctor, pharmacy, salon), "
        "   Shopping (clothes, fashion, accessories), "
        "   Utility (phone bill, subscriptions, cloud storage, streaming), "
        "   Gift & Entertainment (gifts, events, shows). "
        "7. sub_category: infer where confident, else return null."
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            "Extract all expenses from this screenshot."
        ],
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            response_mime_type="application/json",
            response_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_name":    {"type": "string"},
                        "amount":       {"type": "number", "description": "Float, 2 decimal places, no currency symbol"},
                        "platform":     {"type": "string"},
                        "category":     {"type": "string"},
                        "sub_category": {"type": "string"},
                    },
                    "required": ["item_name", "amount"]
                }
            }
        )
    )
    return json.loads(response.text)

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        return df
    return pd.DataFrame(columns=COLUMNS)

def save_data(df: pd.DataFrame):
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

def current_month() -> str:
    return datetime.now().strftime("%B %Y")  # e.g. "May 2026"

def month_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month, one column per category. Mirrors the Main sheet structure."""
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="month", columns="category", values="amount",
        aggfunc="sum", fill_value=0
    )
    ordered = [c for c in CATEGORIES if c and c in pivot.columns]
    pivot   = pivot[ordered]
    pivot["MONTHLY"]    = pivot.drop(columns=["Big Purchases"], errors="ignore").sum(axis=1)
    pivot["CUMULATIVE"] = (pivot["MONTHLY"] + pivot.get("Big Purchases", 0)).cumsum()
    return pivot.reset_index()

# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px 16px;
        transition: border-color 0.2s;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(255,255,255,0.16);
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem !important;
        font-weight: 500;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.45) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 22px;
        font-weight: 500;
        font-size: 0.9rem;
        color: rgba(255,255,255,0.5);
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(249,115,22,0.1) !important;
        color: #F97316 !important;
        border-bottom: 2px solid #F97316;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.2);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: #F97316;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.8rem;
        transition: background 0.2s, transform 0.1s;
    }
    .stButton > button[kind="primary"]:hover {
        background: #EA6C0A;
        transform: translateY(-1px);
    }

    /* Form labels */
    .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stMultiSelect label {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(255,255,255,0.4) !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 8px;
    }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.06); margin: 1.8rem 0; }

    /* Captions */
    .stCaption { color: rgba(255,255,255,0.35) !important; font-size: 0.8rem; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* App header */
    h1 { font-weight: 600; letter-spacing: -0.5px; }
    h2 { font-weight: 600; font-size: 1.1rem; color: rgba(255,255,255,0.8); }
    h3 { font-weight: 600; font-size: 0.95rem; color: rgba(255,255,255,0.7); }
    </style>
    """, unsafe_allow_html=True)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="mindful-spend", page_icon="🧾", layout="wide", initial_sidebar_state="expanded")
inject_css()

if "df" not in st.session_state:
    st.session_state.df = load_data()

# Header
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("# 🧾 mindful-spend")
    st.caption("AI-powered expense tracker · Gemini Vision · Streamlit")
with c2:
    st.metric("Today", current_month())

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("#### 📋 Monthly Checklist")
    st.caption("Tick off each source as you pull from it.")
    st.markdown("")
    for p in PLATFORM_CHECKLIST:
        st.checkbox(p, key=f"chk_{p}")
    st.divider()
    with st.expander("📖 Sub-category reference"):
        st.markdown(SUBCATEGORY_HELP)
    st.divider()
    st.caption("Gemini Vision · [GitHub](https://github.com) · Built by OD")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_add, tab_dash, tab_data, tab_graphs = st.tabs(["➕  Add", "📊  Dashboard", "🗂  Data", "📈  Graphs"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ADD
# ══════════════════════════════════════════════════════════════════════════════

with tab_add:

    # Month — auto from current date, but fully editable
    this_month   = current_month()
    saved_months = sorted(st.session_state.df["month"].dropna().unique().tolist()) if not st.session_state.df.empty else []
    month_opts   = sorted(set(saved_months + [this_month]))
    default_idx  = month_opts.index(this_month) if this_month in month_opts else 0

    selected_month = st.selectbox(
        "Month — applies to everything added in this session",
        month_opts, index=default_idx,
    )

    st.divider()

    # ── Screenshots ───────────────────────────────────────────────────────────

    st.markdown("## Upload Screenshots")
    st.caption("Gemini Vision extracts items, amounts, and attempts categorization. Review and save.")

    files = st.file_uploader(
        "Drop screenshots here",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    for f in (files or []):
        # Cache bytes — Streamlit clears the uploader on reruns
        bkey = f"bytes_{f.name}"
        if bkey not in st.session_state:
            st.session_state[bkey] = f.read()
        img_bytes = st.session_state[bkey]

        # OCR — runs once per file, result cached in session state
        okey = f"ocr_{f.name}"
        if okey not in st.session_state:
            with st.spinner(f"Gemini reading {f.name}…"):
                try:
                    st.session_state[okey] = extract_spends(img_bytes, f.type or "image/jpeg")
                except Exception as e:
                    st.session_state[okey] = []
                    st.error(f"OCR failed: {e}")

        items = st.session_state[okey]

        with st.expander(f"**{f.name}** — {len(items)} item{'s' if len(items) != 1 else ''} extracted", expanded=True):
            col_img, col_form = st.columns([1, 2])

            with col_img:
                st.image(img_bytes, use_container_width=True)

            with col_form:
                if not items:
                    st.warning("Nothing extracted. Try a clearer screenshot.")
                    continue

                edited = []
                for i, item in enumerate(items):
                    r1c1, r1c2, r1c3 = st.columns([3, 2, 2])
                    name   = r1c1.text_input("Item", value=item.get("item_name", ""),         key=f"n_{f.name}_{i}")
                    amount = r1c2.number_input(f"Amount ({CURRENCY})", value=float(item.get("amount") or 0), min_value=0.0, step=0.5, format="%.2f", key=f"a_{f.name}_{i}")
                    plat   = r1c3.text_input("Platform", value=item.get("platform") or "",    key=f"p_{f.name}_{i}")

                    r2c1, r2c2, r2c3 = st.columns([2, 2, 3])
                    ocr_cat = item.get("category") or ""
                    cat_idx = CATEGORIES.index(ocr_cat) if ocr_cat in CATEGORIES else 0
                    cat    = r2c1.selectbox("Category", CATEGORIES, index=cat_idx,             key=f"c_{f.name}_{i}")
                    subcat = r2c2.text_input("Sub-category", value=item.get("sub_category") or "", key=f"s_{f.name}_{i}")
                    notes  = r2c3.text_input("Notes",                                          key=f"no_{f.name}_{i}")

                    edited.append({
                        "month": selected_month, "item_name": name,
                        "amount": round(amount, 2), "category": cat,
                        "sub_category": subcat, "platform": plat, "notes": notes,
                    })
                    if i < len(items) - 1:
                        st.divider()

                st.markdown("")
                akey = f"added_{f.name}"
                if st.session_state.get(akey):
                    st.success(f"✓  {len(items)} item{'s' if len(items) != 1 else ''} saved to {selected_month}")
                elif st.button(f"Save {len(items)} item{'s' if len(items) != 1 else ''}", key=f"add_{f.name}", type="primary"):
                    st.session_state.df = pd.concat(
                        [st.session_state.df, pd.DataFrame(edited)], ignore_index=True
                    )
                    save_data(st.session_state.df)
                    st.session_state[akey] = True
                    st.rerun()

    # ── Manual entry ──────────────────────────────────────────────────────────

    st.divider()
    st.markdown("## Manual Entry")
    st.caption("For foreign accounts, cheques, or anything without a screenshot.")

    with st.form("manual", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        m_item   = c1.text_input("Item name")
        m_amount = c2.number_input(f"Amount ({CURRENCY})", min_value=0.0, step=0.5, format="%.2f")
        m_plat   = c3.text_input("Platform / source")
        c4, c5, c6 = st.columns(3)
        m_cat    = c4.selectbox("Category", CATEGORIES)
        m_subcat = c5.text_input("Sub-category")
        m_notes  = c6.text_input("Notes")
        if st.form_submit_button("Add entry", type="primary"):
            if m_item:
                st.session_state.df = pd.concat([
                    st.session_state.df,
                    pd.DataFrame([{
                        "month": selected_month, "item_name": m_item,
                        "amount": round(m_amount, 2), "category": m_cat,
                        "sub_category": m_subcat, "platform": m_plat, "notes": m_notes,
                    }])
                ], ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"Added: {m_item}")
            else:
                st.warning("Item name is required.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    df = st.session_state.df.copy()

    if df.empty:
        st.info("Add some expenses to see your dashboard.")
    else:
        months     = sorted(df["month"].dropna().unique(), reverse=True)
        dash_month = st.selectbox("View month", months, key="dash_month")
        mdf        = df[df["month"] == dash_month]

        st.divider()

        # Top metrics
        total_month  = mdf["amount"].sum()
        excl_big     = mdf[mdf["category"] != "Big Purchases"]["amount"].sum()
        n_orders     = mdf[mdf["category"] == "Eating Out"]["item_name"].nunique()
        cat_spend    = mdf.groupby("category")["amount"].sum()
        top_cat      = cat_spend.idxmax() if not cat_spend.empty else "—"
        top_cat_val  = cat_spend.max()    if not cat_spend.empty else 0
        n_items      = len(mdf)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total spent",          f"{CURRENCY}{total_month:,.0f}")
        m2.metric("Excl. big purchases",  f"{CURRENCY}{excl_big:,.0f}")
        m3.metric("Items tracked",        n_items)
        m4.metric("Eating out orders",    n_orders)
        m5.metric("Biggest category",     top_cat, f"{CURRENCY}{top_cat_val:,.0f}")

        st.divider()

        # Category breakdown + donut
        st.markdown("#### Category breakdown")
        cat_df = (
            mdf.groupby("category")["amount"].sum()
            .reset_index()
            .sort_values("amount", ascending=False)
        )
        cat_df["share"] = (cat_df["amount"] / excl_big * 100).round(1).astype(str) + "%"
        cat_df["amt"]   = cat_df["amount"].apply(lambda x: f"{CURRENCY}{x:,.2f}")

        col_t, col_p = st.columns([1, 1])
        with col_t:
            st.dataframe(
                cat_df[["category", "amt", "share"]].rename(columns={"amt": "amount", "share": "% of monthly"}),
                use_container_width=True, hide_index=True,
            )
        with col_p:
            pie_df = cat_df[cat_df["category"] != "Big Purchases"]
            fig_pie = px.pie(
                pie_df, names="category", values="amount",
                color="category", color_discrete_map=CAT_COLORS,
                hole=0.45,
            )
            fig_pie.update_traces(textposition="outside", textinfo="label+percent")
            fig_pie.update_layout(
                showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Sub-category drilldown
        st.divider()
        st.markdown("#### Category drilldown")
        drill_cat = st.selectbox("Category", [c for c in CATEGORIES if c], key="drill")
        drill_df  = mdf[mdf["category"] == drill_cat]

        if not drill_df.empty:
            sub_agg = (
                drill_df.groupby("sub_category")["amount"].sum()
                .reset_index().sort_values("amount", ascending=True)
            )
            fig_sub = px.bar(
                sub_agg, x="amount", y="sub_category", orientation="h",
                color_discrete_sequence=[CAT_COLORS.get(drill_cat, "#888")],
                labels={"amount": f"Amount ({CURRENCY})", "sub_category": ""},
                text=sub_agg["amount"].apply(lambda x: f"{CURRENCY}{x:,.0f}"),
            )
            fig_sub.update_traces(textposition="outside")
            fig_sub.update_layout(
                margin=dict(t=10, b=10), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            st.caption(f"No {drill_cat} entries for {dash_month}.")

        # Year summary table — mirrors your Main sheet
        st.divider()
        st.markdown("#### Year so far — monthly summary")
        summary = month_summary(df)
        if not summary.empty:
            display = summary.copy()
            for col in [c for c in display.columns if c != "month"]:
                display[col] = display[col].apply(lambda x: f"{CURRENCY}{x:,.0f}" if x else "—")
            st.dataframe(display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:
    df = st.session_state.df.copy()

    if df.empty:
        st.info("No data yet — add expenses in the Add tab.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        months    = sorted(df["month"].dropna().unique(), reverse=True)
        sel_month = fc1.selectbox("Month",    ["All"] + months,                                      key="d_mo")
        sel_cat   = fc2.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique()),    key="d_cat")
        sel_plat  = fc3.selectbox("Platform", ["All"] + sorted(df["platform"].dropna().unique()),    key="d_plat")

        if sel_month != "All": df = df[df["month"]    == sel_month]
        if sel_cat   != "All": df = df[df["category"] == sel_cat]
        if sel_plat  != "All": df = df[df["platform"] == sel_plat]

        st.metric(f"Total  ·  {len(df)} rows", f"{CURRENCY}{df['amount'].sum():,.2f}")
        st.dataframe(df.sort_values("month", ascending=False), use_container_width=True, hide_index=True)

        dl1, dl2 = st.columns(2)
        dl1.download_button("⬇️  Export filtered CSV", df.to_csv(index=False).encode(), "receipts_filtered.csv", "text/csv")
        dl2.download_button("⬇️  Export full CSV",     st.session_state.df.to_csv(index=False).encode(), "receipts_full.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GRAPHS
# ══════════════════════════════════════════════════════════════════════════════

with tab_graphs:
    df_g = st.session_state.df.copy()

    if df_g.empty:
        st.info("No data to plot yet.")
    else:
        all_months = sorted(df_g["month"].dropna().unique())
        sel_months = st.multiselect("Months to include", all_months, default=all_months, key="g_mo")
        if sel_months:
            df_g = df_g[df_g["month"].isin(sel_months)]

        st.divider()

        # Default 1: Category trend — the main useful chart
        st.markdown("#### Category spend by month")
        st.caption("Each bar is a category. Grouped by month. Excludes Big Purchases for readability.")
        trend = (
            df_g[df_g["category"] != "Big Purchases"]
            .groupby(["month", "category"])["amount"].sum()
            .reset_index()
        )
        fig_trend = px.bar(
            trend, x="month", y="amount", color="category",
            barmode="group", color_discrete_map=CAT_COLORS,
            labels={"amount": f"Amount ({CURRENCY})", "month": "", "category": "Category"},
        )
        fig_trend.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Monthly total")
            mon_agg = df_g.groupby("month")["amount"].sum().reset_index()
            fig_mon = px.line(
                mon_agg, x="month", y="amount", markers=True,
                labels={"amount": f"Amount ({CURRENCY})", "month": ""},
            )
            fig_mon.update_traces(line_color="#F97316", line_width=2.5, marker_size=7)
            fig_mon.update_layout(
                margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_mon, use_container_width=True)

        with col2:
            st.markdown("#### Spend by platform")
            plat_agg = (
                df_g.groupby("platform")["amount"].sum()
                .reset_index().sort_values("amount", ascending=True)
                .tail(12)   # top 12 platforms max
            )
            fig_plat = px.bar(
                plat_agg, x="amount", y="platform", orientation="h",
                labels={"amount": f"Amount ({CURRENCY})", "platform": ""},
                color_discrete_sequence=["#22C55E"],
                text=plat_agg["amount"].apply(lambda x: f"{CURRENCY}{x:,.0f}"),
            )
            fig_plat.update_traces(textposition="outside")
            fig_plat.update_layout(
                margin=dict(t=20, b=20), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_plat, use_container_width=True)

        # Custom chart builder
        st.divider()
        st.markdown("#### Custom chart")

        GROUPBY_COLS = ["category", "sub_category", "platform", "month", "item_name"]
        CHART_TYPES  = ["bar", "pie", "line", "scatter", "box"]

        cc1, cc2, cc3, cc4 = st.columns(4)
        g_x     = cc1.selectbox("Group by",   GROUPBY_COLS, key="gx")
        g_color = cc2.selectbox("Color by",   ["None"] + GROUPBY_COLS, key="gc")
        g_chart = cc3.selectbox("Chart type", CHART_TYPES, key="gt")
        g_catf  = cc4.multiselect("Filter category", df_g["category"].dropna().unique(), key="gf")

        df_c      = df_g[df_g["category"].isin(g_catf)].copy() if g_catf else df_g.copy()
        color_arg = None if g_color == "None" else g_color
        agg_cols  = [g_x] + ([g_color] if color_arg else [])
        agg       = df_c.groupby(agg_cols)["amount"].sum().reset_index()

        chart_map = {
            "bar":     lambda: px.bar(    agg,  x=g_x, y="amount", color=color_arg, color_discrete_map=CAT_COLORS),
            "pie":     lambda: px.pie(    agg,  names=g_x, values="amount",         color_discrete_map=CAT_COLORS),
            "line":    lambda: px.line(   agg,  x=g_x, y="amount", color=color_arg, color_discrete_map=CAT_COLORS, markers=True),
            "scatter": lambda: px.scatter(df_c, x=g_x, y="amount", color=color_arg, color_discrete_map=CAT_COLORS),
            "box":     lambda: px.box(    df_c, x=g_x, y="amount", color=color_arg, color_discrete_map=CAT_COLORS),
        }
        fig_custom = chart_map[g_chart]()
        fig_custom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_custom, use_container_width=True)
