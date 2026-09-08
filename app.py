
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from pathlib import Path
from azure.storage.blob import BlobServiceClient


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Amazon Product Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIG
# ============================================================

CONTAINER_NAME = "amazon-data"
BLOB_NAME = "amazon_clean.csv"

MODEL_DIR = Path("models")

PRICE_STEP = 250


# ============================================================
# HELPERS
# ============================================================

def money(value):
    try:
        if pd.isna(value):
            return "—"

        return f"₹{float(value):,.0f}"

    except Exception:
        return "—"


def hierarchy_parts(category):
    if pd.isna(category):
        return []

    return [
        part.strip()
        for part in str(category).split("|")
        if part.strip()
    ]


def get_level(category, level):
    parts = hierarchy_parts(category)

    if len(parts) > level:
        return parts[level]

    return None


def clamp_price(value):
    """
    Keeps a price inside the valid dataset range
    and aligns it to the nearest ₹250.
    """
    try:
        value = float(value)
    except Exception:
        value = min_price

    value = max(min_price, min(value, max_price))

    value = round(value / PRICE_STEP) * PRICE_STEP

    value = max(min_price, min(value, max_price))

    return int(value)


def normalize_price_range(low, high):
    """
    Makes sure the minimum and maximum prices are valid.
    """
    low = clamp_price(low)
    high = clamp_price(high)

    if low > high:
        high = low

    return low, high


# ============================================================
# LOAD DATA FROM AZURE
# ============================================================

@st.cache_data
def load_data_from_azure():

    connection_string = st.secrets[
        "AZURE_STORAGE_CONNECTION_STRING"
    ]

    blob_service_client = (
        BlobServiceClient
        .from_connection_string(connection_string)
    )

    blob_client = (
        blob_service_client
        .get_blob_client(
            container=CONTAINER_NAME,
            blob=BLOB_NAME
        )
    )

    data = blob_client.download_blob().readall()

    return pd.read_csv(
        BytesIO(data)
    )


# ============================================================
# LOAD EXISTING AI MODELS
# ============================================================

@st.cache_resource
def load_ai_models():

    kmeans_model = joblib.load(
        MODEL_DIR / "kmeans_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    tfidf_vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    cosine_sim = joblib.load(
        MODEL_DIR / "cosine_similarity.pkl"
    )

    recommendation_data = joblib.load(
        MODEL_DIR / "recommendation_data.pkl"
    )

    return {
        "kmeans": kmeans_model,
        "scaler": scaler,
        "tfidf": tfidf_vectorizer,
        "cosine": cosine_sim,
        "recommendation_data": recommendation_data
    }


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data_from_azure()

except Exception as e:

    st.error(
        "Unable to load data from Azure Blob Storage."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# LOAD AI
# ============================================================

try:

    ai_models = load_ai_models()

    ai_available = True

except Exception as e:

    ai_available = False
    ai_error = str(e)
    ai_models = None


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Your dataset is missing these columns:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df = df.copy()


numeric_columns = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df["category"] = (
    df["category"]
    .fillna("Unknown")
    .astype(str)
)


# ============================================================
# CREATE CATEGORY HIERARCHY
# ============================================================

df["main_category"] = df["category"].apply(
    lambda x: get_level(x, 0)
)

df["subcategory"] = df["category"].apply(
    lambda x: get_level(x, 1)
)

df["product_group"] = df["category"].apply(
    lambda x: get_level(x, 2)
)

df["product_type"] = df["category"].apply(
    lambda x: get_level(x, 3)
)


# ============================================================
# PRICE LIMITS
# ============================================================

price_series = pd.to_numeric(
    df["discounted_price"],
    errors="coerce"
).dropna()


if price_series.empty:

    min_price = 0
    max_price = 100000

else:

    raw_min_price = float(
        price_series.min()
    )

    raw_max_price = float(
        price_series.max()
    )

    min_price = int(
        np.floor(
            raw_min_price / PRICE_STEP
        ) * PRICE_STEP
    )

    max_price = int(
        np.ceil(
            raw_max_price / PRICE_STEP
        ) * PRICE_STEP
    )


if min_price == max_price:

    max_price = min_price + PRICE_STEP


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_filters():

    keys_to_reset = [
        "search_keyword",
        "category_keyword",
        "selected_main_category",
        "selected_subcategory",
        "selected_product_group",
        "selected_product_type",
        "price_min_value",
        "price_max_value",
        "price_slider_value",
        "selected_rating"
    ]

    for key in keys_to_reset:

        if key in st.session_state:

            del st.session_state[key]


# ============================================================
# PRICE SYNC FUNCTIONS
# ============================================================

def sync_price_from_slider():

    low, high = st.session_state.price_slider_value

    low, high = normalize_price_range(
        low,
        high
    )

    st.session_state.price_min_value = low
    st.session_state.price_max_value = high

    st.session_state.price_slider_value = (
        low,
        high
    )


def sync_price_from_min():

    low = st.session_state.price_min_value
    high = st.session_state.price_max_value

    low = clamp_price(low)
    high = clamp_price(high)

    if low > high:
        high = low

    st.session_state.price_min_value = low
    st.session_state.price_max_value = high

    st.session_state.price_slider_value = (
        low,
        high
    )


def sync_price_from_max():

    low = st.session_state.price_min_value
    high = st.session_state.price_max_value

    low = clamp_price(low)
    high = clamp_price(high)

    if high < low:
        low = high

    st.session_state.price_min_value = low
    st.session_state.price_max_value = high

    st.session_state.price_slider_value = (
        low,
        high
    )


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "price_min_value" not in st.session_state:

    st.session_state.price_min_value = min_price


if "price_max_value" not in st.session_state:

    st.session_state.price_max_value = max_price


if "price_slider_value" not in st.session_state:

    st.session_state.price_slider_value = (
        min_price,
        max_price
    )


# ============================================================
# SAFE INITIAL PRICE VALUES
# ============================================================

initial_low, initial_high = normalize_price_range(
    st.session_state.price_min_value,
    st.session_state.price_max_value
)

st.session_state.price_min_value = initial_low
st.session_state.price_max_value = initial_high

st.session_state.price_slider_value = (
    initial_low,
    initial_high
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:10px;
        margin-bottom:25px;
    ">
        <div style="
            font-size:46px;
            font-weight:800;
            letter-spacing:-1px;
            line-height:1.1;
        ">
            Amazon Product Intelligence
        </div>
        <div style="
            font-size:15px;
            opacity:0.7;
            margin-top:10px;
        ">
            Azure-powered product search and AI recommendations
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Product Filters"
)


# ============================================================
# RESET BUTTON
# ============================================================

st.sidebar.button(
    "🔄 Reset Filters",
    use_container_width=True,
    on_click=reset_filters
)


st.sidebar.divider()


# ============================================================
# PRODUCT SEARCH
# ============================================================

search_keyword = st.sidebar.text_input(
    "Search Product",
    placeholder="e.g. headphones, keyboard, mouse...",
    key="search_keyword"
)


# ============================================================
# CATEGORY SEARCH
# ============================================================

category_keyword = st.sidebar.text_input(
    "Search Category",
    placeholder="e.g. cables, accessories...",
    key="category_keyword"
)


st.sidebar.divider()


# ============================================================
# CATEGORY HIERARCHY
# ============================================================

st.sidebar.subheader(
    "Category Hierarchy"
)


# ============================================================
# MAIN CATEGORY
# ============================================================

main_categories = sorted(
    df["main_category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_main_category = st.sidebar.selectbox(
    "Main Category",
    ["All"] + main_categories,
    key="selected_main_category"
)


# ============================================================
# SUBCATEGORY OPTIONS
# ============================================================

category_df = df.copy()


if selected_main_category != "All":

    category_df = category_df[
        category_df["main_category"]
        == selected_main_category
    ]


subcategories = sorted(
    category_df["subcategory"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if (
    "selected_subcategory" in st.session_state
    and st.session_state.selected_subcategory
    not in ["All"] + subcategories
):

    st.session_state.selected_subcategory = "All"


selected_subcategory = st.sidebar.selectbox(
    "Subcategory",
    ["All"] + subcategories,
    key="selected_subcategory"
)


# ============================================================
# PRODUCT GROUP OPTIONS
# ============================================================

if selected_subcategory != "All":

    category_df = category_df[
        category_df["subcategory"]
        == selected_subcategory
    ]


product_groups = sorted(
    category_df["product_group"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if (
    "selected_product_group" in st.session_state
    and st.session_state.selected_product_group
    not in ["All"] + product_groups
):

    st.session_state.selected_product_group = "All"


selected_product_group = st.sidebar.selectbox(
    "Product Group",
    ["All"] + product_groups,
    key="selected_product_group"
)


# ============================================================
# PRODUCT TYPE OPTIONS
# ============================================================

if selected_product_group != "All":

    category_df = category_df[
        category_df["product_group"]
        == selected_product_group
    ]


product_types = sorted(
    category_df["product_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if (
    "selected_product_type" in st.session_state
    and st.session_state.selected_product_type
    not in ["All"] + product_types
):

    st.session_state.selected_product_type = "All"


selected_product_type = st.sidebar.selectbox(
    "Product Type",
    ["All"] + product_types,
    key="selected_product_type"
)


st.sidebar.divider()


# ============================================================
# PRICE & RATING
# ============================================================

st.sidebar.subheader(
    "💰 Price & Rating"
)


# ============================================================
# TYPABLE PRICE INPUTS
# ============================================================

price_col1, price_col2 = (
    st.sidebar.columns(2)
)


with price_col1:

    st.number_input(
        "Min ₹",
        min_value=min_price,
        max_value=max_price,
        step=PRICE_STEP,
        key="price_min_value",
        on_change=sync_price_from_min
    )


with price_col2:

    st.number_input(
        "Max ₹",
        min_value=min_price,
        max_value=max_price,
        step=PRICE_STEP,
        key="price_max_value",
        on_change=sync_price_from_max
    )


# ============================================================
# DRAGGABLE PRICE SLIDER
# ============================================================

st.sidebar.slider(
    "Drag Price Range",
    min_value=min_price,
    max_value=max_price,
    step=PRICE_STEP,
    key="price_slider_value",
    on_change=sync_price_from_slider
)


# ============================================================
# FINAL PRICE VALUES
# ============================================================

final_min_price = clamp_price(
    st.session_state.price_min_value
)

final_max_price = clamp_price(
    st.session_state.price_max_value
)


if final_min_price > final_max_price:

    final_max_price = final_min_price


# ============================================================
# RATING
# ============================================================

selected_rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.1,
    key="selected_rating"
)


# ============================================================
# SEARCH BUTTON
# ============================================================

search_clicked = st.sidebar.button(
    "🔍 Search Products",
    type="primary",
    use_container_width=True
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


# ------------------------------------------------------------
# PRODUCT NAME
# ------------------------------------------------------------

if search_keyword.strip():

    filtered_df = filtered_df[
        filtered_df["product_name"]
        .astype(str)
        .str.contains(
            search_keyword.strip(),
            case=False,
            na=False
        )
    ]


# ------------------------------------------------------------
# CATEGORY SEARCH
# ------------------------------------------------------------

if category_keyword.strip():

    filtered_df = filtered_df[
        filtered_df["category"]
        .astype(str)
        .str.contains(
            category_keyword.strip(),
            case=False,
            na=False
        )
    ]


# ------------------------------------------------------------
# MAIN CATEGORY
# ------------------------------------------------------------

if selected_main_category != "All":

    filtered_df = filtered_df[
        filtered_df["main_category"]
        == selected_main_category
    ]


# ------------------------------------------------------------
# SUBCATEGORY
# ------------------------------------------------------------

if selected_subcategory != "All":

    filtered_df = filtered_df[
        filtered_df["subcategory"]
        == selected_subcategory
    ]


# ------------------------------------------------------------
# PRODUCT GROUP
# ------------------------------------------------------------

if selected_product_group != "All":

    filtered_df = filtered_df[
        filtered_df["product_group"]
        == selected_product_group
    ]


# ------------------------------------------------------------
# PRODUCT TYPE
# ------------------------------------------------------------

if selected_product_type != "All":

    filtered_df = filtered_df[
        filtered_df["product_type"]
        == selected_product_type
    ]


# ------------------------------------------------------------
# PRICE
# ------------------------------------------------------------

filtered_df = filtered_df[
    filtered_df["discounted_price"].between(
        final_min_price,
        final_max_price
    )
]


# ------------------------------------------------------------
# RATING
# ------------------------------------------------------------

filtered_df = filtered_df[
    filtered_df["rating"] >= selected_rating
]


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

st.subheader(
    "📊 Dashboard Overview"
)


dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = (
    st.columns(4)
)


with dashboard_col1:

    with st.container(border=True):

        st.metric(
            "Products Found",
            f"{len(filtered_df):,}"
        )

        st.caption(
            "Matching current filters"
        )


with dashboard_col2:

    with st.container(border=True):

        st.metric(
            "Categories",
            f"{df['main_category'].nunique():,}"
        )

        st.caption(
            "Main product categories"
        )


with dashboard_col3:

    with st.container(border=True):

        average_rating = df[
            "rating"
        ].mean()

        st.metric(
            "Average Rating",
            f"{average_rating:.2f} ⭐"
        )

        st.caption(
            "Across all products"
        )


with dashboard_col4:

    with st.container(border=True):

        average_discount = df[
            "discount_percentage"
        ].mean()

        st.metric(
            "Average Discount",
            f"{average_discount:.1f}%"
        )

        st.caption(
            "Average product discount"
        )


st.divider()


# ============================================================
# PRODUCT SEARCH RESULTS
# ============================================================

st.header(
    "🔎 Product Search"
)


if filtered_df.empty:

    st.warning(
        "No products found with the current filters."
    )

else:

    st.write(
        f"Found **{len(filtered_df):,} products**"
    )


    # ========================================================
    # SORT SEARCH RESULTS
    # ========================================================

    sorted_products = (
        filtered_df
        .sort_values(
            by=[
                "rating",
                "rating_count"
            ],
            ascending=[
                False,
                False
            ],
            na_position="last"
        )
    )


    # ========================================================
    # PRODUCT TABLE
    # ========================================================

    display_columns = [
        "product_name",
        "category",
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating",
        "rating_count"
    ]


    display_df = sorted_products[
        display_columns
    ].copy()


    display_df.columns = [
        "Product",
        "Category",
        "Discounted Price",
        "Actual Price",
        "Discount %",
        "Rating",
        "Reviews"
    ]


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # AI RECOMMENDATIONS
    # ========================================================

    st.header(
        "🤖 Top Suggestion by AI"
    )


    if not ai_available:

        st.error(
            "The existing AI model files could not be loaded."
        )

        st.code(
            ai_error
        )

    else:

        # ====================================================
        # GET EXISTING MODEL DATA
        # ====================================================

        recommendation_data = (
            ai_models["recommendation_data"]
            .reset_index(drop=True)
            .copy()
        )

        cosine_sim = ai_models["cosine"]


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if len(cosine_sim) != len(recommendation_data):

            st.error(
                "The saved cosine similarity matrix and "
                "recommendation data do not have matching sizes."
            )

        else:

            # ================================================
            # NORMALIZE PRODUCT IDS
            # ================================================

            recommendation_data["_match_id"] = (
                recommendation_data["product_id"]
                .astype(str)
                .str.strip()
            )


            current_products = filtered_df.copy()


            current_products["_match_id"] = (
                current_products["product_id"]
                .astype(str)
                .str.strip()
            )


            # ================================================
            # ONLY CURRENT SEARCH RESULTS
            # ================================================

            current_ids = set(
                current_products["_match_id"]
            )


            ai_candidates = (
                recommendation_data[
                    recommendation_data["_match_id"]
                    .isin(current_ids)
                ]
                .copy()
            )


            if ai_candidates.empty:

                st.warning(
                    "The existing AI model has no matching "
                    "model records for the products found."
                )

            else:

                # ============================================
                # KEEP ORIGINAL MODEL POSITION
                # ============================================

                ai_candidates["_model_position"] = (
                    ai_candidates.index
                )


                # ============================================
                # ADD CURRENT DATA
                # ============================================

                current_info = current_products[
                    [
                        "_match_id",
                        "about_product",
                        "product_link"
                    ]
                ].copy()


                current_info = (
                    current_info
                    .drop_duplicates(
                        subset=["_match_id"]
                    )
                )


                ai_candidates = ai_candidates.merge(
                    current_info,
                    on="_match_id",
                    how="left",
                    suffixes=("", "_current")
                )


                # ============================================
                # CHOOSE HIGHEST-RATED REFERENCE
                # ============================================

                reference_position = int(
                    ai_candidates
                    .sort_values(
                        by=[
                            "rating",
                            "rating_count"
                        ],
                        ascending=[
                            False,
                            False
                        ],
                        na_position="last"
                    )
                    .iloc[0]["_model_position"]
                )


                # ============================================
                # EXISTING COSINE SIMILARITY
                # ============================================

                def get_similarity(position):

                    try:

                        position = int(position)

                        score = float(
                            cosine_sim[
                                reference_position,
                                position
                            ]
                        )

                        return score

                    except Exception:

                        return 0.0


                ai_candidates[
                    "similarity_score"
                ] = (
                    ai_candidates[
                        "_model_position"
                    ]
                    .apply(get_similarity)
                )


                # ============================================
                # MATCH PERCENTAGE
                # ============================================

                ai_candidates[
                    "match_percentage"
                ] = (
                    ai_candidates[
                        "similarity_score"
                    ]
                    * 100
                ).clip(
                    0,
                    100
                ).round(1)


                # ============================================
                # FINAL RANKING
                # ============================================

                ai_candidates = (
                    ai_candidates
                    .sort_values(
                        by=[
                            "rating",
                            "rating_count",
                            "similarity_score"
                        ],
                        ascending=[
                            False,
                            False,
                            False
                        ],
                        na_position="last"
                    )
                )


                # ============================================
                # MAXIMUM 5
                # ============================================

                ai_candidates = (
                    ai_candidates
                    .head(5)
                )


                st.caption(
                    f"Showing **{len(ai_candidates)}** "
                    f"AI-ranked product(s) from the "
                    f"**{len(filtered_df)}** product(s) "
                    f"found by your search."
                )


                # ============================================
                # DISPLAY AI PRODUCTS
                # ============================================

                for index, product in (
                    ai_candidates
                    .reset_index(drop=True)
                    .iterrows()
                ):

                    rank = index + 1


                    product_name = str(
                        product.get(
                            "product_name",
                            "Unknown Product"
                        )
                    )


                    rating = product.get(
                        "rating",
                        None
                    )


                    rating_count = product.get(
                        "rating_count",
                        None
                    )


                    discounted_price = product.get(
                        "discounted_price",
                        None
                    )


                    actual_price = product.get(
                        "actual_price",
                        None
                    )


                    discount_percentage = product.get(
                        "discount_percentage",
                        None
                    )


                    category = product.get(
                        "category",
                        "Unknown"
                    )


                    match_percentage = product.get(
                        "match_percentage",
                        0
                    )


                    # ========================================
                    # PRODUCT CARD
                    # ========================================

                    with st.container(
                        border=True
                    ):

                        # ====================================
                        # PRODUCT TITLE
                        # ====================================

                        if rank == 1:

                            icon = "🥇"

                        elif rank == 2:

                            icon = "🥈"

                        elif rank == 3:

                            icon = "🥉"

                        else:

                            icon = "🔹"


                        st.markdown(
                            f"""
                            <div style="
                                font-size:22px;
                                font-weight:750;
                                margin-bottom:15px;
                            ">
                                {icon} #{rank} — {product_name}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        # ====================================
                        # TOP 3 HIGHLIGHT
                        # ====================================

                        if rank == 1:

                            st.success(
                                "🏆 Top AI Recommendation"
                            )

                        elif rank == 2:

                            st.info(
                                "⭐ 2nd Best AI Recommendation"
                            )

                        elif rank == 3:

                            st.warning(
                                "⭐ 3rd Best AI Recommendation"
                            )


                        # ====================================
                        # METRICS
                        # ====================================

                        metric1, metric2, metric3, metric4 = (
                            st.columns(4)
                        )


                        with metric1:

                            st.metric(
                                "AI Match",
                                f"{float(match_percentage):.1f}%"
                            )


                        with metric2:

                            st.metric(
                                "Price",
                                money(discounted_price)
                            )


                        with metric3:

                            if pd.notna(rating):

                                st.metric(
                                    "Rating",
                                    f"{float(rating):.1f} ⭐"
                                )

                            else:

                                st.metric(
                                    "Rating",
                                    "—"
                                )


                        with metric4:

                            if pd.notna(
                                discount_percentage
                            ):

                                st.metric(
                                    "Discount",
                                    f"{float(discount_percentage):.0f}%"
                                )

                            else:

                                st.metric(
                                    "Discount",
                                    "—"
                                )


                        # ====================================
                        # PRODUCT DETAILS
                        # ====================================

                        st.write(
                            f"**Category:** "
                            f"{str(category).replace('|', ' → ')}"
                        )


                        if pd.notna(actual_price):

                            st.write(
                                f"**Original Price:** "
                                f"{money(actual_price)}"
                            )


                        if pd.notna(rating_count):

                            st.write(
                                f"**Reviews:** "
                                f"{int(float(rating_count)):,}"
                            )


                        # ====================================
                        # DESCRIPTION
                        # ====================================

                        description = product.get(
                            "about_product",
                            None
                        )


                        if pd.notna(description):

                            st.write(
                                f"**Description:** "
                                f"{description}"
                            )


                        # ====================================
                        # AMAZON LINK
                        # ====================================

                        product_link = product.get(
                            "product_link",
                            None
                        )


                        if (
                            isinstance(product_link, str)
                            and product_link.startswith(
                                (
                                    "http://",
                                    "https://"
                                )
                            )
                        ):

                            link_col1, link_col2, link_col3 = (
                                st.columns([1, 2, 1])
                            )

                            with link_col2:

                                st.link_button(
                                    "🛒 View Amazon Product",
                                    product_link,
                                    use_container_width=True
                                )


# ============================================================
# ABOUT THE AI MODEL
# ============================================================

st.divider()

st.subheader(
    "🧠 About the AI Model"
)

st.markdown(
    """
    ### TF-IDF + Cosine Similarity

    The recommendation system uses **TF-IDF** to convert
    product text into numerical representations.

    **Cosine Similarity** then compares products based on
    how similar their textual information is.

    This allows the system to identify products that are
    closely related to the product selected from the
    current search results.

    ### K-Means Clustering

    **K-Means clustering** groups products with similar
    characteristics together.

    This helps the system understand product segments
    and identify products belonging to similar groups.

    ### How the AI Suggestions Work

    1. Your filters first determine the products shown
       in the current search.

    2. The AI only considers products from those current
       search results.

    3. The existing similarity model calculates how closely
       the products match the highest-rated product in
       your current results.

    4. Products are ranked using rating, review count,
       and AI similarity.

    5. The top 5 results are displayed as AI suggestions,
       or fewer if fewer products match your search.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Amazon Product Intelligence • "
    "Azure Blob Storage • "
    "TF-IDF • Cosine Similarity • "
    "K-Means • Streamlit"
)
