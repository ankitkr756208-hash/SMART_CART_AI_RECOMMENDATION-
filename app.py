import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from kneed import KneeLocator

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

st.set_page_config(layout="wide")

# SIDEBAR 
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset")
model_choice = st.sidebar.selectbox(
    "Clustering Model",
    ["KMeans", "Agglomerative", "Gaussian Mixture", "Birch"],
    index=0
)
glass_opacity = st.sidebar.slider("Glass Opacity", 0.08, 0.35, 0.16, 0.01)
glass_blur = st.sidebar.slider("Glass Blur", 6, 24, 14, 1)
theme_choice = st.sidebar.selectbox(
    "Visual Theme",
    ["Ocean Neon", "Sunset Copper", "Emerald Night"],
    index=0
)
enable_motion = st.sidebar.toggle("Enable Motion Effects", value=True)
compare_models = st.sidebar.checkbox("Compare All Models", value=True)

THEME_COLORS = {
    "Ocean Neon": {
        "ink": "#f3f6ff",
        "accent": "rgba(38, 134, 255, 0.35)",
        "bg": "radial-gradient(circle at 15% 20%, rgba(84, 173, 255, 0.22), transparent 45%),\n"
              "      radial-gradient(circle at 88% 14%, rgba(255, 168, 112, 0.20), transparent 40%),\n"
              "      radial-gradient(circle at 80% 90%, rgba(125, 255, 217, 0.18), transparent 44%),\n"
              "      linear-gradient(128deg, #071426 0%, #122840 45%, #1c3552 100%)"
    },
    "Sunset Copper": {
        "ink": "#fff6ef",
        "accent": "rgba(255, 163, 82, 0.4)",
        "bg": "radial-gradient(circle at 12% 18%, rgba(255, 193, 131, 0.22), transparent 44%),\n"
              "      radial-gradient(circle at 84% 12%, rgba(255, 115, 78, 0.22), transparent 38%),\n"
              "      radial-gradient(circle at 76% 86%, rgba(255, 208, 152, 0.18), transparent 42%),\n"
              "      linear-gradient(128deg, #2a120d 0%, #5c2719 45%, #7e3f29 100%)"
    },
    "Emerald Night": {
        "ink": "#effff8",
        "accent": "rgba(98, 255, 193, 0.34)",
        "bg": "radial-gradient(circle at 16% 22%, rgba(71, 255, 199, 0.18), transparent 45%),\n"
              "      radial-gradient(circle at 86% 13%, rgba(116, 220, 255, 0.18), transparent 38%),\n"
              "      radial-gradient(circle at 78% 87%, rgba(132, 255, 182, 0.16), transparent 44%),\n"
              "      linear-gradient(128deg, #061b18 0%, #103a33 45%, #185045 100%)"
    }
}
theme = THEME_COLORS[theme_choice]

#  GLASSMORPHISM CSS 
css_template = """
<style>
:root {
    --ink: __INK__;
    --muted: #b7c1da;
    --card-border: rgba(255,255,255,0.22);
    --glass-bg: rgba(255,255,255,__GLASS_OPACITY__);
    --radius: 18px;
}

.stApp {
    color: var(--ink);
    background: __THEME_BG__;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 34px 34px;
    opacity: 0.3;
    z-index: 0;
}

section.main > div {
    position: relative;
    z-index: 1;
}

.block-container {
    padding-top: 1.4rem;
    animation: fadeIn __ANIM_TIME__ms ease-out;
}

[data-testid="stMetric"] {
    background: var(--glass-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    backdrop-filter: blur(__GLASS_BLUR__px);
    -webkit-backdrop-filter: blur(__GLASS_BLUR__px);
    padding: 0.8rem 1rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, opacity 0.25s ease;
    opacity: 0.92;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 14px 35px rgba(8, 16, 34, 0.45);
    opacity: 1;
}

.glass {
    background: var(--glass-bg);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(__GLASS_BLUR__px);
    -webkit-backdrop-filter: blur(__GLASS_BLUR__px);
    padding: 22px;
    border-radius: var(--radius);
    box-shadow: 0 10px 36px rgba(13, 21, 40, 0.36);
    transition: transform 0.25s ease, box-shadow 0.25s ease, opacity 0.25s ease;
    opacity: 0.95;
}

.glass:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 42px rgba(5, 11, 24, 0.52);
    opacity: 1;
}

.segment-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 14px;
}

.segment-badge {
    border: 1px solid rgba(255,255,255,0.24);
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 0.88rem;
    letter-spacing: 0.2px;
    background: rgba(255,255,255,0.08);
    opacity: 0.92;
    transition: all 0.24s ease;
}

.segment-badge:hover {
    transform: translateY(-2px) scale(1.03);
    opacity: 1;
    background: rgba(255,255,255,0.16);
}

h1 {
    color: #f2f7ff;
    letter-spacing: 0.3px;
    text-shadow: 0 2px 16px __ACCENT__;
}

h2, h3, p, label, .stMarkdown, [data-testid="stSidebar"] * {
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: linear-gradient(170deg, rgba(8, 20, 36, 0.92), rgba(12, 34, 58, 0.88));
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    transition: all 0.25s ease;
}

[data-testid="stExpander"]:hover {
    background: rgba(255,255,255,0.1);
    transform: translateY(-2px);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

@media (max-width: 900px) {
    .glass {
        padding: 16px;
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
"""

st.markdown(
    css_template
    .replace("__GLASS_OPACITY__", f"{glass_opacity:.2f}")
    .replace("__GLASS_BLUR__", str(glass_blur))
    .replace("__THEME_BG__", theme["bg"])
    .replace("__INK__", theme["ink"])
    .replace("__ACCENT__", theme["accent"])
    .replace("__ANIM_TIME__", "620" if enable_motion else "1"),
    unsafe_allow_html=True
)

st.title("🛒 SmartCart AI Customer Segmentation")


def cluster_strategy(row, global_income, global_spending):
    """Return a business-friendly segment name and action plan from cluster averages."""
    if row["Total_Spending"] >= global_spending and row["Income"] >= global_income:
        return "High Value", "Premium bundles, loyalty VIP tier, early-access launches"
    if row["Total_Spending"] >= global_spending and row["Income"] < global_income:
        return "Deal Seekers", "Time-limited offers, coupon campaigns, combo discounts"
    if row["Total_Spending"] < global_spending and row["Income"] >= global_income:
        return "Potential Premium", "Upsell via curated recommendations, concierge messaging"
    return "Budget Conscious", "Value packs, cashback nudges, price-sensitive campaigns"


def customer_recommendation(customer_row, q75_spending, q25_spending):
    """Create concise personalized recommendations for a single customer."""
    recs = []
    if customer_row["Total_Spending"] >= q75_spending:
        recs.append("Offer premium membership and new-arrival previews")
    elif customer_row["Total_Spending"] <= q25_spending:
        recs.append("Send discount coupon and low-price bundle suggestions")
    else:
        recs.append("Promote cross-sell products based on recent purchase mix")

    if customer_row["Total_Children"] >= 2:
        recs.append("Highlight family combo packs and school-season campaigns")
    if customer_row["Age"] <= 30:
        recs.append("Use app notifications and social-first campaign creatives")
    elif customer_row["Age"] >= 55:
        recs.append("Use email/SMS campaigns with clear savings messaging")

    return recs


def safe_silhouette_score(X, labels):
    """Return silhouette score only when label/sample constraints are valid."""
    unique_labels = len(np.unique(labels))
    n_samples = X.shape[0]
    if 2 <= unique_labels <= (n_samples - 1):
        return silhouette_score(X, labels)
    return np.nan


def build_cluster_pdf(cluster_profile_df, active_model, active_k, silhouette_value):
    """Generate a concise strategy PDF report in-memory for download."""
    if not REPORTLAB_AVAILABLE:
        return None

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "SmartCart Cluster Strategy Report")

    y -= 24
    c.setFont("Helvetica", 10)
    sil_text = f"{silhouette_value:.3f}" if not np.isnan(silhouette_value) else "N/A"
    c.drawString(40, y, f"Model: {active_model} | Clusters: {active_k} | Silhouette: {sil_text}")

    y -= 24
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Cluster Summary")

    y -= 16
    c.setFont("Helvetica", 9)
    for cluster_id, row in cluster_profile_df.iterrows():
        line = (
            f"Cluster {cluster_id} | {row['Segment']} | Customers: {int(row['Customers'])} | "
            f"Income: {int(row['Income'])} | Spending: {int(row['Total_Spending'])}"
        )
        c.drawString(40, y, line[:118])
        y -= 14

        action_line = f"Action: {row['Recommended_Action']}"
        c.drawString(52, y, action_line[:112])
        y -= 18

        if y <= 70:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

#  DATA PIPELINE 
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "Income" not in df.columns:
        st.error("Uploaded dataset must contain 'Income' column.")
        st.stop()

    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Income"].fillna(df["Income"].median(), inplace=True)

    if "Age" not in df.columns:
        if "Year_Birth" in df.columns:
            df["Age"] = 2026 - pd.to_numeric(df["Year_Birth"], errors="coerce")
        else:
            st.error("Uploaded dataset must contain either 'Age' or 'Year_Birth' column.")
            st.stop()
    else:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    if "Total_Spending" not in df.columns:
        spend_cols = [
            "MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"
        ]
        missing_spend_cols = [c for c in spend_cols if c not in df.columns]
        if missing_spend_cols:
            st.error(
                "Uploaded dataset must contain 'Total_Spending' or spending columns: "
                + ", ".join(spend_cols)
            )
            st.stop()
        df["Total_Spending"] = df[spend_cols].sum(axis=1)

    if "Total_Children" not in df.columns:
        if "Kidhome" in df.columns and "Teenhome" in df.columns:
            df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
        else:
            df["Total_Children"] = 0

    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate"
        })

    if "Living_With" not in df.columns and "Marital_Status" in df.columns:
        df["Living_With"] = df["Marital_Status"].replace({
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone"
        })

    df = df[df["Age"] < 90]
    df = df[df["Income"] < 600000]

    if df.empty:
        st.error("No rows left after filtering. Please upload a valid dataset with realistic Age/Income values.")
        st.stop()

    drop_cols = ["ID", "Year_Birth", "Marital_Status",
                 "Kidhome", "Teenhome", "Dt_Customer",
                 "MntWines", "MntFruits", "MntMeatProducts",
                 "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encoding
    cat_cols = [c for c in ["Education", "Living_With"] if c in df.columns]
    if cat_cols:
        ohe = OneHotEncoder()
        enc = ohe.fit_transform(df[cat_cols]).toarray()
        enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out())
        df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

    # Keep only numeric model inputs; unexpected text values are coerced to NaN.
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df = df.dropna(axis=1, how="all")

    if df.shape[1] == 0:
        st.error("No usable numeric columns found for clustering.")
        st.stop()

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed)

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    n_samples = X_pca.shape[0]
    if n_samples < 2:
        st.error("Need at least 2 rows to perform clustering.")
        st.stop()

    max_clusters = min(10, n_samples)

    #  AUTO K DETECTION 
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i,random_state=42)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)

    knee = KneeLocator(range(1, max_clusters + 1), wcss, curve="convex", direction="decreasing")
    default_k = min(4, max_clusters)
    optimal_k = knee.elbow if knee.elbow is not None else default_k
    optimal_k = int(np.clip(optimal_k, 2, max_clusters))

    k = st.sidebar.slider("Clusters", 2, max_clusters, optimal_k)

    if model_choice == "KMeans":
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_pca)
    elif model_choice == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_pca)
    elif model_choice == "Birch":
        model = Birch(n_clusters=k)
        labels = model.fit_predict(X_pca)
    else:
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X_pca)

    df["Cluster"] = labels
    sil_score = safe_silhouette_score(X_pca, labels)

    #  KPI CARDS 
    col1,col2,col3 = st.columns(3)

    col1.metric("Customers",len(df))
    col2.metric("Avg Income",int(df["Income"].mean()))
    col3.metric("Avg Spending",int(df["Total_Spending"].mean()))

    col4, col5 = st.columns(2)
    col4.metric("Model", model_choice)
    col5.metric("Silhouette", f"{sil_score:.3f}" if not np.isnan(sil_score) else "N/A")

    if compare_models:
        compare_rows = []
        model_configs = {
            "KMeans": KMeans(n_clusters=k, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=k),
            "Birch": Birch(n_clusters=k),
            "Gaussian Mixture": GaussianMixture(n_components=k, random_state=42)
        }

        for name, candidate in model_configs.items():
            temp_labels = candidate.fit_predict(X_pca)
            temp_sil = safe_silhouette_score(X_pca, temp_labels)
            compare_rows.append({
                "Model": name,
                "Silhouette": round(float(temp_sil), 4) if not np.isnan(temp_sil) else np.nan
            })

        compare_df = pd.DataFrame(compare_rows).sort_values("Silhouette", ascending=False, na_position="last")
        medals = ["Gold", "Silver", "Bronze"]
        compare_df["Rank"] = range(1, len(compare_df) + 1)
        compare_df["Medal"] = compare_df["Rank"].apply(
            lambda x: medals[x - 1] if x <= 3 else "Top"
        )
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Model Comparison")
        st.dataframe(compare_df, use_container_width=True)

        chart_df = compare_df.dropna(subset=["Silhouette"]).copy()
        if not chart_df.empty:
            fig_compare = px.bar(
                chart_df,
                x="Model",
                y="Silhouette",
                color="Medal",
                text="Medal",
                title="Model Ranking by Silhouette"
            )
            fig_compare.update_traces(textposition="outside")
            fig_compare.update_layout(showlegend=False)
            st.plotly_chart(fig_compare, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    #  3D ANIMATED CLUSTER 
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    fig = px.scatter_3d(
        x=X_pca[:,0],
        y=X_pca[:,1],
        z=X_pca[:,2],
        color=labels,
        title="3D Customer Segmentation",
        animation_frame=labels
    )

    st.plotly_chart(fig,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    #  BUSINESS GRAPH 
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    plot_df = df.copy()
    for col in ["Income", "Total_Spending", "Age", "Total_Children"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df["Age"] = plot_df["Age"].clip(lower=1)
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    plot_df = plot_df.dropna(subset=["Income", "Total_Spending", "Age", "Total_Children", "Cluster"])

    fig2 = px.scatter(
        plot_df,
        x="Income",
        y="Total_Spending",
        color="Cluster",
        size="Age",
        hover_data=["Total_Children"],
        title="Income vs Spending Behaviour"
    )

    st.plotly_chart(fig2,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # BUSINESS RECOMMENDATION ENGINE 
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📊 AI Business Recommendations")

    summary = df.groupby("Cluster")[["Income","Total_Spending","Total_Children"]].mean()

    for i,row in summary.iterrows():

        if row["Total_Spending"] > df["Total_Spending"].mean():
            st.success(f"Cluster {i}: Premium customers → Target luxury ads")

        elif row["Income"] < df["Income"].mean():
            st.warning(f"Cluster {i}: Low income → Give discounts & offers")

        else:
            st.info(f"Cluster {i}: Moderate → Cross selling recommended")

    st.markdown('</div>', unsafe_allow_html=True)

    #  CLUSTER DISTRIBUTION 
    fig3 = px.histogram(df,x="Cluster",title="Cluster Distribution")
    st.plotly_chart(fig3,use_container_width=True)

    #  SUMMARY 
    st.subheader("Cluster Insights")
    st.dataframe(summary)

    #  SMART RECOMMENDATION CENTER 
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🎯 SmartCart Recommendation Center")

    cluster_profile = (
        df.groupby("Cluster")[["Income", "Total_Spending", "Total_Children", "Age"]]
        .mean()
        .round(2)
    )
    cluster_counts = df["Cluster"].value_counts().sort_index()
    cluster_profile["Customers"] = cluster_counts

    global_income = df["Income"].mean()
    global_spending = df["Total_Spending"].mean()

    segment_labels = []
    action_plans = []
    for _, row in cluster_profile.iterrows():
        label, action = cluster_strategy(row, global_income, global_spending)
        segment_labels.append(label)
        action_plans.append(action)

    cluster_profile["Segment"] = segment_labels
    cluster_profile["Recommended_Action"] = action_plans

    segment_colors = {
        "High Value": "rgba(113, 232, 186, 0.23)",
        "Deal Seekers": "rgba(255, 187, 109, 0.23)",
        "Potential Premium": "rgba(113, 176, 255, 0.23)",
        "Budget Conscious": "rgba(255, 143, 143, 0.23)"
    }

    badge_html = '<div class="segment-row">'
    for segment_name in cluster_profile["Segment"].unique().tolist():
        color = segment_colors.get(segment_name, "rgba(255,255,255,0.16)")
        badge_html += (
            f'<div class="segment-badge" style="background:{color};">'
            f'{segment_name}</div>'
        )
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    st.dataframe(cluster_profile)

    for cluster_id, row in cluster_profile.iterrows():
        with st.expander(f"Cluster {cluster_id} | {row['Segment']}"):
            st.write(f"Customers: {int(row['Customers'])}")
            st.write(f"Avg Income: {int(row['Income'])}")
            st.write(f"Avg Spending: {int(row['Total_Spending'])}")
            st.write(f"Strategy: {row['Recommended_Action']}")

    csv_export = cluster_profile.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Cluster Strategy CSV",
        data=csv_export,
        file_name="smartcart_cluster_recommendations.csv",
        mime="text/csv"
    )

    if REPORTLAB_AVAILABLE:
        pdf_export = build_cluster_pdf(cluster_profile, model_choice, k, sil_score)
        st.download_button(
            "Download Strategy PDF",
            data=pdf_export,
            file_name="smartcart_cluster_strategy_report.pdf",
            mime="application/pdf"
        )
    else:
        st.caption("Install reportlab to enable PDF export: pip install reportlab")

    #  CUSTOMER LEVEL RECOMMENDATION 
    st.subheader("🧠 Customer-Level Recommendation")
    customer_idx = st.selectbox("Select customer index", options=df.index.tolist())
    customer = df.loc[customer_idx]

    q75_spending = df["Total_Spending"].quantile(0.75)
    q25_spending = df["Total_Spending"].quantile(0.25)
    rec_list = customer_recommendation(customer, q75_spending, q25_spending)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", int(customer["Income"]))
    c2.metric("Spending", int(customer["Total_Spending"]))
    c3.metric("Children", int(customer["Total_Children"]))
    c4.metric("Cluster", int(customer["Cluster"]))

    st.write("Recommended actions for this customer:")
    for rec in rec_list:
        st.write(f"- {rec}")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload dataset from sidebar to start")