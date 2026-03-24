import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from kneed import KneeLocator

st.set_page_config(layout="wide")

# ============ GLASSMORPHISM CSS ============
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.glass {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
}
h1,h2,h3 {color:white;}
</style>
""", unsafe_allow_html=True)

st.title("🛒 SmartCart AI Customer Segmentation")

# ============ SIDEBAR ============
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset")


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

# ============ DATA PIPELINE ============
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    df["Income"].fillna(df["Income"].median(), inplace=True)
    df["Age"] = 2026 - df["Year_Birth"]
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate",
        "PhD": "Postgraduate"
    })

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

    drop_cols = ["ID","Year_Birth","Marital_Status",
                 "Kidhome","Teenhome","Dt_Customer",
                 "MntWines","MntFruits","MntMeatProducts",
                 "MntFishProducts","MntSweetProducts","MntGoldProds"]

    df = df.drop(columns=drop_cols)

    # Encoding
    ohe = OneHotEncoder()
    cat_cols = ["Education","Living_With"]
    enc = ohe.fit_transform(df[cat_cols]).toarray()

    enc_df = pd.DataFrame(enc,columns=ohe.get_feature_names_out())

    df = pd.concat([df.drop(columns=cat_cols),enc_df],axis=1)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed)

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # ============ AUTO K DETECTION ============
    wcss = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters=i,random_state=42)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)

    knee = KneeLocator(range(1,10),wcss,curve="convex",direction="decreasing")
    optimal_k = knee.elbow if knee.elbow is not None else 4
    optimal_k = int(np.clip(optimal_k, 2, 10))

    k = st.sidebar.slider("Clusters",2,10,optimal_k)

    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X_pca)

    df["Cluster"] = labels

    # ============ KPI CARDS ============
    col1,col2,col3 = st.columns(3)

    col1.metric("Customers",len(df))
    col2.metric("Avg Income",int(df["Income"].mean()))
    col3.metric("Avg Spending",int(df["Total_Spending"].mean()))

    # ============ 3D ANIMATED CLUSTER ============
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

    # ============ BUSINESS GRAPH ============
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

    # ============ BUSINESS RECOMMENDATION ENGINE ============
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

    # ============ CLUSTER DISTRIBUTION ============
    fig3 = px.histogram(df,x="Cluster",title="Cluster Distribution")
    st.plotly_chart(fig3,use_container_width=True)

    # ============ SUMMARY ============
    st.subheader("Cluster Insights")
    st.dataframe(summary)

    # ============ SMART RECOMMENDATION CENTER ============
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

    # ============ CUSTOMER LEVEL RECOMMENDATION ============
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