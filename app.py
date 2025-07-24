import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Trader Profiling Dashboard", layout="wide")

# Load pre-trained models
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/best_kmeans_model.pkl')
    return scaler, model

scaler, kmeans_model = load_models()

# Personality assignment function
def assign_personality(row):
    if row['win_rate'] > 0.6 and row['avg_pnl'] > 1000:
        return "Aggressive"
    elif row['win_rate'] < 0.4 and row['pnl_std'] > 2000:
        return "Erratic"
    elif row['win_rate'] > 0.5 and row['pnl_std'] < 1000:
        return "Conservative"
    else:
        return "Moderate"

# App Title
st.title("💹 Trader Profiling & Personality Clustering")

st.markdown("""
This dashboard helps identify trader personalities like **Aggressive**, **Conservative**, or **Erratic** 
based on trading metrics, using clustering and PCA.

👉 Choose how you'd like to input trader data:
""")

mode = st.radio("Select Input Mode:", ['🔘 Manual Entry (Single Trader)', '📁 Upload CSV (Batch Profiling)'])

# ---- SINGLE ENTRY ----
if mode == '🔘 Manual Entry (Single Trader)':
    st.subheader("Enter Trader Metrics")
    
    with st.form("manual_form"):
        avg_pnl = st.number_input("Average PnL", value=1000.0, step=100.0, help="E.g., 1200.0")
        pnl_std = st.number_input("PnL Standard Deviation", value=800.0, step=50.0, help="E.g., 500.0")
        max_drawdown = st.number_input("Max Drawdown", value=-3000.0, step=100.0, help="E.g., -2500.0")
        win_rate = st.slider("Win Rate", min_value=0.0, max_value=1.0, value=0.55, help="Fraction between 0 and 1")
        trade_count = st.number_input("Number of Trades", value=50, step=1)
        avg_quantity = st.number_input("Average Quantity", value=25.0, step=1.0)
        avg_price = st.number_input("Average Price", value=2500.0, step=10.0)
        total_pnl = st.number_input("Total PnL", value=15000.0, step=500.0)

        submitted = st.form_submit_button("🔍 Profile Trader")
    
    if submitted:
        input_df = pd.DataFrame([{
            'avg_pnl': avg_pnl,
            'pnl_std': pnl_std,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': trade_count,
            'avg_quantity': avg_quantity,
            'avg_price': avg_price,
            'total_pnl': total_pnl
        }])

        scaled_input = scaler.transform(input_df)
        cluster = kmeans_model.predict(scaled_input)[0]
        personality = assign_personality(input_df.iloc[0])
        
        st.success(f"📌 Trader belongs to **Cluster {cluster}** and is identified as **{personality}**")

        # PCA visualization (project new point)
        pca = PCA(n_components=2)
        X_existing = scaler.transform(pd.read_pickle('data/final_trader_features.pkl')[[
            'avg_pnl', 'pnl_std', 'max_drawdown', 'win_rate',
            'trade_count', 'avg_quantity', 'avg_price', 'total_pnl']])
        X_all = np.vstack([X_existing, scaled_input])
        pca_result = pca.fit_transform(X_all)

        st.subheader("PCA Projection")
        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_result[:-1, 0], y=pca_result[:-1, 1], alpha=0.3, label="Existing Traders")
        plt.scatter(pca_result[-1, 0], pca_result[-1, 1], color='red', s=150, label="Your Input", edgecolor='black')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Projection of Trader Behavior")
        plt.legend()
        st.pyplot(fig)

# ---- BATCH ENTRY ----
else:
    st.subheader("Upload CSV of Trader Metrics")
    st.markdown("""
    Ensure your CSV has the following columns:
    `avg_pnl`, `pnl_std`, `max_drawdown`, `win_rate`, `trade_count`, `avg_quantity`, `avg_price`, `total_pnl`
    """)

    uploaded_file = st.file_uploader("📤 Upload CSV", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['avg_pnl', 'pnl_std', 'max_drawdown', 'win_rate',
                             'trade_count', 'avg_quantity', 'avg_price', 'total_pnl']
            if not all(col in df.columns for col in required_cols):
                st.error("❌ CSV missing required columns.")
            else:
                st.write("📄 Preview of Uploaded Data:")
                st.dataframe(df.head())

                X = scaler.transform(df[required_cols])
                clusters = kmeans_model.predict(X)
                df['cluster'] = clusters
                df['personality'] = df.apply(assign_personality, axis=1)

                # PCA for visual
                pca = PCA(n_components=2)
                components = pca.fit_transform(X)
                df['PC1'], df['PC2'] = components[:, 0], components[:, 1]

                st.subheader("🔍 Clustering & Personality Results")
                st.dataframe(df)

                # Visualize clusters
                st.subheader("🧠 PCA Cluster Visualization")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=df, x='PC1', y='PC2', hue='personality', palette='Set2', s=100, edgecolor='black')
                plt.title("Trader Clusters and Personalities")
                st.pyplot(fig2)

                # Summary stats
                st.subheader("📊 Cluster-wise Summary Stats")
                st.dataframe(df.groupby('cluster')[required_cols].agg(['mean', 'median', 'std']))

                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", data=csv, file_name="clustered_traders.csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
