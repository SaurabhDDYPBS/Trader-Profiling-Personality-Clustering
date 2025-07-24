import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Trader Personality Predictor", layout="centered")
st.title("📊 Trader Personality Predictor")
st.markdown("Enter **aggregated trading features** of a trader below to predict their personality cluster.")

# Load models
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    kmeans_model = joblib.load('models/best_kmeans_model.pkl')
    return scaler, kmeans_model

scaler, kmeans_model = load_models()

# Form layout
with st.form("input_form"):
    st.subheader("🔢 Trader Feature Inputs")
    avg_pnl = st.number_input("Average PnL (avg_pnl)", 
                              value=500.0, 
                              format="%.2f", 
                              help="Example: 500.0 — Average profit or loss per trade")
    
    pnl_std = st.number_input("PnL Standard Deviation (pnl_std)", 
                              value=1200.0, 
                              format="%.2f", 
                              help="Example: 1200.0 — Higher = more volatile trader")
    
    max_drawdown = st.number_input("Max Drawdown (max_drawdown)", 
                                   value=-7000.0, 
                                   format="%.2f", 
                                   help="Example: -7000.0 — Total losses from negative trades")
    
    win_rate = st.number_input("Win Rate (win_rate)", 
                               min_value=0.0, max_value=1.0, 
                               value=0.55, 
                               format="%.2f", 
                               help="Fraction of profitable trades. Example: 0.55 for 55% win rate")
    
    trade_count = st.number_input("Total Trades (trade_count)", 
                                  min_value=1, 
                                  value=50, 
                                  step=1, 
                                  help="Example: 50 — Number of trades the trader made")
    
    avg_quantity = st.number_input("Average Quantity (avg_quantity)", 
                                   min_value=1.0, 
                                   value=25.0, 
                                   format="%.2f", 
                                   help="Example: 25 — Average quantity per trade")
    
    avg_price = st.number_input("Average Price (avg_price)", 
                                min_value=1.0, 
                                value=2400.0, 
                                format="%.2f", 
                                help="Example: 2400.0 — Average traded price")
    
    total_pnl = st.number_input("Total PnL (total_pnl)", 
                                value=12000.0, 
                                format="%.2f", 
                                help="Example: 12000.0 — Cumulative profit or loss")

    submitted = st.form_submit_button("🔍 Predict Personality")

if submitted:
    # Prepare input array
    input_features = np.array([[avg_pnl, pnl_std, max_drawdown, win_rate,
                                trade_count, avg_quantity, avg_price, total_pnl]])

    # Scale and predict
    X_scaled = scaler.transform(input_features)
    cluster = kmeans_model.predict(X_scaled)[0]

    # Assign personality
    def assign_personality(row):
        if row[3] > 0.6 and row[0] > 1000:
            return "Aggressive"
        elif row[3] < 0.4 and row[1] > 2000:
            return "Erratic"
        elif row[3] > 0.5 and row[1] < 1000:
            return "Conservative"
        else:
            return "Moderate"

    personality = assign_personality(input_features[0])

    # Output
    st.success("Prediction complete!")
    st.subheader("🧠 Result")
    st.markdown(f"**Cluster ID:** `{cluster}`")
    st.markdown(f"**Predicted Personality:** `{personality}`")

    st.info("Note: Personality is based on thresholds defined by average PnL, win rate, and volatility (PnL std dev).")
