# Trader-Profiling-Personality-Clustering
This project clusters professional and retail traders based on their historical trading behavior using unsupervised machine learning. It assigns trader personalities such as "Aggressive", "Conservative", "Erratic", and "Moderate".

## Features
- Manual input of trade data via a Streamlit web app
- Feature extraction and scaling using saved models
- Clustering with a pretrained KMeans model
- Personality assignment based on cluster features
- Visualization with PCA scatterplots and cluster statistics
- Export clustered results as CSV

## Folder Structure
trader_profiling_project/
├── app.py
├── requirements.txt
├── models/
│ ├── best_kmeans_model.pkl
│ ├── scaler.pkl
│ └── final_trader_features.pkl
├── data/
│ └── trader_logs_api_generated.csv
├── utils/
│ └── init.py
└── README.md




## Setup and Run

1. Clone or download this repo
2. Install dependencies:

```bash
pip install -r requirements.txt



streamlit run app.py

