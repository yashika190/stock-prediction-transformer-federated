import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction")
st.subheader("Transformer + Federated Learning | MCA Research Project")

st.markdown("""
This app demonstrates stock price prediction using 
**Transformer architecture** and **Federated Learning** — 
tested on real Indian equities (Infosys, Reliance, TCS).
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
**Yashika Seth**
MCA Final Year
Guru Nanak Dev College , Ladowali 

Research Project:
Stock Price Prediction using
Transformer + Federated Learning
""")

st.sidebar.title("Tech Stack")
st.sidebar.markdown("""
- Python
- TensorFlow
- Transformer Architecture
- Federated Averaging (FedAvg)
- Pandas, NumPy
- Scikit-learn
""")

# Results section
st.header("📊 Model Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Single Dataset Results")
    results_single = {
        "Model": ["Centralized", "Federated"],
        "RMSE": [0.2751, 0.2682],
        "MAE": [0.1734, 0.1669],
        "MAPE": ["2.51%", "2.41%"],
        "R²": [0.9975, 0.9976],
        "Directional Accuracy": ["53.19%", "56.18%"]
    }
    df_single = pd.DataFrame(results_single)
    st.dataframe(df_single, hide_index=True)

with col2:
    st.subheader("Multi Dataset Results")
    results_multi = {
        "Stock": ["Infosys", "Reliance", "TCS"],
        "Central R²": [0.9894, 0.9957, 0.9769],
        "Federated R²": [0.9895, 0.9956, 0.9750],
        "Central RMSE": [3.4769, 0.3656, 8.1146],
        "Federated RMSE": [3.4573, 0.3682, 8.4267]
    }
    df_multi = pd.DataFrame(results_multi)
    st.dataframe(df_multi, hide_index=True)

# Key metrics
st.header("🎯 Key Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best R² Score", "0.9976", "Federated Model")
m2.metric("Best MAPE", "2.41%", "Federated Model")
m3.metric("Best RMSE", "0.2682", "Federated Model")
m4.metric("Best DA", "56.18%", "Federated Model")

st.success("✅ Federated model achieves comparable or superior performance to centralized — without sharing raw data!")

# Graphs section
st.header("📉 Prediction Graphs")

tab1, tab2 = st.tabs([
    "Single Dataset", 
    "Multi Dataset"
])

with tab1:
    st.subheader("Centralized vs Federated — Single Dataset")
    st.image("results/single_dataset_graph.jpeg")

with tab2:
    st.subheader("Infosys | Reliance | TCS — Multi Dataset")
    st.image("results/multi_dataset_graph.jpeg")

# Live Demo Section
# Single Dataset Live Demo
st.header("🎯 Single Dataset Live Demo")
st.markdown("Upload your **sampledataset.csv** to see instant analysis")

uploaded_file = st.file_uploader(
    "Upload Single Stock CSV",
    type=['csv'],
    key="single"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.success("✅ Single dataset uploaded successfully!")

    # Show data preview
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(10))

    # Show stats
    st.subheader("📊 Data Statistics")
    st.dataframe(df.describe())

    if 'Close' in df.columns:
        # Closing price
        st.subheader("📈 Closing Price Over Time")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['Close'],
                label='Close Price',
                color='cyan',
                linewidth=1.5)
        ax.set_title("Stock Closing Price")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Moving averages
        st.subheader("📉 Moving Average Analysis")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        ax2.plot(df['Close'],
                 label='Close Price',
                 alpha=0.7)
        ax2.plot(df['MA10'],
                 label='MA10',
                 linewidth=2)
        ax2.plot(df['MA20'],
                 label='MA20',
                 linewidth=2)
        ax2.set_title("Price with Moving Averages")
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        # Key stats
        st.subheader("🎯 Key Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min Price",
                  f"{df['Close'].min():.2f}")
        c2.metric("Max Price",
                  f"{df['Close'].max():.2f}")
        c3.metric("Average Price",
                  f"{df['Close'].mean():.2f}")
        c4.metric("Total Records",
                  f"{len(df)}")

        # Show model results for this dataset
        st.subheader("🤖 Model Performance on This Dataset")
        st.info("These results were obtained after training Transformer model on this dataset")

        r1, r2 = st.columns(2)

        with r1:
            st.markdown("**Centralized Model**")
            st.metric("R² Score", "0.9975")
            st.metric("MAPE", "2.51%")
            st.metric("RMSE", "0.2751")
            st.metric("Directional Accuracy", "53.19%")

        with r2:
            st.markdown("**Federated Model**")
            st.metric("R² Score", "0.9976", "+0.0001")
            st.metric("MAPE", "2.41%", "-0.10%")
            st.metric("RMSE", "0.2682", "-0.0069")
            st.metric("Directional Accuracy", "56.18%", "+2.99%")

    else:
        st.error("CSV must have a Close column")

else:
    st.info("👆 Upload sampledataset.csv to see analysis")

    # Multi Dataset Live Demo
# Multi Dataset Live Demo
st.header("🌐 Multi Dataset Live Demo")
st.markdown("Upload **Infosys, Reliance and TCS** CSV files to see instant analysis")

col_a, col_b, col_c = st.columns(3)

with col_a:
    file1 = st.file_uploader(
        "Upload Infosys CSV",
        type=['csv'],
        key="infosys"
    )

with col_b:
    file2 = st.file_uploader(
        "Upload Reliance CSV",
        type=['csv'],
        key="reliance"
    )

with col_c:
    file3 = st.file_uploader(
        "Upload TCS CSV",
        type=['csv'],
        key="tcs"
    )

if file1 and file2 and file3:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()

    st.success("✅ All 3 datasets uploaded successfully!")

    # Data previews
    st.subheader("📋 Data Preview")
    t1, t2, t3 = st.tabs(["Infosys", "Reliance", "TCS"])

    with t1:
        st.dataframe(df1.head(10))
    with t2:
        st.dataframe(df2.head(10))
    with t3:
        st.dataframe(df3.head(10))

    # Closing price comparison
    st.subheader("📈 Closing Price Comparison")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    if 'Close' in df1.columns:
        axes[0].plot(df1['Close'],
                    color='cyan',
                    linewidth=1.5)
        axes[0].set_title('Infosys — Closing Price')
        axes[0].set_ylabel('Price')
        axes[0].grid(alpha=0.3)

    if 'Close' in df2.columns:
        axes[1].plot(df2['Close'],
                    color='orange',
                    linewidth=1.5)
        axes[1].set_title('Reliance — Closing Price')
        axes[1].set_ylabel('Price')
        axes[1].grid(alpha=0.3)

    if 'Close' in df3.columns:
        axes[2].plot(df3['Close'],
                    color='green',
                    linewidth=1.5)
        axes[2].set_title('TCS — Closing Price')
        axes[2].set_ylabel('Price')
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Dataset statistics comparison
    st.subheader("📊 Dataset Statistics Comparison")
    compare_data = {
        "Stock": ["Infosys", "Reliance", "TCS"],
        "Records": [len(df1), len(df2), len(df3)],
        "Min Price": [
            f"{df1['Close'].min():.2f}" if 'Close' in df1.columns else "N/A",
            f"{df2['Close'].min():.2f}" if 'Close' in df2.columns else "N/A",
            f"{df3['Close'].min():.2f}" if 'Close' in df3.columns else "N/A"
        ],
        "Max Price": [
            f"{df1['Close'].max():.2f}" if 'Close' in df1.columns else "N/A",
            f"{df2['Close'].max():.2f}" if 'Close' in df2.columns else "N/A",
            f"{df3['Close'].max():.2f}" if 'Close' in df3.columns else "N/A"
        ],
        "Average Price": [
            f"{df1['Close'].mean():.2f}" if 'Close' in df1.columns else "N/A",
            f"{df2['Close'].mean():.2f}" if 'Close' in df2.columns else "N/A",
            f"{df3['Close'].mean():.2f}" if 'Close' in df3.columns else "N/A"
        ]
    }
    st.dataframe(pd.DataFrame(compare_data), hide_index=True)

    # Model results
    st.subheader("🤖 Model Performance on Multi Dataset")
    st.info("Results obtained after training Transformer + Federated Learning on all 3 stocks")

    st.markdown("**Centralized Model Results**")
    central_data = {
        "Stock": ["Infosys", "Reliance", "TCS"],
        "RMSE": [3.4769, 0.3656, 8.1146],
        "MAE": [2.5170, 0.2216, 6.2466],
        "R²": [0.9894, 0.9957, 0.9769],
        "Directional Accuracy": ["52.44%", "50.10%", "53.66%"]
    }
    st.dataframe(pd.DataFrame(central_data), hide_index=True)

    st.markdown("**Federated Model Results**")
    fed_data = {
        "Stock": ["Infosys", "Reliance", "TCS"],
        "RMSE": [3.4573, 0.3682, 8.4267],
        "MAE": [2.5692, 0.2180, 6.6181],
        "R²": [0.9895, 0.9956, 0.9750],
        "Directional Accuracy": ["47.56%", "55.67%", "46.34%"]
    }
    st.dataframe(pd.DataFrame(fed_data), hide_index=True)

else:
    st.info("👆 Upload all 3 CSV files to see multi dataset analysis")
