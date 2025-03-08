import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    customers = pd.read_csv('https://raw.githubusercontent.com/RifaldiAchmad/dicoding_task/refs/heads/main/dataset/customers_dataset.csv')
    orders = pd.read_csv('https://raw.githubusercontent.com/RifaldiAchmad/dicoding_task/refs/heads/main/dataset/orders_dataset.csv')
    order_reviews = pd.read_csv('https://raw.githubusercontent.com/RifaldiAchmad/dicoding_task/refs/heads/main/dataset/order_reviews_dataset.csv')
    order_items = pd.read_csv('https://raw.githubusercontent.com/RifaldiAchmad/dicoding_task/refs/heads/main/dataset/order_items_dataset.csv')
    products = pd.read_csv('https://raw.githubusercontent.com/RifaldiAchmad/dicoding_task/refs/heads/main/dataset/products_dataset.csv')
    product_category = pd.read_csv('https://raw.githubusercontent.com/RifaldiAchmad/dicoding_task/refs/heads/main/dataset/product_category_name_translation.csv')

    # Gabungkan data
    all_data = customers.merge(orders, on='customer_id', how='inner') \
        .merge(order_reviews, on='order_id', how='inner') \
        .merge(order_items, on='order_id', how='inner') \
        .merge(products, on='product_id', how='inner') \
        .merge(product_category, on='product_category_name', how='inner')

    # Konversi datetime
    all_data["order_purchase_timestamp"] = pd.to_datetime(all_data["order_purchase_timestamp"])
    all_data["order_delivered_customer_date"] = pd.to_datetime(all_data["order_delivered_customer_date"])
    all_data["order_delivered_carrier_date"] = pd.to_datetime(all_data["order_delivered_carrier_date"])

    return all_data

# Load data
all_data = load_data()

# Sidebar - Filter kategori produk
st.sidebar.header("🔍 Filter Data")
all_categories = all_data["product_category_name_english"].unique().tolist()
selected_categories = st.sidebar.multiselect("Pilih Kategori Produk:", all_categories, default=all_categories[:5])

# Filter data berdasarkan kategori produk
filtered_data = all_data[all_data["product_category_name_english"].isin(selected_categories)]

# 📌 **Analisis RFM**
rfm = filtered_data.groupby("customer_id", as_index=False).agg({
    "order_purchase_timestamp": "max",
    "order_id": "nunique",
    "price": "sum"
})
rfm.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

# Konversi ke datetime
rfm["max_order_timestamp"] = pd.to_datetime(rfm["max_order_timestamp"])

# Hitung recency
recent_date = filtered_data["order_purchase_timestamp"].max()
rfm["recency"] = (recent_date - rfm["max_order_timestamp"]).dt.days

# Normalisasi ranking
rfm["r_rank"] = rfm["recency"].rank(ascending=False)
rfm["f_rank"] = rfm["frequency"].rank(ascending=True)
rfm["m_rank"] = rfm["monetary"].rank(ascending=True)

rfm["r_rank_norm"] = (rfm["r_rank"] / rfm["r_rank"].max()) * 100
rfm["f_rank_norm"] = (rfm["f_rank"] / rfm["f_rank"].max()) * 100
rfm["m_rank_norm"] = (rfm["m_rank"] / rfm["m_rank"].max()) * 100

rfm["RFM_score"] = 0.15 * rfm["r_rank_norm"] + 0.28 * rfm["f_rank_norm"] + 0.57 * rfm["m_rank_norm"]
rfm["RFM_score"] *= 0.05
rfm = rfm.round(2)

# Segmentasi pelanggan
rfm["customer_segment"] = np.where(
    rfm["RFM_score"] > 4.5, "Top", np.where(
        rfm["RFM_score"] > 4, "High value", np.where(
            rfm["RFM_score"] > 3, "Medium value", np.where(
                rfm["RFM_score"] > 1.6, "Low value", "Lost"))))

# Sidebar - Filter segmen pelanggan
all_segments = rfm["customer_segment"].unique().tolist()
selected_segments = st.sidebar.multiselect("Pilih Segmen Pelanggan:", all_segments, default=all_segments)

# Filter data berdasarkan segmen pelanggan
filtered_rfm = rfm[rfm["customer_segment"].isin(selected_segments)]

# **📊 Dashboard**
st.title("📊 Dashboard Analisis E-Commerce")

# 📌 **Grafik 1: Pesanan Per Jam**
st.subheader("📅 Monitoring Waktu Pemesanan")
filtered_data["purchase_hour"] = filtered_data["order_purchase_timestamp"].dt.hour
hourly_orders = filtered_data["purchase_hour"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hourly_orders.index, hourly_orders.values, marker='o', linestyle='-', color='royalblue')
ax.set_xlabel("Jam")
ax.set_ylabel("Jumlah Pesanan")
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# 📌 **Grafik 2: Distribusi Review Score 1 per Kategori**
st.subheader("⭐ Distribusi Review Score 1")
score_counts = filtered_data[filtered_data["review_score"] == 1] \
    .groupby("product_category_name_english").size().reset_index(name="count")

total_counts = filtered_data.groupby("product_category_name_english").size().reset_index(name="total")
merged = pd.merge(total_counts, score_counts, on="product_category_name_english", how="left")
merged["count"] = merged["count"].fillna(0)
merged["percentage"] = (merged["count"] / merged["total"]) * 100

top_10 = merged.sort_values(by="percentage", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(top_10["product_category_name_english"], top_10["percentage"], color="crimson")
ax.set_ylabel("Persentase (%)")
ax.set_xticklabels(top_10["product_category_name_english"], rotation=90)

# Tambahkan label nilai di atas bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{round(yval, 2)}%", ha="center")

st.pyplot(fig)

# 📌 **Grafik 3: Jumlah Pelanggan per Segmen RFM**
st.subheader("📈 Jumlah Pelanggan per Segmen (Analisis RFM)")

segment_counts = filtered_rfm["customer_segment"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(segment_counts.index, segment_counts.values, color=["red", "blue", "green", "gray", "purple"])
ax.set_ylabel("Jumlah Pelanggan")
st.pyplot(fig)

# Tampilkan data RFM setelah difilter
st.dataframe(filtered_rfm[["customer_id", "RFM_score", "customer_segment"]])
