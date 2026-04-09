import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

def create_bycity_df(df):
    bycity_df = df.groupby(by="customer_city").customer_unique_id.nunique().reset_index()
    bycity_df.rename(columns={
        "customer_unique_id": "customer_count"
    }, inplace=True)
    
    return bycity_df

def create_bypayment_df(df):
    bypayment_df=df.groupby(by='payment_type').order_id.nunique().reset_index()
    bypayment_df.rename(columns={
        'order_id':'order_count'
    }, inplace=True)

    return bypayment_df.sort_values(by='order_count', ascending=False)

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    }).reset_index()
    
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    
    return daily_orders_df

def create_product_category_df(df):
    product_df = df.groupby(by="product_category_name").agg({
        "order_id": "nunique",  
        "price": "sum"  
    }).reset_index()
    
    product_df.rename(columns={
        "order_id": "order_volume",
        "price": "revenue"
    }, inplace=True)
    
    return product_df

def create_rfm_df(df):
    rfm_df = df[df['order_status'] == 'delivered'].copy()
    rfm_df = rfm_df[['customer_unique_id', 'order_id', 'order_purchase_timestamp', 'price']]
    latest_date = rfm_df['order_purchase_timestamp'].max()
    rfm = rfm_df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,
        'order_id': 'nunique',                                              
        'price': 'sum'                                                      
    }).reset_index()
    rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']
    
    return rfm


all_df=pd.read_csv('/Users/nurainaz/Documents/analisis data/all_data.csv')
all_df["order_purchase_timestamp"] = pd.to_datetime(all_df["order_purchase_timestamp"])
all_df.sort_values(by='order_purchase_timestamp', inplace=True)
all_df.reset_index(inplace=True)
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.title('Dashboard Analisis E-Commerce')
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

all_state=all_df['customer_state'].dropna().unique()
selected_state=st.sidebar.multiselect(
    label='Choose State',
    options=all_state,
    default=[]
)

df_ecommerce = all_df[(all_df["order_purchase_timestamp"].dt.date >= start_date) & 
                 (all_df["order_purchase_timestamp"].dt.date <= end_date)]

if selected_state:
    df_ecommerce=df_ecommerce[df_ecommerce['customer_state'].isin(selected_state)]

product_df=create_product_category_df(df_ecommerce)
payment_df=create_bypayment_df(df_ecommerce)
city_customer=create_bycity_df(df_ecommerce)
order_harian=create_daily_orders_df(df_ecommerce)
rfm_df = create_rfm_df(df_ecommerce)

st.header('E-Commerce Performance 📊')

st.subheader("Tren Pemesanan Harian")
chart_data = order_harian.set_index('order_purchase_timestamp')['order_count']
st.line_chart(chart_data)

st.subheader('Kota dengan Pelanggan Terbanyak')
top_city=city_customer.sort_values(by="customer_count", ascending=False).head(5)
fig, ax = plt.subplots(figsize=(10, 5))
colors =["#fe1fc9", "#e2dae2", "#e2dae2", "#e2dae2", "#e2dae2"]
sns.barplot(x='customer_count', y='customer_city', data=top_city, ax=ax, palette=colors)
ax.set_title("Top 5 Kota Berdasarkan Jumlah Pelanggan", loc="center", fontsize=15)
ax.set_xlabel("Pelanggan")
ax.set_ylabel('Kota')
st.pyplot(fig)

st.subheader('Top 5 Kategori Produk Terlaris')
col1, col2 = st.columns(2)
with col1:
    st.markdown('**Berdasarkan Volume**')
    top_vol=product_df.sort_values(by='order_volume', ascending=False).head(5)
    fig, ax =plt.subplots(figsize=(8,5))
    sns.barplot(x='order_volume', y='product_category_name', data=top_vol)
    ax.set_xlabel("Volume Pesanan")
    ax.set_ylabel('Kategori Produk')
    st.pyplot(fig)
with col2:
    st.markdown("**Berdasarkan Pendapatan**")
    top_rev=product_df.sort_values(by="revenue", ascending=False).head(5)
    fig, ax =plt.subplots(figsize=(8,5))
    sns.barplot(x="revenue", y="product_category_name", data=top_rev, ax=ax)
    ax.set_xlabel("Pendapatan")
    ax.set_ylabel('Kategori Produk')
    st.pyplot(fig)

st.subheader('Preferensi Metode Pembayaran')

fig, ax = plt.subplots(figsize=(7, 7))
colors = sns.color_palette("bright")

wedges, texts, autotexts = ax.pie(
    x=payment_df['order_count'],
    labels=None,
    autopct='%1.1f%%',
    colors=colors,
    textprops={'fontsize': 12},  
)

for autotext in autotexts:
    autotext.set_fontsize(12)

for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))

    ax.annotate(
        payment_df['payment_type'].iloc[i],
        xy=(x, y),
        xytext=(1.3 * x, 1.3 * y),
        arrowprops=dict(arrowstyle='->'),
        fontsize=11,
        ha='center'
    )

ax.set_title('Metode Pembayaran Paling Populer', fontsize=16)
st.pyplot(fig)

st.markdown("---")
st.subheader("Best Customer Based on RFM Parameters")
rfm_df['short_customer_id'] = rfm_df['customer_unique_id'].apply(lambda x: str(x)[:10] + "...")
colors =["#fe1fc9", "#e2dae2", "#e2dae2", "#e2dae2", "#e2dae2"]
col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_monetary = format_currency(rfm_df.monetary.mean(), "BRL", locale='pt_BR') 
    st.metric("Average Monetary", value=avg_monetary)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35,15))
sns.barplot(y="recency", x="short_customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), 
            palette=colors, ax=ax[0])
ax[0].set_ylabel('Recency')
ax[0].set_xlabel('Customer ID')
ax[0].set_title("By Recency (days)", loc="center", fontsize=18)
ax[0].tick_params(axis ='x', labelsize=10, rotation=45)

sns.barplot(y="frequency", x="short_customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), 
            palette=colors, ax=ax[1])
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Customer ID')
ax[1].set_title("By Frequency", loc="center", fontsize=18)
ax[1].tick_params(axis='x', labelsize=10, rotation=45)

sns.barplot(y="monetary", x="short_customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), 
            palette=colors, ax=ax[2])
ax[2].set_ylabel('Monetary')
ax[2].set_xlabel('Customer ID')
ax[2].set_title("By Monetary", loc="center", fontsize=18)
ax[2].tick_params(axis='x', labelsize=10, rotation=45)

plt.tight_layout()
st.pyplot(fig)