import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model
with open('Model/mall_customer_segmentation.pkl', 'rb') as file:
    model = pickle.load(file)

# Load clustered data
data = pd.read_csv('Dataset/Mall_Customers_Clustered.csv')

st.title('🛍️ Mall Customer Segmentation')
st.write('Enter customer data below to predict their cluster.')

# Input fields
annual_income = st.text_input('Annual Income (k$) (15-150)')
spending_score = st.text_input('Spending Score (1-100)')

if st.button('Predict Cluster'):
    try:
        # Convert inputs to float
        annual_income = float(annual_income)
        spending_score = float(spending_score)

        # Predict cluster
        features = np.array([[annual_income, spending_score]])
        cluster = model.predict(features)[0]

        st.success(f'✅ The customer belongs to **cluster: {cluster}**')

        # Find customers in this cluster
        cluster_customers = data[data['label'] == cluster]
        count = len(cluster_customers)
        st.info(f"👥 Number of customers in this cluster: **{count}**")

        # Show summary stats
        avg_income = cluster_customers['Annual Income (k$)'].mean()
        avg_spending = cluster_customers['Spending Score (1-100)'].mean()

        st.subheader("📊 Cluster Summary")
        st.write(f"• Average Annual Income: **{avg_income:.2f} k$**")
        st.write(f"• Average Spending Score: **{avg_spending:.2f}**")

        # Show customer IDs as text
        if 'CustomerID' in cluster_customers.columns:
            ids = cluster_customers['CustomerID'].tolist()
            ids_text = ', '.join(map(str, ids))
            st.write("🆔 **Customer IDs in this cluster:**")
            st.write(ids_text)
        else:
            st.warning("⚠️ CustomerID column not found in your CSV. Please check the column name.")

        # 📊 Show charts
        st.subheader("📈 Distribution in this cluster")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].hist(cluster_customers['Annual Income (k$)'], bins=10, color='skyblue', edgecolor='black')
        ax[0].set_title('Annual Income Distribution')
        ax[0].set_xlabel('Annual Income (k$)')
        ax[0].set_ylabel('Count')

        ax[1].hist(cluster_customers['Spending Score (1-100)'], bins=10, color='salmon', edgecolor='black')
        ax[1].set_title('Spending Score Distribution')
        ax[1].set_xlabel('Spending Score')
        ax[1].set_ylabel('Count')

        st.pyplot(fig)

    except ValueError:
        st.error("❗ Please enter valid numeric values for Annual Income and Spending Score.")
