import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load your data
df = pd.read_csv('your_airbnb_data.csv')  # Replace with your actual file

st.title("🏡 Airbnb Data Analysis Dashboard")

# Descriptive Analysis
st.header("📊 Descriptive Analysis")

st.subheader("Dataset Info")
buffer = []
df.info(buf=buffer)
s = "\n".join(buffer)
st.text(s)

st.subheader("Summary Statistics")
st.write(df.describe())

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 6))
sb.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
st.pyplot(fig)

st.markdown("""
- **Low Correlation Among Variables** – Most correlations are close to zero, indicating weak relationships among features.
- **Price & Service Fee** – Highly correlated, suggesting service fees may be proportional to prices.
- **Availability & Host Listings Count** – Slight positive correlation (~0.15).
- **No Strong Correlation with Ratings** – Suggests ratings depend on subjective factors.
- **Latitude & Longitude** – Minimal correlation with other features.
""")

# Price Distribution
st.subheader("Price Distribution")
fig, ax = plt.subplots(figsize=(12, 5))
sb.set_theme(style='darkgrid')
sb.histplot(df['price'], kde=True, color='r', ax=ax)
ax.set_xlabel('Price', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Distribution of Airbnb Prices', fontsize=15)
st.pyplot(fig)

st.markdown("""
- **Uniform Distribution**: Prices are fairly evenly distributed across ranges.
- **Low and High Prices Present**: Diverse listings for various budgets.
- **Density Peaks**: Mid-range pricing is common.
- **No Extreme Outliers**: Balanced price spread.
""")

# Demographic Analysis
st.header("📍 Demographic Analysis")

# Listings by Neighbourhood Group
st.subheader("Listings Count Per Neighbourhood Group")
fig, ax = plt.subplots(figsize=(12, 5))
top_neighborhoods = df["Neighbourhood_group"].value_counts().head(10)
sb.barplot(x=top_neighborhoods.index, y=top_neighborhoods.values, palette="coolwarm", ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

st.markdown("""
- **Manhattan and Brooklyn** dominate listings.
- **Queens** follows distantly.
- **Staten Island and Bronx** have fewer listings.
- **Skewed Distribution** towards central boroughs.
""")

# Average Price by Neighbourhood Group
st.subheader("Average Price by Neighbourhood Group")
fig, ax = plt.subplots(figsize=(10, 5))
sb.pointplot(x='Neighbourhood_group', y='price', data=df, estimator=np.mean, ax=ax)
ax.set_xlabel('Neighbourhood Group', fontsize=14)
ax.set_ylabel('Average Price', fontsize=14)
ax.set_title('Average Price by Neighbourhood Group', fontsize=15)
st.pyplot(fig)

st.markdown("""
- **Queens has the highest average price**.
- **Staten Island shows high variability**.
- **Brooklyn & Manhattan** have stable pricing.
""")

# Listings per Neighborhood (Pie Chart)
st.subheader("Top 5 Neighborhoods by Number of Listings")
fig, ax = plt.subplots(figsize=(5, 5))
top_neigh = df["neighbourhood"].value_counts().head(5)
ax.pie(top_neigh, labels=top_neigh.index, autopct="%1.1f%%", colors=sb.color_palette("pastel"), startangle=140)
ax.set_title("Top 5 Neighborhoods by Number of Listings")
st.pyplot(fig)

st.markdown("""
- **Bedford-Stuyvesant & Williamsburg** lead with the most listings.
- **Harlem & Bushwick** are also prominent.
- **Hell’s Kitchen** has the fewest among top 5.
""")

# Average Price per Neighborhood
st.subheader("Top 10 Least Expensive Neighborhoods (Min Price)")
fig, ax = plt.subplots(figsize=(12, 5))
avg_price_neigh = df.groupby("neighbourhood")["price"].min().sort_values(ascending=False).head(10)
sb.barplot(x=avg_price_neigh.values, y=avg_price_neigh.index, palette="coolwarm", ax=ax)
ax.set_xlabel("Average Price ($)")
ax.set_ylabel("Neighborhood")
ax.set_title("Top 10 Least Expensive Neighborhoods")
st.pyplot(fig)

# Availability per Neighborhood
st.subheader("Top 10 Neighborhoods with Highest Availability (Days per Year)")
fig, ax = plt.subplots(figsize=(12, 5))
avg_availability = df.groupby("neighbourhood")["availability_in_days"].mean().sort_values(ascending=False).head(10)
sb.barplot(x=avg_availability.values, y=avg_availability.index, palette="magma", ax=ax)
ax.set_xlabel("Average Availability (Days)")
ax.set_ylabel("Neighborhood")
ax.set_title("Highest Availability by Neighborhood")
st.pyplot(fig)

# Neighborhood-wise Ratings
st.subheader("Top 15 Neighborhoods with Highest Ratings")
fig, ax = plt.subplots(figsize=(12, 5))
avg_ratings = df.groupby("neighbourhood")["Ratings"].mean().sort_values(ascending=False).head(15)
sb.barplot(x=avg_ratings.values, y=avg_ratings.index, palette="crest", ax=ax)
ax.set_xlabel("Average Rating")
ax.set_ylabel("Neighborhood")
ax.set_title("Neighborhood Ratings")
st.pyplot(fig)

# Top 10 Listings with Best Ratings
st.subheader("Top 10 Listings with Best Ratings")
top_rated = df.dropna(subset=["Ratings"]).sort_values(by="Ratings", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
sb.barplot(x=top_rated["Name"], y=top_rated["price"], palette="coolwarm", ax=ax)
ax.set_xticklabels(top_rated["Name"], rotation=45, ha='right')
ax.set_xlabel("Listing Name")
ax.set_ylabel("Price ($)")
ax.set_title("Top 10 Listings with Best Ratings")
st.pyplot(fig)

st.dataframe(top_rated[["Name", "price", "Ratings", "Minimum_nights", "availability_in_days", "neighbourhood", "Neighbourhood_group"]])
