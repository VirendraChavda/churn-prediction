import streamlit as st
import pandas as pd
import seaborn as sns
import shap
import numpy as np
import matplotlib.pyplot as plt
import folium
import pickle
from streamlit_folium import folium_static
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# Load Data
#@st.cache_data
def load_data():
    file_path = "Data/new_data.csv"  # Replace with your file path
    df = pd.read_csv(file_path)
    
    # Ensure Total Charges is numeric
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    
    # Handle NaN values in Total Charges
    df['Total Charges'].fillna(0, inplace=True)
    
    return df

#@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

#@st.cache_resource
def xg_load_model():
    with open("xg_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Segmentation Functions
def churn_by_contract_tenure(data):
    # Create Tenure Group
    data['Tenure Group'] = pd.cut(data['Tenure Months'], bins=[0, 12, 36, 60, 72], labels=['0-1 Year', '1-3 Years', '3-5 Years', '5+ Years'])
    
    # Group by Contract and Tenure Group
    grouped = data.groupby(['Contract', 'Tenure Group']).agg({
        'Churn Value': 'mean',  # Average churn rate
        'CustomerID': 'count'  # Count of customers
    }).reset_index()

    # Rename columns for clarity
    grouped.rename(columns={'Churn Value': 'Churn Rate', 'CustomerID': 'Count'}, inplace=True)

    return grouped


def churn_by_service_bundles(data):
    data['Service Bundle'] = data['Internet Service'] + " + " + data['Phone Service']
    grouped = data.groupby('Service Bundle').agg({'Churn Value': 'mean', 'CustomerID': 'count'}).reset_index()
    grouped.rename(columns={'Churn Value': 'Churn Rate', 'CustomerID': 'Count'}, inplace=True)
    return grouped

def payment_method_analysis(data):
    grouped = data.groupby('Payment Method').agg({'Churn Value': 'mean', 'CustomerID': 'count'}).reset_index()
    grouped.rename(columns={'Churn Value': 'Churn Rate', 'CustomerID': 'Count'}, inplace=True)
    return grouped

def tenure_charges_analysis(data):
    grouped = data.groupby(pd.cut(data['Tenure Months'], bins=[0, 12, 36, 60, 72], labels=['0-1 Year', '1-3 Years', '3-5 Years', '5+ Years'])).agg({
        'Monthly Charges': 'mean',
        'Churn Value': 'mean',
        'CustomerID': 'count'
    }).reset_index()
    grouped.rename(columns={'Churn Value': 'Churn Rate', 'Monthly Charges': 'Average Monthly Charges', 'CustomerID': 'Count'}, inplace=True)
    return grouped

def churn_by_reason(data):
    grouped = data.groupby('Churn Reason').agg({'Churn Value': 'mean', 'CustomerID': 'count'}).reset_index()
    grouped.rename(columns={'Churn Value': 'Churn Rate', 'CustomerID': 'Count'}, inplace=True)
    return grouped


def visualize_map(data):
    # Filter for churned customers
    churn_data = data[data['Churn Value'] == 1]
    if churn_data.empty:
        st.info("No churned customer data available for the selected filters.")
        return

    # Use CartoDB positron for a modern, clean tile style
    m = folium.Map(
        location=[churn_data['Latitude'].mean(), churn_data['Longitude'].mean()],
        zoom_start=5.5
    )

    # Add churn markers
    for _, row in churn_data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            popup=f"<b>City:</b> {row['City']}<br><b>State:</b> {row['State']}",
            color="teal",
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # Render the map in Streamlit
    folium_static(m)


def visualize_contract_tenure(segmented_data):
    # Plot grouped data by Tenure Group and Contract
    sns.barplot(data=segmented_data, x='Tenure Group', y='Churn Rate', hue='Contract', palette='magma')
    plt.title("Churn Rate by Contract and Tenure")
    plt.xlabel("Tenure Group")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=0, ha='center')  # Ensure labels are properly aligned
    st.pyplot(plt)


def visualize_service_bundles(segmented_data):
    sns.barplot(data=segmented_data, x='Service Bundle', y='Churn Rate', palette='magma')
    plt.title("Churn Rate by Service Bundles")
    plt.xlabel("Service Bundle")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

def visualize_payment_method(segmented_data):
    sns.barplot(data=segmented_data, x='Payment Method', y='Churn Rate', palette='magma')
    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

def visualize_tenure_charges(segmented_data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=segmented_data, x='Tenure Months', y='Average Monthly Charges', marker='o', color='blue', label='Monthly Charges')
    sns.barplot(data=segmented_data, x='Tenure Months', y='Churn Rate', alpha=0.5, color='orange', label='Churn Rate')
    plt.title("Tenure vs Monthly Charges and Churn Rate")
    plt.xlabel("Tenure Group")
    plt.ylabel("Values")
    plt.legend()
    st.pyplot(plt)

def visualize_top_churn_reasons(data):
    # Filter data for Churn Value == 1 (Yes)
    churned_data = data[data['Churn Value'] == 1]
    
    # Count top 5 reasons for churn
    top_reasons = churned_data['Churn Reason'].value_counts().head(7).reset_index()
    top_reasons.columns = ['Churn Reason', 'Count']
    
    # Plot the chart
    sns.barplot(data=top_reasons, x='Count', y='Churn Reason', palette='magma')
    plt.title("Top 7 Reasons for Churn")
    plt.xlabel("Count")
    plt.ylabel("Churn Reason")
    plt.xticks([], [])
    plt.legend().set_visible(False) 
    st.pyplot(plt)

# Filter Application
def apply_filters(data):
    st.sidebar.header("Interactive Filters")
    
    # City Dropdown
    city = st.sidebar.selectbox("Select City", options=["All"] + list(data['City'].unique()), index=0)

    # Additional Filter Dropdown
    additional_filter = st.sidebar.selectbox("Select Additional Filter", [
        None, 'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Multiple Lines',
        'Internet Service', 'Online Security', 'Online Backup', 'Tech Support',
        'Streaming TV'
    ])

    # Add filter options dynamically
    if additional_filter:
        filter_values = st.sidebar.multiselect(f"Filter by {additional_filter}", options=data[additional_filter].unique(), default=data[additional_filter].unique())
    else:
        filter_values = None

    # Tenure Slider
    tenure_min, tenure_max = st.sidebar.slider("Filter by Tenure (Months)", min_value=int(data['Tenure Months'].min()), max_value=int(data['Tenure Months'].max()), value=(0, 72))

    # # Monthly Charges Slider
    # monthly_min, monthly_max = st.sidebar.slider("Filter by Monthly Charges", min_value=float(data['Monthly Charges'].min()), max_value=float(data['Monthly Charges'].max()), value=(0.0, 120.0))

    # # Total Charges Slider
    # total_min, total_max = st.sidebar.slider("Filter by Total Charges", min_value=float(data['Total Charges'].min()), max_value=float(data['Total Charges'].max()), value=(0.0, float(data['Total Charges'].max())))

    # Customer Lifetime Value
    total_min_cltv, total_max_cltv = st.sidebar.slider("Filter by Customer Lifetime Value", min_value=float(data['CLTV'].min()), max_value=float(data['CLTV'].max()), value=(0.0, float(data['CLTV'].max())))

    # Apply Filters
    filtered_data = data[
        (data['Tenure Months'] >= tenure_min) &
        (data['Tenure Months'] <= tenure_max) &
        # (data['Monthly Charges'] >= monthly_min) &
        # (data['Monthly Charges'] <= monthly_max) &
        # (data['Total Charges'] >= total_min) &
        # (data['Total Charges'] <= total_max) &
        (data['CLTV'] >= total_min_cltv) &
        (data['CLTV'] <= total_max_cltv) 
    ]

    # Apply city filter
    if city != "All":
        filtered_data = filtered_data[filtered_data['City'] == city]

    # Apply additional filter
    if additional_filter and filter_values:
        filtered_data = filtered_data[filtered_data[additional_filter].isin(filter_values)]

    return filtered_data

# Streamlit App Layout
st.set_page_config(page_title="Telco Customer Churn Analysis", page_icon="📊", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Telco Customer Churn Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black;'>By Virendrasinh Chavda.</p>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center;'>
        <a href="https://github.com/your-repo-link" target="_blank" style="color: orange; text-decoration: none;">
            Click here for code.
        </a>
    </p>
""", unsafe_allow_html=True)



# Load Data
df = load_data()
filtered_data = apply_filters(df)
rf_model = load_model()

# Average Churn Score Tile
st.markdown(f"""
    <div style="
        display: flex; 
        justify-content: center; 
        align-items: center; 
        margin: 10px 0;
    ">
        <div style="
            background-color: #f0f8ff; 
            padding: 10px; 
            border-radius: 15px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); 
            width: 420px;
            text-align: center;
        ">
            <h2 style="color: #4CAF50; margin: 0;">📈 Average Churn Score</h2>
            <p style="font-size: 35px; font-weight: bold; margin: 10px 0; color: #333;">
                {filtered_data['Predicted Churn Score'].mean():.2f}
            </p>
        </div>
    </div>
""", unsafe_allow_html=True) 

# Sidebar and Dynamic Segmentation Options (Retained)
st.sidebar.header("Segmentation Options")
segmentation = st.sidebar.selectbox("Choose Segmentation", [
    None,
    "Churn by Contract Type and Tenure",
    "Service Bundles and Churn Rate",
    "Payment Method Analysis",
    "Tenure and Monthly Charges Analysis"
])

# Dynamic Segmentation Visualizations
if segmentation == "Churn by Contract Type and Tenure":
    segmented_data = churn_by_contract_tenure(filtered_data)
    st.subheader("Churn by Contract Type and Tenure")
    visualize_contract_tenure(segmented_data)
    st.write("### Detailed Segmentation Data", segmented_data)

elif segmentation == "Service Bundles and Churn Rate":
    segmented_data = churn_by_service_bundles(filtered_data)
    st.subheader("Service Bundles and Churn Rate")
    visualize_service_bundles(segmented_data)
    st.write("### Detailed Segmentation Data", segmented_data)

elif segmentation == "Payment Method Analysis":
    segmented_data = payment_method_analysis(filtered_data)
    st.subheader("Payment Method Analysis")
    visualize_payment_method(segmented_data)
    st.write("### Detailed Segmentation Data", segmented_data)

elif segmentation == "Tenure and Monthly Charges Analysis":
    segmented_data = tenure_charges_analysis(filtered_data)
    st.subheader("Tenure and Monthly Charges Analysis")
    visualize_tenure_charges(segmented_data)
    st.write("### Detailed Segmentation Data", segmented_data)



# Adding to the Streamlit App Layout
st.subheader("Top 5 Reasons for Churn")
visualize_top_churn_reasons(filtered_data)

# Features resulting in churn
st.subheader("Features importance leading to Churn")
xg_model = xg_load_model()
try:
    # Select a random sample from filtered data for SHAP
    sample_data = filtered_data[["City", "Zip Code", "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months", "Phone Service", "Multiple Lines", "Internet Service",
                                 "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
                                 "Payment Method", "Monthly Charges", "Total Charges", "CLTV"]].sample(n=min(200, len(filtered_data)), random_state=42).reset_index(drop=True)

    #Replace blank entries (' ') with NaN
    sample_data['Total Charges'] = sample_data['Total Charges'].replace(' ', np.nan)
    # Step 2: Convert the column to numeric
    sample_data['Total Charges'] = pd.to_numeric(sample_data['Total Charges'])
    # Step 3: Replace NaN with the median
    median_value = sample_data['Total Charges'].median()
    sample_data['Total Charges'].fillna(median_value, inplace=True)

    cols_one_hot = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines', 'Internet Service','Online Security', 'Online Backup',
                    'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']

    # Define the column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('', OneHotEncoder(), cols_one_hot)
        ],
        remainder='passthrough'  # Leave other columns as-is
    )

    # Apply the transformation
    transformed = column_transformer.fit_transform(sample_data)

    # Get feature names without the 'remainder__' prefix
    new_columns = column_transformer.get_feature_names_out()
    new_columns = [col.replace('remainder__', '') for col in new_columns]
    new_columns = [col.replace('__', '') for col in new_columns]

    # Convert to a dense DataFrame
    sample_data = pd.DataFrame(transformed, columns=new_columns)

    cols_one_hot = ['City', 'Zip Code']
    # Initialize the OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    # Fit and transform the data
    sample_data[cols_one_hot] = ordinal_encoder.fit_transform(sample_data[cols_one_hot])

    transform_cols = ['Tenure Months', 'Monthly Charges',	'Total Charges',	'CLTV']
    # Initialize RobustScaler
    scaler = MinMaxScaler()
    # Fit on X_train and transform X_train
    sample_data[transform_cols] = scaler.fit_transform(sample_data[transform_cols])
    sample_data = sample_data.apply(pd.to_numeric, errors='coerce')

    explainer = shap.TreeExplainer(xg_model)
    shap_values = explainer.shap_values(sample_data)

    # Redirect SHAP summary plot to Streamlit
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, show=False, plot_size=(10, 6))

    # Customize the plot
    plt.title("Top Features Impacting Churn", fontsize=16)  # Custom title
    plt.xlabel("Average Impact on Churn (SHAP value)", fontsize=14)  # Custom x-axis label
    plt.ylabel("Features", fontsize=14)  # Custom y-axis label
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the plot in Streamlit
    st.pyplot(fig)
except:
    st.info("SHAP analysis not available due to missing categories.")

# Map Visualization
st.subheader("Map of Churned Customers")
visualize_map(filtered_data)

# add average churn score
# add shapely