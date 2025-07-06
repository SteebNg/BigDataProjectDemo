import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # To handle directory creation
import numpy as np # Ensure numpy is imported
from sklearn.linear_model import LinearRegression

# --- Configuration and Constants ---
DATASET_PATH = 'std_state.csv'
OUTPUT_DIR = 'temp_streamlit_plots/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Refined Disease Categories (YOU MUST VERIFY AND JUSTIFY THESE IN YOUR REPORT!)
# This is a critical research component for your assignment.
DISEASE_CATEGORIES = {
    'chancroid': 'Chancroid',
    'gonorrhea': 'Gonorrhea',
    'hiv': 'Hiv',
    'syphillis': 'Syphillis',
    'aids': 'Aids'
}

# --- 1. Data Loading and Initial Pre-processing ---
@st.cache_data # Cache this function to run only once for efficiency
def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {filepath}. Please ensure '{filepath}' is in the same directory as the app.")
        st.stop() # Stop execution if data loading fails
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}. Please check your dataset format.")
        st.stop() # Stop execution if data loading fails

@st.cache_data # Cache this function as well for efficiency
def preprocess_data(df):
    """
    Performs initial data cleaning and necessary transformations.
    - Converts 'date' column to datetime objects.
    - Adds a 'year' column.
    - Adds 'disease_category' based on predefined mapping.
    - Basic data quality checks.
    """
    df_processed = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Convert 'date' to datetime and extract 'year'
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed['year'] = df_processed['date'].dt.year

    # Add disease categories
    df_processed['disease_category'] = df_processed['disease'].map(DISEASE_CATEGORIES)

    # Handle diseases not found in the mapping
    unmapped_diseases = df_processed[df_processed['disease_category'].isnull()]['disease'].unique()
    if len(unmapped_diseases) > 0:
        st.sidebar.warning(f"Unmapped Diseases: {', '.join(unmapped_diseases)}. Assigned to 'Other/Unspecified'. Please refine mapping.")
        df_processed['disease_category'].fillna('Other/Unspecified', inplace=True)
    
    # Basic Data Quality Checks: Check for non-negative cases and incidence
    if (df_processed['cases'] < 0).any():
        st.sidebar.warning("Negative 'cases' values found and set to 0.")
        df_processed.loc[df_processed['cases'] < 0, 'cases'] = 0
    if (df_processed['incidence'] < 0).any():
        st.sidebar.warning("Negative 'incidence' values found and set to 0.")
        df_processed.loc[df_processed['incidence'] < 0, 'incidence'] = 0

    return df_processed

# --- 2. Data Analysis Functions (Refactored for interactivity) ---
@st.cache_data
def get_yearly_category_trends(df):
    return df.groupby(['year', 'disease_category']).agg(
        total_cases=('cases', 'sum'),
        average_incidence=('incidence', 'mean')
    ).reset_index()

@st.cache_data
def get_state_category_trends(df):
    return df.groupby(['state', 'disease_category']).agg(
        total_cases=('cases', 'sum'),
        average_incidence=('incidence', 'mean')
    ).reset_index()

@st.cache_data
def get_overall_category_summary(df):
    return df.groupby('disease_category').agg(
        total_cases=('cases', 'sum'),
        avg_incidence=('incidence', 'mean')
    ).sort_values(by='total_cases', ascending=False).reset_index()

# --- 3. Visualization Functions ---
def plot_cases_over_time(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='total_cases', hue='disease_category', marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Cases')
    ax.set_xticks(df['year'].unique())
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Disease Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_incidence_over_time(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='average_incidence', hue='disease_category', marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Incidence')
    ax.set_xticks(df['year'].unique())
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Disease Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_overall_cases_by_category(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='disease_category', y='total_cases', palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Disease Category')
    ax.set_ylabel('Total Cases (Sum)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_cases_by_state_for_category(df, category, title):
    filtered_df = df[df['disease_category'] == category].sort_values(by='total_cases', ascending=False)
    if filtered_df.empty:
        return None, f"No data for '{category}' to plot by state."

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=filtered_df, x='state', y='total_cases', palette='plasma', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('State')
    ax.set_ylabel('Total Cases')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig, None

# --- NEW: Machine Learning Functions (Linear Regression Forecasting) ---
@st.cache_resource # Use st.cache_resource for models
def train_linear_regression_model(df_filtered):
    """
    Trains a Linear Regression model for forecasting.
    df_filtered must contain 'year' and 'total_cases'.
    """
    if df_filtered.empty or len(df_filtered) < 2:
        return None, "Not enough data points for linear regression (at least 2 required)."

    X = df_filtered[['year']]
    y = df_filtered['total_cases']

    model = LinearRegression()
    model.fit(X, y)
    return model, None

def make_linear_regression_forecast(model, last_year, years_to_forecast):
    """
    Generates future dates and makes predictions using a trained Linear Regression model.
    """
    future_years = np.array(range(last_year + 1, last_year + 1 + years_to_forecast)).reshape(-1, 1)
    forecasted_cases = model.predict(future_years)
    
    # Create a DataFrame for display
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted Total Cases': forecasted_cases
    })
    return forecast_df

# --- Main Streamlit App Structure ---
def main():
    st.set_page_config(
        page_title="Disease Trends in Malaysia",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Header and Introduction ---
    st.image("INTI_Logo.png")

    st.title("🇲🇾 Healthcare Data Insights: Disease Trends in Malaysia")
    st.markdown("""
    This interactive dashboard, developed for the **5011CEM Big Data Programming Project**,
    provides insights into disease patterns and prevalence in Malaysia.
    Leveraging Big Data principles, it categorizes diseases, analyzes their trends over time (2017-2021),
    and highlights geographical areas with higher disease prevalence.
    """)

    st.markdown("---")

    # --- Sidebar for Navigation and Filters ---
    st.sidebar.header("Navigation & Filters")
    analysis_options = [
        "Dashboard Overview",
        "Disease Category Trends",
        "Geographical Analysis",
        "Predictive Analysis (ML)",
        "Data Explorer",
        "About This Project"
    ]
    selected_analysis = st.sidebar.radio("Go to:", analysis_options)

    # --- Data Loading and Preprocessing ---
    # Perform these steps once and cache the results
    with st.spinner('Loading and pre-processing data...'):
        df_raw = load_data(DATASET_PATH)
        processed_df = preprocess_data(df_raw)
        yearly_category_trends = get_yearly_category_trends(processed_df)
        state_category_trends = get_state_category_trends(processed_df)
        overall_category_summary = get_overall_category_summary(processed_df)
    st.sidebar.success("Data Ready!")

    # --- Dashboard Overview Section ---
    if selected_analysis == "Dashboard Overview":
        st.header("Dashboard Overview: Key Insights at a Glance")
        st.write("A summary of the most critical findings from the disease data.")

        # Display KPIs
        total_cases = processed_df['cases'].sum()
        num_diseases = processed_df['disease'].nunique()
        num_categories = processed_df['disease_category'].nunique()
        num_states = processed_df['state'].nunique()
        reporting_years = processed_df['year'].unique()
        min_year, max_year = min(reporting_years), max(reporting_years)


        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="Total Recorded Cases (2017-2021)", value=f"{total_cases:,}")
        with col2:
            st.metric(label="Unique Diseases Tracked", value=num_diseases)
        with col3:
            st.metric(label="Disease Categories", value=num_categories)
        with col4:
            st.metric(label="States Covered", value=num_states)
        with col5:
            st.metric(label="Analysis Period", value=f"{min_year}-{max_year}")

        st.markdown("---")

        # Top Disease Categories Bar Chart
        st.subheader("Overall Top Disease Categories by Total Cases")
        fig_overall_cases = plot_overall_cases_by_category(overall_category_summary, 'Overall Total Cases by Disease Category')
        st.pyplot(fig_overall_cases)
        st.markdown(f"""
        <p style='font-size: small; text-align: center;'>
        The chart above illustrates the aggregated number of cases for each disease category across all states and years.
        It helps to quickly identify which disease categories have historically had the highest burden in Malaysia.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Cases Over Time for Top Categories (Interactive)
        st.subheader("Disease Cases Over Time")
        st.write("Select disease categories to compare their total cases over the years.")
        unique_categories = sorted(processed_df['disease_category'].unique())
        selected_categories_time = st.multiselect(
            "Select Disease Categories for Time Trend:",
            options=unique_categories,
            default=unique_categories[0] if unique_categories else [] # Default to first category if available
        )

        if selected_categories_time:
            filtered_time_df = yearly_category_trends[yearly_category_trends['disease_category'].isin(selected_categories_time)]
            fig_cases_time = plot_cases_over_time(filtered_time_df, 'Selected Disease Categories: Total Cases Over Time')
            st.pyplot(fig_cases_time)
            st.markdown(f"""
            <p style='font-size: small; text-align: center;'>
            This graph shows the trend of total cases for the selected disease categories over the years.
            Observe if cases are increasing, decreasing, or remaining stable.
            </p>
            """, unsafe_allow_html=True)
        else:
            st.info("Please select at least one disease category to view its trend over time.")

    # --- Disease Category Trends Section ---
    elif selected_analysis == "Disease Category Trends":
        st.header("Detailed Disease Category Trends (2017-2021)")
        st.write("Explore how different disease categories have evolved over the years in terms of cases and incidence.")

        st.subheader("Total Cases by Disease Category Over Time")
        fig_cases = plot_cases_over_time(yearly_category_trends, 'Total Cases by Disease Category Over Time')
        st.pyplot(fig_cases)
        st.markdown(f"""
        <p style='font-size: small; text-align: center;'>
        This line chart visualizes the total number of cases for each disease category
        from 2017 to 2021. It helps identify overall patterns of growth or decline.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("Average Incidence by Disease Category Over Time")
        fig_incidence = plot_incidence_over_time(yearly_category_trends, 'Average Incidence by Disease Category Over Time')
        st.pyplot(fig_incidence)
        st.markdown(f"""
        <p style='font-size: small; text-align: center;'>
        The average incidence rate per category over time. Incidence reflects the rate
        at which new cases occur, providing a view of disease spread relative to population.
        </p>
        """, unsafe_allow_html=True)

    # --- Geographical Analysis Section ---
    elif selected_analysis == "Geographical Analysis":
        st.header("Geographical Analysis: High-Risk States")
        st.write("Identify states with higher total cases or incidence for specific disease categories.")

        unique_categories_geo = sorted(state_category_trends['disease_category'].unique())
        selected_category_geo = st.selectbox(
            "Select Disease Category for State-wise Analysis:",
            options=unique_categories_geo
        )

        if selected_category_geo:
            fig_state_cases, msg = plot_cases_by_state_for_category(state_category_trends, selected_category_geo,
                                                                   f'Total {selected_category_geo} Cases by State')
            if fig_state_cases:
                st.pyplot(fig_state_cases)
                st.markdown(f"""
                <p style='font-size: small; text-align: center;'>
                This bar chart displays the total cases of **{selected_category_geo}** in each Malaysian state.
                States with taller bars indicate higher cumulative cases for this category, suggesting areas for targeted intervention.
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info(msg)

    # --- Predictive Analysis Section ---
    elif selected_analysis == "Predictive Analysis (ML)":
        st.header("Predictive Analysis: Forecast Future Disease Cases")
        st.write("Utilize a Machine Learning model (Linear Regression) to forecast future total cases for a selected disease category.")

        unique_categories_forecast = sorted(yearly_category_trends['disease_category'].unique())
        selected_category_forecast = st.selectbox(
            "Select Disease Category to Forecast:",
            options=unique_categories_forecast
        )

        years_to_forecast = st.slider(
            "Number of Years to Forecast (beyond 2021):",
            min_value=1,
            max_value=5,
            value=2
        )

        if selected_category_forecast:
            st.subheader(f"Forecasting Total Cases for: {selected_category_forecast}")

            df_to_model = yearly_category_trends[
                yearly_category_trends['disease_category'] == selected_category_forecast
            ]
            
            model, error_msg = train_linear_regression_model(df_to_model)

            if error_msg:
                st.warning(error_msg)
            elif model:
                last_year = df_to_model['year'].max()
                forecast_df = make_linear_regression_forecast(model, last_year, years_to_forecast)

                st.success("Forecast Generated Successfully!")

                # Plotting Historical + Forecasted with Linear Regression
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Historical data
                sns.lineplot(data=df_to_model, x='year', y='total_cases', marker='o', label='Historical Data', ax=ax)
                sns.scatterplot(data=df_to_model, x='year', y='total_cases', color='blue', s=100, ax=ax) # Add points
                
                # Linear Regression line for historical period
                # Create a sequence of years from min historical to max forecast
                all_years = np.concatenate((df_to_model['year'].values, forecast_df['Year'].values))
                min_plot_year = min(all_years)
                max_plot_year = max(all_years)
                
                # Generate years for the regression line (including forecast period)
                regression_years = np.array(range(min_plot_year, max_plot_year + 1)).reshape(-1, 1)
                predicted_line_cases = model.predict(regression_years)
                
                ax.plot(regression_years, predicted_line_cases, color='red', linestyle='--', label='Linear Regression Trend')

                # Forecasted points
                sns.scatterplot(data=forecast_df, x='Year', y='Predicted Total Cases', color='green', marker='X', s=150, label='Forecasted Data', ax=ax)
                
                ax.set_title(f'Linear Regression Forecast for {selected_category_forecast} Total Cases')
                ax.set_xlabel('Year')
                ax.set_ylabel('Total Cases')
                ax.set_xticks(np.arange(min_plot_year, max_plot_year + 1, 1)) # Ensure all relevant years are ticks
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(f"""
                <p style='font-size: small; text-align: center;'>
                This chart displays the historical total cases (blue dots) and the forecasted cases (green X marks)
                for **{selected_category_forecast}** using a simple Linear Regression model.
                The red dashed line represents the linear trend.
                </p>
                """, unsafe_allow_html=True)

                st.subheader("Forecasted Data (Next Years)")
                st.dataframe(forecast_df)
            else:
                st.info(f"No historical data found for '{selected_category_forecast}' to perform a forecast.")

    # --- Data Explorer Section ---
    elif selected_analysis == "Data Explorer":
        st.header("Data Explorer")
        st.write("Browse the raw and processed datasets.")

        data_view_option = st.radio(
            "Select Data View:",
            ("Raw Data", "Processed Data")
        )

        if data_view_option == "Raw Data":
            st.subheader("Raw Dataset (`std_state.csv`)")
            st.dataframe(df_raw)
            with st.expander("Show Raw Data Info"):
                buffer = pd.io.common.StringIO()
                df_raw.info(buf=buffer)
                st.text(buffer.getvalue())
        else:
            st.subheader("Processed Dataset (with Categories and Year)")
            st.dataframe(processed_df)
            with st.expander("Show Processed Data Info"):
                buffer = pd.io.common.StringIO()
                processed_df.info(buf=buffer)
                st.text(buffer.getvalue())
            with st.expander("Unique Disease Categories and Counts"):
                st.dataframe(processed_df['disease_category'].value_counts())

    # --- About This Project Section ---
    elif selected_analysis == "About This Project":
        st.header("About This Project")
        st.markdown("""
        This project is part of the **5011CEM Big Data Programming Project** module at INTI International College Penang,
        in collaboration with Coventry University, UK.

        ### Project Objective
        To apply big data analysis and programming techniques to a realistic healthcare scenario in Malaysia,
        specifically focusing on **Disease Category Analysis**.

        ### Key Features
        * **Data Loading & Pre-processing:** Handles raw `.csv` data, converts date formats, and applies custom disease categorization.
        * **Disease Categorization:** Groups similar diseases into broader categories for macro-level analysis.
            *(Note: The current categorization is illustrative and should be thoroughly researched and justified in your assignment report.)*
        * **Trend Analysis:** Visualizes trends of disease cases and incidence rates over time (2017-2021).
        * **Geographical Hotspot Identification:** Identifies states with higher disease burdens for specific categories.
        * **Interactive Dashboard:** Provides a user-friendly interface to explore data and visualizations.

        ### Technologies Used
        * **Python:** The core programming language.
        * **Pandas:** For data manipulation and analysis.
        * **Matplotlib & Seaborn:** For static data visualization.
        * **Streamlit:** For building the interactive web application/dashboard.

        ### Future Enhancements (For Discussion in Report)
        * Integration of demographic data (age, gender) for more granular risk group analysis.
        * Implementation of predictive models for future disease outbreaks.
        * More advanced geospatial visualizations.
        * Connection to larger, real-time healthcare datasets.

        ### Assignment Guidance Reminder
        Remember to detail all aspects of this project in your assignment report, including:
        * Problem Definition & Literature Review
        * Data Analysis (EDA, Machine Learning Algorithms if applied, Algorithm Complexity)
        * Professional Practices (Version control, ethics, etc.)
        * Interpretation of Results
        * Appropriate Diagrams (DFD, ERD, Flowcharts, UML, Gantt chart)
        * A critical reflection on your work during the VIVA.
        """)
        st.markdown("---")
        st.write("Developed for the 5011CEM Big Data Programming Project.")
        st.write(f"Current Date: {pd.to_datetime('today').strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    sns.set_style("whitegrid")
    # Set a larger default font size for plots
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100 # Adjust for better resolution if needed

    main()
