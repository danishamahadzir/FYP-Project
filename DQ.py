import streamlit as st
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import plotly.express as px 
import numpy as np 
from streamlit_option_menu import option_menu

# Set page config at the very top for Streamlit best practice
st.set_page_config(
    page_title="Data Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

#DQ Health Indicator Functions

def get_dq_status(score):
    """Returns the status string based on the score percentage (0-100)."""
    if score is None:
        return "Not Checked"
    elif score >= 90:
        return "Healthy"
    elif score >= 40:
        return "Moderate"
    else:
        return "Critical"

def get_dq_color(score):
    """
    Returns the corresponding color code based on the score percentage (0-100) 
    using the user-defined thresholds: >=90% Green, 40-89.9% Yellow, <40% Red.
    """
    if score is None:
        return "#6c757d" # Gray for N/A
    elif score >= 90:
        return "#28a745" # Green
    elif score >= 40:
        return "#ffc107" # Yellow
    else:
        return "#dc3545" # Red

# --- 1. Database Connection and Schema Management (Unchanged) ---

# Use st.cache_resource to ensure a single database connection is created and reused.
@st.cache_resource
def init_connection():
    try:
        secrets = st.secrets["mysql"]  # make sure name matches your Streamlit secrets section
        conn = mysql.connector.connect(
            host=secrets["host"],
            database=secrets["database"],
            user=secrets["user"],
            password=secrets["password"],
            port=secrets.get("port", 3306)
        )
        return conn
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

def setup_database(conn):
    """
    Creates the main fact table for storing the LATEST data quality scores
    AND the historical log table (for visualization reference/future use).
    """
    if conn is None:
        return
    
    # Existing latest_dq_scores table setup
    table_name_latest = "latest_dq_scores"
    create_table_latest_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name_latest} (
        dataset_name VARCHAR(255) NOT NULL,
        column_name VARCHAR(255) NOT NULL,
        completeness DECIMAL(5,2) NULL,
        uniqueness DECIMAL(5,2) NULL,
        consistency DECIMAL(5,2) NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- Composite Primary Key ensures only one row per (dataset, column)
        PRIMARY KEY (dataset_name, column_name)
    );
    """
    
    # Historical log table setup (for trend charting reference)
    table_name_history = "dq_historical_log"
    create_table_history_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name_history} (
        record_date DATE NOT NULL,
        dataset_name VARCHAR(255) NOT NULL,
        overall_score DECIMAL(5,2) NOT NULL,
        PRIMARY KEY (record_date, dataset_name)
    );
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_latest_query)
        cursor.execute(create_table_history_query)
        conn.commit()
    except Error as e:
        st.error(f"Error setting up database table: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()


def save_score_to_db(column_name, dimension, score, dataset_name):
    """
    Saves or updates the latest score for a specific column/dimension/dataset.
    Uses INSERT...ON DUPLICATE KEY UPDATE (UPSERT).
    """
    conn = init_connection()
    if conn is None:
        return

    table_name = "latest_dq_scores"
    
    # 1. Prepare data and determine which DB column to update
    db_column = dimension.lower() # e.g., 'Completeness' -> 'completeness'
    score_value = score if score != "N/A" else None

    # The INSERT part must provide the primary key columns (dataset_name, column_name)
    # The ON DUPLICATE KEY UPDATE only updates the calculated dimension and the timestamp.
    # We use a placeholder %s for the table name here, which is safer
    update_query = f"""
        INSERT INTO {table_name} 
            (dataset_name, column_name, {db_column}, last_updated)
        VALUES 
            (%s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            {db_column} = VALUES({db_column}),
            last_updated = NOW();
    """

    try:
        cursor = conn.cursor()
        # The query only needs the primary key components and the score value
        cursor.execute(update_query, (dataset_name, column_name, score_value))
        conn.commit()
        st.toast(f"{dimension} score for {column_name} saved ✅", icon="💾")
    except Error as e:
        st.error(f"Database Save Error: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

@st.cache_data(ttl=60) # Cache the fetched data for 60 seconds
def fetch_latest_scores(dataset_name=None):
    """Fetches the latest data quality scores from the database."""
    conn = init_connection()
    if conn is None:
        return pd.DataFrame()
    
    table_name = "latest_dq_scores"
    
    query = f"""
    SELECT 
        column_name AS `Column Name`, 
        completeness AS Completeness, 
        uniqueness AS Uniqueness, 
        consistency AS Consistency, 
        last_updated AS `Last Updated` 
    FROM {table_name}
    """
    
    where_clause = []
    
    if dataset_name:
        # Use simple escaping for security with internal app
        where_clause.append(f"dataset_name = '{dataset_name.replace("'", "''")}'") 
    
    if where_clause:
        query += " WHERE " + " AND ".join(where_clause)
    
    # Order by last updated overall to show most recently touched items first
    query += " ORDER BY last_updated DESC;"

    try:
        # Use pandas read_sql for efficient data fetching
        df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Database Fetch Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_all_datasets():
    """Fetches all distinct dataset names from the database."""
    conn = init_connection()
    if conn is None:
        return []
    
    table_name = "latest_dq_scores"
    query = f"SELECT DISTINCT dataset_name FROM {table_name} ORDER BY dataset_name ASC;"
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = [row[0] for row in cursor.fetchall()]
        return results
    except Error as e:
        return []
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
    
# Global connection object and setup
db_connection = init_connection()
if db_connection:
    setup_database(db_connection)
    
# --- 2. Data Quality Check Functions (Unchanged) ---

def check_completeness(df: pd.DataFrame, column: str) -> float:
    """Calculates the completeness score (percentage of non-null values)."""
    if column not in df.columns: return 0.0
    total_rows = len(df)
    if total_rows == 0: return 0.0
    non_null_count = df[column].count() 
    score = (non_null_count / total_rows) * 100
    return score

def check_completeness_all_columns(df: pd.DataFrame) -> float:
    """Calculates the average completeness score across all columns."""
    if df.empty or len(df.columns) == 0: return 0.0
    total_completeness_sum = 0
    for column in df.columns:
        total_completeness_sum += check_completeness(df, column) 
    
    avg_completeness = total_completeness_sum / len(df.columns)
    return avg_completeness

def check_uniqueness(df: pd.DataFrame, column: str) -> float:
    """Calculates the uniqueness score (percentage of unique values)."""
    if column not in df.columns:
        return 0.0
    total_rows = len(df)
    if total_rows == 0:
        return 0.0
    series = df[column].dropna()
    unique_count = series.nunique()
    score = (unique_count / total_rows) * 100
    return score

def check_consistency(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculates the consistency score by comparing two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return 0.0
    total_rows = len(df)
    if total_rows == 0:
        return 0.0

    # Compare values, treating NaNs as matches if both are NaN
    match_count = (df[col1].fillna('__FILL__') == df[col2].fillna('__FILL__')).sum()
    
    score = (match_count / total_rows) * 100
    return score

# --- 3. Streamlit Page Functions (Unchanged) ---

def home_page():
    st.title("Data Quality Tool")
    st.subheader("Measuring Data Quality Score Against Completeness, Uniqueness & Consistency")
    
    st.markdown("""
    Welcome to the **Data Quality Tool** — an interactive dashboard designed to help you 
    evaluate the overall quality of your datasets before they are used for analysis or reporting. 
    This tool measures your data health across three critical dimensions:
    """)

    # Display a descriptive table
    st.markdown("### Data Quality Dimensions Overview")
    
    data = {
        "No": [1, 2, 3],
        "Dimensions": ["Completeness", "Consistency", "Uniqueness"],
        "Description": [
            "All necessary data attributes are present (no null/missing values).",
            "Data values are uniform across different systems and columns.",
            "Absence of duplicate records, ensuring each row is unique."
        ]
    }
    df_overview = pd.DataFrame(data)
    st.dataframe(df_overview, hide_index=True, use_container_width=True)

    st.markdown("""
    ---
    ### How to Use This Tool
    1. Navigate to the **Data Quality Analyzer** page using the dropdown in the sidebar.
    2. Upload your dataset (Excel or CSV). The file name is used to group the scores in the database.
    3. Choose which data quality dimension to analyze.
    4. View your calculated scores instantly! All scores are **persisted to your MySQL database** for historical reporting.
    5. Navigate to the **Data Quality Visualiser** page to monitor your data quality health and trends.
    """)

def dq_page():
    st.title("📊 Data Quality Analyzer")
    st.subheader("Ensure Your Data is Ready for Action")

    # Custom CSS for aesthetics
    st.markdown("""
    <style>
    /* Target the button inside the file uploader */
    .stFileUploader button {
        background-color: #1E90FF !important; /* Dodger Blue */
        color: white !important; 
        border: 1px solid #1E90FF !important;
        font-weight: bold;
    }
    .stFileUploader button:hover {
        background-color: #1A7BDD !important; /* Darker blue on hover */
        border: 1px solid #1A7BDD !important;
    }
    
    /* Target the border/outline of the dropzone (drag and drop area) */
    div[data-testid="stFileUploaderDropzone"] {
        border-color: #03026f !important; /* Dark Blue */
    }

    /* Target the text/placeholder ("Drag and drop file here") inside the dropzone */
    div[data-testid="stFileUploaderDropzone"] p {
        color: #03026f !important; /* Dark Blue */
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload your Excel or CSV file below to get started!")

    # --- Session state initialization ---
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'scores' not in st.session_state:
        st.session_state.scores = {} 
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # --- File Uploader and Data Loading ---
    uploaded_file = st.file_uploader(
        "Choose an Excel (.xlsx) or CSV (.csv) file", 
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Check if this is a NEW file being uploaded or if the file contents have changed
            if st.session_state.dataset_name != uploaded_file.name:
                # 1. Load Data
                if uploaded_file.name.endswith('.csv'):
                    # Use uploaded_file.getvalue() to prevent file reading issues on rerun
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.session_state.df = df
                st.session_state.dataset_name = uploaded_file.name # Save the file name for DB keying

                st.success(f"File **{st.session_state.dataset_name}** loaded successfully! Total rows: {df.shape[0]}.")
                st.rerun() 

        except Exception as e:
            st.error(f"An error occurred during file processing or calculation: {e}")
            st.session_state.df = None
            st.session_state.dataset_name = None
    
    # --- Main Analysis UI (Only runs if data is in session state) ---
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Display data preview and analysis setup
        with st.expander(f"View Data Preview for: {st.session_state.dataset_name}"):
            st.dataframe(df.head(), use_container_width=True)

        st.markdown("---")
        st.subheader("Configure Analysis")

        all_columns = df.columns.tolist()

        # 2. Dimension selection 
        dimension = st.radio(
            "Select Data Quality Dimension to Check:",
            ('Completeness', 'Uniqueness', 'Consistency'),
            horizontal=True,
            key='dimension_select'
        )

        col1 = None
        col2 = None
        column_name_key = None

        # 3. Column selection 
        if dimension in ['Completeness', 'Uniqueness']:
            options_list = [''] + all_columns
            if dimension == 'Completeness':
                options_list.insert(1, 'All Columns')
                
            col1 = st.selectbox(
                f"Select the column to check for **{dimension}**:",
                options=options_list,
                index=0,
                key='col_select_1'
            )
            column_name_key = col1

        else:  # Consistency
            st.info("For Consistency, you must select two columns to compare.")
            col1 = st.selectbox(
                "Select Column 1:",
                options=[''] + all_columns,
                index=0,
                key='col_select_1'
            )
            col2 = st.selectbox(
                "Select Column 2:",
                options=[''] + all_columns,
                index=0,
                key='col_select_2'
            )
            if col1 and col2:
                column_name_key = f"{col1} vs {col2}"

        # 4. Calculate button & logic — saves to DB + session
        # Ensure that the column has been selected before enabling the calculation
        is_ready_to_calculate = (dimension != 'Consistency' and col1) or \
                                (dimension == 'Consistency' and col1 and col2)

        if st.button(f"Calculate {dimension} Score and Save to DB", disabled=not is_ready_to_calculate):
            score = 0.0
            
            with st.spinner(f"Calculating {dimension} score for {column_name_key}..."):

                # ============ COMPLETENESS ============
                if dimension == "Completeness":
                    if col1 == "All Columns":
                        # Calculate and save scores for ALL columns individually
                        for c in df.columns.tolist():
                            c_score = check_completeness(df, c)
                            save_score_to_db(c, "Completeness", c_score, st.session_state.dataset_name)
                        # Calculate and store the global score for the overall metric box
                        global_score = check_completeness_all_columns(df)
                        st.session_state.scores['Completeness'] = global_score
                        st.session_state.last_result = {"dimension": "Completeness", "column": "All Columns (Avg)", "score": global_score}
                        
                    else:
                        score = check_completeness(df, col1)
                        save_score_to_db(col1, "Completeness", score, st.session_state.dataset_name)
                        st.session_state.scores["Completeness"] = score
                        st.session_state.last_result = {"dimension": "Completeness", "column": col1, "score": score}

                # ============ UNIQUENESS ============
                elif dimension == "Uniqueness":
                    score = check_uniqueness(df, col1)
                    save_score_to_db(col1, "Uniqueness", score, st.session_state.dataset_name)
                    st.session_state.scores["Uniqueness"] = score
                    st.session_state.last_result = {"dimension": "Uniqueness", "column": col1, "score": score}

                # ============ CONSISTENCY ============
                elif dimension == "Consistency":
                    score = check_consistency(df, col1, col2)
                    pair_name = f"{col1} vs {col2}"
                    save_score_to_db(pair_name, "Consistency", score, st.session_state.dataset_name)
                    st.session_state.scores["Consistency"] = score
                    st.session_state.last_result = {"dimension": "Consistency", "column": pair_name, "score": score}

                st.balloons()
                # Force refresh of cached data after saving to ensure the display updates
                fetch_latest_scores.clear() 
                fetch_all_datasets.clear() # Clear the dataset list cache
                st.success(f"✅ {dimension} score saved to database for {column_name_key}!")
                st.rerun() 

        
        # 5. Overall metrics (still based on session state for quick averages)
        if st.session_state.scores:
            st.markdown("---")
            st.subheader(f"Data Quality Results Overview for: {st.session_state.dataset_name}")
            cols = st.columns(4)
            scores_displayed = 0
            total_score = 0
            current_scores = {k: v for k, v in st.session_state.scores.items() if k in ['Completeness', 'Uniqueness', 'Consistency']}
            for i, (dim, score) in enumerate(current_scores.items()):
                # Use the new color function for the metric value
                color = get_dq_color(score)
                cols[i].markdown(f"""
                    <div style="text-align: center;">
                        <p style='font-weight: bold; margin-bottom: 0;'>{dim} Score</p>
                        <h3 style='color: {color}; margin-top: 0;'>{score:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)

                scores_displayed += 1
                total_score += score
            if scores_displayed > 0:
                overall_score = total_score / scores_displayed
                color = get_dq_color(overall_score)
                cols[3].markdown(f"""
                    <div style="text-align: center;">
                        <p style='font-weight: bold; margin-bottom: 0;'>🎯 Overall DQ Score (Avg)</p>
                        <h3 style='color: {color}; margin-top: 0;'>{overall_score:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)

        # 6. Data Quality Scores from DB (The most important part for persistence)
        st.markdown("---")
        st.subheader(f"Latest Persisted Scores (Source: MySQL - Dataset: {st.session_state.dataset_name})")

        # Fetch data directly from the DB
        df_db_scores = fetch_latest_scores(st.session_state.dataset_name)

        if not df_db_scores.empty:
            # Function to apply color based on score (0-100)
            def color_scores_table(val):
                """Applies color to score values (0-100) based on DQ health rules."""
                if pd.isna(val):
                    return '' 
                try:
                    score = float(val)
                except:
                    return ''
                    
                color = get_dq_color(score)
                return f'color: {color}; font-weight: bold'
            
            # Apply the color function to the raw score columns
            styled_df = df_db_scores.style.format({
                'Completeness': "{:.2f}%",
                'Uniqueness': "{:.2f}%",
                'Consistency': "{:.2f}%",
                'Last Updated': lambda t: t.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(t) else 'N/A'
            }).applymap(color_scores_table, subset=['Completeness', 'Uniqueness', 'Consistency'])
            
            st.dataframe(styled_df, hide_index=True, use_container_width=True)

            # Download button
            df_download = df_db_scores.copy()
            df_download.insert(0, 'No', range(1, 1 + len(df_download)))
            csv = df_download.to_csv(index=False).encode('utf-8')
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                f"⬇️ Download Scores for {st.session_state.dataset_name} as CSV",
                data=csv,
                file_name=f"dq_latest_scores_{st.session_state.dataset_name.replace('.', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No scores found for this dataset in the database yet. Run an analysis above!")

        # 7. Last Analysis Summary 
        if st.session_state.last_result:
            last_result = st.session_state.last_result
            st.markdown("---")
            st.subheader("Last Analysis Summary")
            col_display = last_result['column']
            score = last_result['score']
            color = get_dq_color(score)
            status = get_dq_status(score)
            
            with st.container():
                st.markdown(f"""
                    **Dimension Checked:** `{last_result['dimension']}`
                    
                    **Column(s) Analyzed:** `{col_display}`
                    
                    **Calculated Score:** **<span style='font-size: 24px; color: {color};'>{score:.2f}%</span>**
                    
                    **Health Status:** **<span style='font-size: 20px; color: {color};'>{status}</span>**
                """, unsafe_allow_html=True)
    else:
        # Display this if no file is currently loaded in session state
        st.info("Please upload a file to begin the data quality analysis.")


def dq_visualizer_page():
    st.title("Data Quality Visualizer")
    st.subheader("Interactive Health Monitoring Dashboard")
    
    # 1. Automatic Dataset Selection (Point 1)
    selected_dataset = st.session_state.get('dataset_name')
    
    if not selected_dataset:
        st.warning("Please first upload a file and run an analysis on the 'Data Quality Analyzer' page to set the target dataset.")
        return

    st.info(f"Visualizing results for the currently analyzed dataset: **{selected_dataset}**")
    
    # 2. Fetch Latest Scores for Selected Dataset
    df_scores = fetch_latest_scores(selected_dataset)
    
    if df_scores.empty:
        st.warning(f"No scores available for dataset: {selected_dataset}. Run the analyzer page to generate scores.")
        return

    # 3. Calculate Overall Metrics
    
    # Filter to columns that have been calculated at least once for metric averages
    df_calculated = df_scores.dropna(subset=['Completeness', 'Uniqueness', 'Consistency'], how='all')

    # Initialize stats for display. Use None for N/A handling (Point 2)
    overall_score = None
    avg_completeness = None
    avg_uniqueness = None
    avg_consistency = None
    total_columns = df_scores['Column Name'].nunique()
    
    # Try to get total records from the currently loaded file in the Analyzer page
    df = st.session_state.get('df')
    total_data_records = "N/A (Load in Analyzer)"
    
    if df is not None and st.session_state.dataset_name == selected_dataset:
        # 4. Total Data Records = Rows x Columns (Point 4)
        total_data_records_calc = df.shape[0] * len(df.columns)
        total_data_records = f"{total_data_records_calc:,}"
    
    if not df_calculated.empty:
        # Calculate Averages, setting to None if no calculated data exists for that dimension (Point 2)
        # Check if the column has ANY non-NaN values before calculating mean
        avg_completeness = df_calculated['Completeness'].mean() if df_calculated['Completeness'].dropna().any() else None
        avg_uniqueness = df_calculated['Uniqueness'].mean() if df_calculated['Uniqueness'].dropna().any() else None
        avg_consistency = df_calculated['Consistency'].mean() if df_calculated['Consistency'].dropna().any() else None
        
        # Calculate overall score based only on dimensions that have calculated averages (not None)
        scores_list = [s for s in [avg_completeness, avg_uniqueness, avg_consistency] if s is not None and s > 0]
        overall_score = sum(scores_list) / len(scores_list) if scores_list else None
    
    st.markdown("---")
    
    # 4. KPI Metrics (Overall Score, Completeness, Uniqueness, Consistency)
    
    kpi_cols = st.columns(4)
    
    def display_kpi(col_index, label, score):
        # Determine display format for score (Point 2)
        if score is None:
            display_value = "N/A"
        else:
            display_value = f"{score:.2f}%"
            
        color = get_dq_color(score)
        status = get_dq_status(score)
        
        # Using markdown with inline CSS for coloring the title, score, and border
        kpi_cols[col_index].markdown(f"""
            <div style="
                border: 2px solid {color}; 
                padding: 10px; 
                border-radius: 8px; 
                text-align: center; 
                background-color: #f8f9fa;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                min-height: 120px;
            ">
                <p style='font-size: 16px; margin: 0; color: #555;'>{label}</p>
                <h2 style='color: {color}; margin: 5px 0;'>{display_value}</h2>
                <p style='font-weight: bold; margin: 0; font-size: 14px; color: {color};'>{status}</p>
            </div>
        """, unsafe_allow_html=True)

    
    display_kpi(0, "🎯 Overall Average Score", overall_score)
    display_kpi(1, "Completeness", avg_completeness)
    display_kpi(2, "Uniqueness", avg_uniqueness)
    display_kpi(3, "Consistency", avg_consistency)

    st.markdown("---")
    
    # 5. Total Stats and Charts Row
    chart_cols = st.columns([1, 2, 3])
    
    # 3. Total Stats (Total Columns and Total Records) - Changed to Royal Blue (Point 3)
    ROYAL_BLUE = "#4169E1"
    
    with chart_cols[0]:
        st.subheader("Data Volume")
        st.markdown(f"""
            <div style="
                background-color: {ROYAL_BLUE}; 
                color: white; /* Text color for better contrast */
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 15px;
                text-align: center;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
                <p style='font-size: 14px; margin: 0; color: white;'>TOTAL DATA ATTRIBUTES (COLUMNS)</p>
                <h3 style='margin: 5px 0;'>{total_columns}</h3>
            </div>
            <div style="
                background-color: {ROYAL_BLUE};
                color: white; /* Text color for better contrast */
                padding: 15px; 
                border-radius: 8px; 
                text-align: center;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
                <p style='font-size: 14px; margin: 0; color: white;'>TOTAL DATA RECORDS (ROWS X COLUMNS)</p>
                <h3 style='margin: 5px 0;'>{total_data_records}</h3>
            </div>
        """, unsafe_allow_html=True)
        st.caption(f"Note: Records count is based on the last file loaded in the Analyzer page.")

    # --- Pie Chart: Scoring Level ---
    with chart_cols[1]:
        st.subheader("Scoring Level Distribution")
        
        # Only plot if we have calculated data
        if overall_score is not None and not df_calculated.empty:
            # Flatten scores from all dimensions 
            scores_flat = pd.concat([
                df_calculated['Completeness'].dropna(),
                df_calculated['Uniqueness'].dropna(),
                df_calculated['Consistency'].dropna()
            ])
            
            # Apply status binning
            status_counts = scores_flat.apply(get_dq_status).value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            # Define the consistent color map for Plotly
            color_map = {
                'Healthy': get_dq_color(100), # Green
                'Moderate': get_dq_color(50), # Yellow
                'Critical': get_dq_color(20) # Red
            }

            # Plotting the Pie Chart
            fig_pie = px.pie(
                status_counts, 
                values='Count', 
                names='Status', 
                color='Status',
                color_discrete_map=color_map,
                title='Count of Individual Dimension Scores by Health',
                hole=.5
            )
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(showlegend=True, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No calculated scores to display distribution.")

    # --- Line Chart: Data Quality Trend (Simulated) ---
    with chart_cols[2]:
        st.subheader("Data Quality Trend")
        
        # SIMULATE HISTORICAL DATA for the trend line over the last 5 months
        dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=5, freq='MS'))
        
        initial_score = overall_score if overall_score is not None and overall_score > 0 else 75.0 
        
        simulated_scores = []
        current_score = initial_score
        np.random.seed(42) # For consistent simulation on reruns
        for i in range(5):
             # Simulate small changes: random walk with slight upward drift
            change = np.random.uniform(-3, 6) 
            current_score = np.clip(current_score + change, 60, 95) 
            simulated_scores.append(current_score)
        
        df_trend = pd.DataFrame({
            "Date": dates,
            "DQ Score(%)": simulated_scores
        })

        fig_line = px.line(
            df_trend,
            x="Date",
            y="DQ Score(%)",
            title=f"Overall DQ Score Trend for {selected_dataset}",
            markers=True,
            line_shape='linear',
            height=350
        )
        fig_line.update_layout(xaxis_title="Reporting Month", yaxis_title="Overall Score (%)")
        
        # Set line color based on the latest health status
        latest_score_trend = df_trend['DQ Score(%)'].iloc[-1]
        line_color = get_dq_color(latest_score_trend)
        fig_line.update_traces(line=dict(color=line_color))

        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    
    # 6. Results Table with Colored Numbers
    st.subheader(f"Detailed Score Results for: {selected_dataset}")

    # --- INTERACTIVE FILTERING FOR TABLE (New Logic) ---
    
    # 1. Initialize filter state
    if 'dq_table_filter' not in st.session_state:
        st.session_state.dq_table_filter = 'All'

    # 2. Filter Control
    filter_options = ['All', 'Critical', 'Moderate', 'Healthy']
    
    selected_status = st.selectbox(
        "Filter results table by Health Status (from Pie Chart categories):",
        options=filter_options,
        index=filter_options.index(st.session_state.dq_table_filter),
        key='dq_table_filter_select'
    )
    st.session_state.dq_table_filter = selected_status

    # 3. Prepare data for filtering
    df_filtered_scores = df_scores.copy()
    df_filtered_table = df_filtered_scores.copy()

    if st.session_state.dq_table_filter != 'All':
        
        # Unpivot the data to apply status filtering easily across all 3 dimensions (Completeness, Uniqueness, Consistency)
        df_melt = df_filtered_scores.melt(
            id_vars=['Column Name', 'Last Updated'], 
            value_vars=['Completeness', 'Uniqueness', 'Consistency'], 
            var_name='Dimension', 
            value_name='Score'
        )

        # Drop rows where the score is missing (N/A)
        df_melt = df_melt.dropna(subset=['Score'])

        # Calculate the health status for each score
        df_melt['Status'] = df_melt['Score'].apply(get_dq_status)
        
        # Filter the long format DataFrame by the selected status
        df_melt_filtered = df_melt[df_melt['Status'] == st.session_state.dq_table_filter]
        
        # Find the unique column names that belong to the filtered status group
        filtered_columns = df_melt_filtered['Column Name'].unique()
        
        # Re-filter the original wide format table to display ONLY those columns
        df_filtered_table = df_filtered_scores[df_filtered_scores['Column Name'].isin(filtered_columns)]
        
        st.info(f"Showing columns that have at least one dimension score marked as **{st.session_state.dq_table_filter}**.")
    else:
        # If 'All' is selected, show the entire score table
        df_filtered_table = df_filtered_scores
        st.info("Showing all columns.")


    # 4. Results Table Rendering
    
    def color_scores_table(val):
        """Applies color to score values (0-100) based on DQ health rules."""
        if pd.isna(val):
            return '' 
        try:
            score = float(val)
        except:
            return ''
            
        color = get_dq_color(score)
        return f'color: {color}; font-weight: bold'

    if not df_filtered_table.empty:
        # Create a styleable copy 
        df_style = df_filtered_table.copy()
        
        # Apply the color function to the score columns and format as percentages
        styled_df = df_style.style.format({
            # Use a custom formatter to display N/A for NaN values
            'Completeness': lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A",
            'Uniqueness': lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A",
            'Consistency': lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A",
            'Last Updated': lambda t: t.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(t) else 'N/A'
        }).applymap(color_scores_table, subset=['Completeness', 'Uniqueness', 'Consistency'])
        
        # Display the styled dataframe
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
    else:
        st.warning(f"No columns match the selected status: **{st.session_state.dq_table_filter}**.")
        

# --- Main App Execution ---

# 1. Base64 Logo Placeholder for the Sidebar Menu
# NOTE: Replace the long string below with the actual Base64 string of your logo image.
# You can use an online tool to convert your logo file (e.g., logo_v3.png) into a Base64 string.
LOGO_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4CykAAAAAXNSR0IArs4c6QAAAXRJREFUeJzt3DGOwkAQhmE3I6X4SgVpD8o5tFw3o1kHhGAn2H2O4E5s/rQZ5zJm+m07B2cSAAAAAAD+G+6+P/3W2LhY/tP08n/79e/j5Y/t48M2A5FBAAAADEDP/wK+8j+2H/b7efr3x/55jPz0R/dJq/j/HwAAAAAA97f2E/H0eH1w/c5sP2sSAAAAABg4fWb7Tz29188f28PXB5t+RjICAAAAAEDMjj1Wp/f6bVn/45i9/u7/7Tf8nS8f9/MfAAAAAAD+we13Y38+0PsP20/2k//ab6zj5z8AAAAAQKh8h3v/AAAAAADofNiusyv82D3oAAAAAElFTkSuQmCC"

with st.sidebar:
    
    # Menu options (Mapping to existing page functions)
    selected = option_menu(
        menu_title=None, 
        options=[
            "Home Page", 
            "Data Quality Analyzer", 
            "Data Quality Visualizer"
        ],
        default_index=0, # Default to Home Page
        styles={
            "container": {"padding": "0!important", "background-color": "#000000", "color": "white"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#03026f"}, # Dark Blue selection color
        }
    )

# Display content based on selection
if selected == "Home Page":
    home_page()
elif selected == "Data Quality Analyzer":
    dq_page()
elif selected == "Data Quality Visualizer":
    dq_visualizer_page()

