import streamlit as st
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import Error

# Set page config at the very top for Streamlit best practice
st.set_page_config(
    page_title="Data Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 1. Database Connection and Schema Management ---

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
    Creates the main fact table for storing the LATEST data quality scores.
    Uses a composite primary key (dataset_name, column_name) to support UPSERT logic.
    """
    if conn is None:
        return
    
    table_name = "latest_dq_scores"
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
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
        # Use parameterized query for safety if possible, or simple formatting for this internal app
        where_clause.append(f"dataset_name = '{dataset_name.replace("'", "''")}'") # Simple escaping for security
    
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

# --- 3. Streamlit Page Functions ---

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
                st.rerun() # Replaced st.experimental_rerun() with st.rerun()

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
                st.success(f"✅ {dimension} score saved to database for {column_name_key}!")
                st.rerun() # Replaced st.experimental_rerun() with st.rerun()

        
        # 5. Overall metrics (still based on session state for quick averages)
        if st.session_state.scores:
            st.markdown("---")
            st.subheader(f"Data Quality Results Overview for: {st.session_state.dataset_name}")
            cols = st.columns(4)
            scores_displayed = 0
            total_score = 0
            current_scores = {k: v for k, v in st.session_state.scores.items() if k in ['Completeness', 'Uniqueness', 'Consistency']}
            for i, (dim, score) in enumerate(current_scores.items()):
                cols[i].metric(
                    label=f"{dim} Score",
                    value=f"{score:.2f}%",
                )
                scores_displayed += 1
                total_score += score
            if scores_displayed > 0:
                overall_score = total_score / scores_displayed
                cols[3].metric(
                    label="🎯 Overall DQ Score (Avg)",
                    value=f"{overall_score:.2f}%"
                )

        # 6. Data Quality Scores from DB (The most important part for persistence)
        st.markdown("---")
        st.subheader(f"Latest Persisted Scores (Source: MySQL - Dataset: {st.session_state.dataset_name})")

        # Fetch data directly from the DB
        df_db_scores = fetch_latest_scores(st.session_state.dataset_name)

        if not df_db_scores.empty:
            # Format the output table for clean display
            df_display = df_db_scores.copy()
            
            # Format percentages and handle N/A (which is NULL from DB)
            for col in ['Completeness', 'Uniqueness', 'Consistency']:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A'
                )
            
            # Reorder and add 'No' column
            df_display.insert(0, 'No', range(1, 1 + len(df_display)))
            
            st.dataframe(df_display, hide_index=True, use_container_width=True)

            # Download button
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                f"⬇️ Download Scores for {st.session_state.dataset_name} as CSV",
                data=csv,
                file_name=f"dq_latest_scores_{st.session_state.dataset_name.replace('.', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No scores found for this dataset in the database yet. Run an analysis above!")

        # 7. Last Analysis Summary (The original target of the question - now correctly displayed)
        if st.session_state.last_result:
            last_result = st.session_state.last_result
            st.markdown("---")
            st.subheader("Last Analysis Summary")
            col_display = last_result['column']
            
            with st.container():
                st.markdown(f"""
                    **Dimension Checked:** `{last_result['dimension']}`
                    
                    **Column(s) Analyzed:** `{col_display}`
                    
                    **Calculated Score:** **<span style='font-size: 24px; color: #1E90FF;'>{last_result['score']:.2f}%</span>**
                """, unsafe_allow_html=True)
    else:
        # Display this if no file is currently loaded in session state
        st.info("Please upload a file to begin the data quality analysis.")


def powerbi_page():
    st.title("🔗 Power BI Connection Guide")
    st.subheader("Connecting Your Data Quality Fact Table to Power BI")
    
    st.markdown("""
        Integrating your persisted Data Quality Scores into Power BI is straightforward, as the scores 
        are stored in a normalized **Fact Table** in your **Railway MySQL database**.
        
        The table is named: `latest_dq_scores`.
        
        ### Connection Steps:
        
        1.  **Open Power BI Desktop** and click **Get Data** -> **MySQL database**.
        2.  Enter your **Railway MySQL credentials**:
            * **Server:** Your Railway MySQL host (e.g., `aws.connect.psdb.cloud`).
            * **Database:** Your database name.
        3.  Power BI will ask for credentials (Username and Password). Use the credentials stored in your Streamlit secrets.
        4.  In the Navigator window, select the table **`latest_dq_scores`**.
        5.  Click **Load**.
        
        ### Report Structure (How to Visualize):
        
        Since the table uses `dataset_name` and `column_name` as keys, you can easily build visualizations:
        
        * **KPI Cards:** Use `Completeness`, `Uniqueness`, and `Consistency` fields to create high-level KPI cards, using the **Average** aggregation (since each score is 0-100).
        * **Table Visualization:** Create a table (like the one displayed in the Analyzer) showing `dataset_name`, `column_name`, and all three percentage scores.
        * **Trend Analysis:** Although this table only holds the *latest* score, you can implement a scheduled task (e.g., a simple cron job or a Streamlit Cloud scheduled run) that inserts the scores into a **history log table** (e.g., `dq_historical_log`) once a day. Power BI can then connect to *that* table for trend charts over time.
    """)


# --- Main App Execution ---

# Sidebar Navigation Control
st.sidebar.title("Menu")
page = st.sidebar.selectbox(
    "Select Page:",
    ["Home Page", "Data Quality Analyzer", "Power BI Connection Guide"],
    index=1 # Default to Analyzer page for convenience
)

# Display content based on selection
if page == "Home Page":
    home_page()
elif page == "Data Quality Analyzer":
    dq_page()
elif page == "Power BI Connection Guide (WIP)":
    powerbi_page()

