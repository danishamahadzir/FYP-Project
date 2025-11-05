import streamlit as st
import pandas as pd
import numpy as np

# Set page config at the very top for Streamlit best practice
st.set_page_config(
    page_title="Data Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 1. Data Quality Check Functions ---

def check_completeness(df: pd.DataFrame, column: str) -> float:
    """
    Calculates the completeness score (percentage of non-null values) for a single column.
    Score = (Count of non-null values / Total rows) * 100
    """
    if column not in df.columns:
        return 0.0
    total_rows = len(df)
    if total_rows == 0:
        return 0.0
    non_null_count = df[column].count() # .count() excludes NaN/None values
    score = (non_null_count / total_rows) * 100
    return score

def check_completeness_all_columns(df: pd.DataFrame) -> float:
    """
    Calculates the average completeness score across all columns in the DataFrame.
    """
    if df.empty or len(df.columns) == 0:
        return 0.0
    
    total_completeness_sum = 0
    for column in df.columns:
        # Calculate completeness for each column and sum them up
        total_completeness_sum += check_completeness(df, column) 
    
    # Average the completeness scores across all columns
    avg_completeness = total_completeness_sum / len(df.columns)
    return avg_completeness

def check_uniqueness(df: pd.DataFrame, column: str) -> float:
    """
    Calculates the uniqueness score (percentage of unique values) for a single column.
    Score = (Count of unique values / Total rows) * 100
    """
    if column not in df.columns:
        return 0.0
    total_rows = len(df)
    if total_rows == 0:
        return 0.0
    # Drop NaNs before counting unique values for strict uniqueness check
    series = df[column].dropna()
    unique_count = series.nunique()
    score = (unique_count / total_rows) * 100
    return score

def check_consistency(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculates the consistency score by comparing two columns.
    Consistency is defined as the percentage of rows where col1 is equal to col2.
    Score = (Count of rows where col1 == col2 / Total rows) * 100
    """
    if col1 not in df.columns or col2 not in df.columns:
        return 0.0
    total_rows = len(df)
    if total_rows == 0:
        return 0.0

    # To include NaN matches as consistent, we compare values after filling NaNs
    # with a unique placeholder string.
    match_count = (df[col1].fillna('__FILL__') == df[col2].fillna('__FILL__')).sum()
    
    score = (match_count / total_rows) * 100
    return score

# --- 2. Streamlit Page Functions ---

def home_page():
    st.title("🏠 Data Quality Tool")
    st.subheader("Measuring Data Quality Score Against Completeness, Uniqueness & Consistency")
    
    st.markdown("""
    Welcome to the **Data Quality Tool** — an interactive dashboard designed to help you 
    evaluate the overall quality of your datasets before they are used for analysis or reporting. 
    This tool measures your data health across three critical dimensions:
    """)

    # Display a descriptive table
    st.markdown("### 📘 Data Quality Dimensions Overview")
    
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
    # Changed st.table to st.dataframe and added hide_index=True to remove the unwanted 0, 1, 2 column
    st.dataframe(df_overview, hide_index=True, use_container_width=True)

    st.markdown("""
    ---
    ### How to Use This Tool
    1. Navigate to the **Data Quality Analyzer** page using the dropdown in the sidebar.
    2. Upload your dataset (Excel or CSV).
    3. Choose which data quality dimension to analyze.
    4. View your calculated scores instantly!

    ---
    *Tip:* For best results, ensure your dataset is well-structured and column names are clearly labeled.
    """)


from datetime import datetime

def dq_page():
    st.title("📊 Data Quality Analyzer")
    st.subheader("Ensure Your Data is Ready for Action")

# Custom CSS to change the color of the file uploader button AND dropzone
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
    
    /* NEW: Target the border/outline of the dropzone (drag and drop area) */
    div[data-testid="stFileUploaderDropzone"] {
        border-color: #03026f !important; /* Dark Blue */
    }

    /* NEW: Target the text/placeholder ("Drag and drop file here") inside the dropzone */
    div[data-testid="stFileUploaderDropzone"] p {
        color: #03026f !important; /* Dark Blue */
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        Welcome to the **Data Quality Analyzer**. This tool helps you assess the health of your dataset 
        across three critical dimensions: 
        * **Completeness**: Checking for null or missing values.
        * **Uniqueness**: Checking for duplicate records.
        * **Consistency**: Checking if values in two columns match.
        
        ---
        ### Upload your Excel or CSV file below to get started!
    """)

    # --- Session state initialization ---
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'scores' not in st.session_state:
        st.session_state.scores = {}  # last/aggregate dimension scores
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'column_scores' not in st.session_state:
        # Structure: { "col_name" : { "Completeness": {"value":float,"ts":str}, "Uniqueness": {...}, "Consistency": {...} } }
        st.session_state.column_scores = {}

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Excel (.xlsx) or CSV (.csv) file", 
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df

            st.success(f"File uploaded successfully! Total rows: {df.shape[0]}. First 5 rows of data:")
            st.dataframe(df.head(), use_container_width=True)

            st.markdown("---")
            st.subheader("Configure Analysis")

            all_columns = df.columns.tolist()

            # 3. dimension selection — use radio (as you wanted)
            dimension = st.radio(
                "Select Data Quality Dimension to Check:",
                ('Completeness', 'Uniqueness', 'Consistency'),
                horizontal=True,
                key='dimension_select'
            )

            col1 = None
            col2 = None
            col_name_for_storage = None

            # 4. column selection (All Columns only for Completeness)
            if dimension in ['Completeness', 'Uniqueness']:
                options_list = [''] + all_columns
                if dimension == 'Completeness':
                    # Insert 'All Columns' as the second option
                    options_list.insert(1, 'All Columns')
                col1 = st.selectbox(
                    f"Select the column to check for **{dimension}**:",
                    options=options_list,
                    index=0,
                    key='col_select_1'
                )
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

            # 5. Calculate button & logic — only updates history when user clicks
            if st.button(f"Calculate {dimension} Score"):
                # validate selection
                if (dimension != 'Consistency' and col1) or (dimension == 'Consistency' and col1 and col2):
                    score = 0.0
                    with st.spinner(f"Calculating {dimension} score..."):
                        if dimension == 'Completeness':
                            if col1 == 'All Columns':
                                # global completeness: compute for all columns and store them individually with timestamp
                                global_score = check_completeness_all_columns(df)
                                st.session_state.scores['Completeness'] = global_score
                                st.session_state.last_result = {
                                    'dimension': 'Completeness',
                                    'column': 'All Columns',
                                    'score': global_score
                                }
                                # Store completeness for every column with timestamp
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                for c in df.columns.tolist():
                                    st.session_state.column_scores.setdefault(c, {})
                                    st.session_state.column_scores[c]['Completeness'] = {
                                        'value': check_completeness(df, c),
                                        'ts': ts
                                    }
                            else:
                                # single column completeness
                                score = check_completeness(df, col1)
                                col_name_for_storage = col1
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.column_scores.setdefault(col1, {})['Completeness'] = {
                                    'value': score,
                                    'ts': ts
                                }
                                st.session_state.scores['Completeness'] = score
                                st.session_state.last_result = {
                                    'dimension': 'Completeness',
                                    'column': col1,
                                    'score': score
                                }

                        elif dimension == 'Uniqueness':
                            score = check_uniqueness(df, col1)
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.column_scores.setdefault(col1, {})['Uniqueness'] = {
                                'value': score,
                                'ts': ts
                            }
                            st.session_state.scores['Uniqueness'] = score
                            st.session_state.last_result = {
                                'dimension': 'Uniqueness',
                                'column': col1,
                                'score': score
                            }

                        elif dimension == 'Consistency':
                            score = check_consistency(df, col1, col2)
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            pair_name = f"{col1} vs {col2}"
                            st.session_state.column_scores.setdefault(pair_name, {})['Consistency'] = {
                                'value': score,
                                'ts': ts
                            }
                            st.session_state.scores['Consistency'] = score
                            st.session_state.last_result = {
                                'dimension': 'Consistency',
                                'column': pair_name,
                                'score': score
                            }

                        st.balloons()
                        st.success(f"Analysis Complete for {dimension}!")
                else:
                    st.error("Please select the required column(s) to run the analysis.")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
            st.session_state.df = None

    # 6. Overall metrics (same as before)
    if st.session_state.scores:
        st.markdown("---")
        st.subheader("Data Quality Results (Average of all calculations)")
        cols = st.columns(4)
        scores_displayed = 0
        total_score = 0
        current_scores = {k: v for k, v in st.session_state.scores.items() if k in ['Completeness', 'Uniqueness', 'Consistency']}
        for i, (dim, score) in enumerate(current_scores.items()):
            cols[i].metric(
                label=f"{dim} Score",
                value=f"{score:.2f}%",
                delta=f"Based on {st.session_state.df.shape[0]} rows" if st.session_state.df is not None else None
            )
            scores_displayed += 1
            total_score += score
        if scores_displayed > 0:
            overall_score = total_score / scores_displayed
            cols[3].metric(
                label="🎯 Overall DQ Score (Avg)",
                value=f"{overall_score:.2f}%"
            )

    # 7. Data Quality Scores History — appear ONLY if user ran any checks
    if st.session_state.column_scores:
        st.markdown("---")
        st.subheader("Data Quality Scores History")

        # Build table rows — each row corresponds to a column or pair stored in column_scores
        data_for_table = []
        all_cols_in_scores = list(st.session_state.column_scores.keys())

        for i, col_name in enumerate(all_cols_in_scores):
            scores = st.session_state.column_scores[col_name]
            # Extract values if present, else 'N/A'
            if 'Completeness' in scores:
                comp_v = scores['Completeness']['value']
                comp_ts = scores['Completeness']['ts']
                comp_display = f"{comp_v:.2f}%"
            else:
                comp_display = 'N/A'
                comp_ts = None

            if 'Uniqueness' in scores:
                uniq_v = scores['Uniqueness']['value']
                uniq_ts = scores['Uniqueness']['ts']
                uniq_display = f"{uniq_v:.2f}%"
            else:
                uniq_display = 'N/A'
                uniq_ts = None

            if 'Consistency' in scores:
                cons_v = scores['Consistency']['value']
                cons_ts = scores['Consistency']['ts']
                cons_display = f"{cons_v:.2f}%"
            else:
                cons_display = 'N/A'
                cons_ts = None

            # Choose latest timestamp among metrics for display (if any)
            timestamps = [t for t in (comp_ts, uniq_ts, cons_ts) if t is not None]
            latest_ts = timestamps[-1] if timestamps else ''

            data_for_table.append({
                "No": i + 1,
                "Column Name": col_name,
                "Completeness": comp_display,
                "Uniqueness": uniq_display,
                "Consistency": cons_display,
                "Last Updated": latest_ts
            })

        df_column_scores = pd.DataFrame(data_for_table)
        st.dataframe(df_column_scores, hide_index=True, use_container_width=True)

        # Download button (CSV) including timestamp
        csv = df_column_scores.to_csv(index=False).encode('utf-8')
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name="data_quality_history.csv",
            mime="text/csv",
            help="Download your data quality history for Power BI or record keeping"
        )

    # 8. Last Analysis Summary (as before)
    if st.session_state.last_result:
        last_result = st.session_state.last_result
        st.markdown("---")
        st.subheader("Last Analysis Summary")
        col_display = last_result['column']
        if isinstance(last_result['column'], tuple):
            col_display = f"'{last_result['column'][0]}' vs '{last_result['column'][1]}'"
        elif col_display == 'All Columns':
            col_display = "All Columns"
        else:
            col_display = f"'{col_display}'"

        with st.container():
            st.markdown(f"""
                **Dimension Checked:** `{last_result['dimension']}`
                
                **Column(s) Analyzed:** `{col_display}`
                
                **Calculated Score:** **<span style='font-size: 24px; color: #1E90FF;'>{last_result['score']:.2f}%</span>**
            """, unsafe_allow_html=True)



def powerbi_page():
    st.title("🔗 Power BI Connection Guide")
    st.subheader("Seamless Integration with Business Intelligence Tools")
    
    st.markdown("""
        This page demonstrates how you can integrate the data quality analysis results 
        or the underlying data with Power BI for further visualization and reporting.
        
        ### 1. Direct Data Connection
        If you are working with a local data source:
        * **For Excel/CSV:** You can directly import the data file into Power BI Desktop.
        
        ### 2. Live Data Source (Enterprise Setup)
        In a real-world application, this Streamlit tool would connect to an SQL/NoSQL database. 
        Power BI would then connect to the same database (e.g., SQL Server, Azure SQL, PostgreSQL) 
        to ensure live data reporting.
        
        1. **Host the Data:** Ensure your processed data resides in a centralized, secure data store (e.g., Azure Data Lake, Snowflake, dedicated SQL server).
        2. **Power BI Gateway:** Set up an On-premises data gateway if your data source is not cloud-based.
        3. **Get Data in Power BI:** Use the native connector for your data source (e.g., 'SQL Server' connector) in Power BI Desktop.
        4. **Publish:** Publish your report to the Power BI Service to share it.
        
        *Note: This page is conceptual. The actual implementation depends on your database and hosting environment.*
    """)


# --- Main App Execution ---

# Sidebar Navigation Control
st.sidebar.title("Menu")
# Changed st.radio back to st.selectbox for navigation
page = st.sidebar.selectbox(
    "Select Page:",
    ["Home Page", "Data Quality Analyzer", "Power BI Connection Guide"],
    # Set the default index to 0 (Home Page)
    index=0 
)

# Display content based on selection
if page == "Home Page":
    home_page()
elif page == "Data Quality Analyzer":
    dq_page()
elif page == "Power BI Connection Guide":
    powerbi_page()

