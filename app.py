import streamlit as st
from llama_cloud_services import LlamaExtract
from llama_cloud_services.extract import ExtractConfig, ExtractTarget, ExtractMode
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import hashlib
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# ---------------------------
# 🔹 Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Bank Statement Parser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 🔹 Enhanced Schema with Validation
# ---------------------------

class Transaction(BaseModel):
    date: str = Field(description="Date of the transaction (YYYY-MM-DD format preferred)")
    description: str = Field(description="Name or type of the transaction")
    debit: Optional[str] = Field(default="0", description="Outflow transactions (negative values)")
    credit: Optional[str] = Field(default="0", description="Inflow transactions (positive values)")
    balance: str = Field(description="Remaining balance after the transaction")
    
    class Config:
        str_strip_whitespace = True

class Statement(BaseModel):
    account_holder: Optional[str] = Field(default=None, description="Account holder name")
    account_number: Optional[str] = Field(default=None, description="Account number")
    statement_period: Optional[str] = Field(default=None, description="Statement period")
    opening_balance: Optional[str] = Field(default=None, description="Opening balance")
    closing_balance: Optional[str] = Field(default=None, description="Closing balance")
    transactions: List[Transaction] = Field(description="List of transaction entries")

# ---------------------------
# 🔹 Utility Functions
# ---------------------------

def clean_currency_value(value: str) -> float:
    """Clean and convert currency strings to float values"""
    if pd.isna(value) or value is None or value == "":
        return 0.0
    
    # Convert to string and clean
    value_str = str(value).strip()
    
    # Remove currency symbols and clean
    value_str = re.sub(r'[$£€¥₹,\s]', '', value_str)
    value_str = re.sub(r'[()]', '', value_str)  # Remove parentheses
    
    # Handle negative values in parentheses format
    if '(' in str(value) and ')' in str(value):
        value_str = '-' + value_str
    
    try:
        return float(value_str) if value_str else 0.0
    except ValueError:
        return 0.0

def parse_date(date_str: str) -> datetime:
    """Parse date string with multiple format support"""
    if not date_str or pd.isna(date_str):
        return None
    
    # Common date formats
    formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', 
        '%m-%d-%Y', '%d-%m-%Y', '%B %d, %Y',
        '%b %d, %Y', '%d %b %Y', '%d %B %Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    
    return None

def get_file_hash(file_bytes: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.sha256(file_bytes).hexdigest()

# ---------------------------
# 🔹 Streamlit UI Setup
# ---------------------------

st.title("🏦 Advanced Bank Statement Parser")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File format selection
    supported_formats = st.multiselect(
        "📁 Supported File Formats",
        options=["PDF", "PNG", "JPG", "JPEG"],
        default=["PDF"],
        disabled=True
    )
    
    # Currency selection
    currency_symbol = st.selectbox(
        "💱 Currency Symbol",
        options=["$", "£", "€", "¥", "₹"],
        index=0
    )
    
    # Date format preference
    date_format = st.selectbox(
        "📅 Expected Date Format",
        options=["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD", "Auto-detect"],
        index=3
    )

# ---------------------------
# 🔹 API Key Validation
# ---------------------------

try:
    api_key = st.secrets["LLAMA_API_KEY"]
    if not api_key:
        st.error("❌ LLAMA_API_KEY not found in secrets. Please configure your API key.")
        st.stop()
except KeyError:
    st.error("❌ LLAMA_API_KEY not found in secrets. Please configure your API key.")
    st.stop()

# ---------------------------
# 🔹 Enhanced Agent Initialization
# ---------------------------

@st.cache_resource
def initialize_agent():
    """Initialize and cache the LlamaExtract agent"""
    try:
        extractor = LlamaExtract(api_key=api_key)
        config = ExtractConfig(
            extraction_target=ExtractTarget.PER_DOC,
            extraction_mode=ExtractMode.MULTIMODAL
        )

        agent_name = "enhanced_bank_statement_parser"

        try:
            agent = extractor.get_agent(agent_name)
            st.sidebar.success("✅ Agent loaded successfully")
        except Exception:
            agent = extractor.create_agent(
                name=agent_name,
                data_schema=Statement,
                config=config
            )
            st.sidebar.success("✅ New agent created successfully")
        
        return agent
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return None

agent = initialize_agent()

if not agent:
    st.stop()

# ---------------------------
# 🔹 Enhanced Extraction Function
# ---------------------------

@st.cache_data(show_spinner="🔍 Extracting data from your statement...")
def extract_statement_data(file_bytes: bytes, filename: str) -> dict:
    """Extract data from uploaded statement file"""
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            result = agent.extract(tmp_file_path)
            return result.data
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"❌ Extraction failed: {str(e)}")
        return None

# ---------------------------
# 🔹 Enhanced File Upload
# ---------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your bank statement",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Name", uploaded_file.name)
    with col2:
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        st.metric("📊 File Size", f"{file_size_mb:.2f} MB")
    with col3:
        file_type = uploaded_file.type
        st.metric("📋 File Type", file_type)
    
    # Process file
    file_bytes = uploaded_file.getbuffer()
    file_hash = get_file_hash(file_bytes)
    
    # Check if file was already processed
    if ("last_file_hash" in st.session_state and 
        st.session_state["last_file_hash"] == file_hash):
        extracted_data = st.session_state["cached_data"]
        st.info("📋 Using cached extraction results")
    else:
        with st.spinner("🔄 Processing your statement..."):
            extracted_data = extract_statement_data(file_bytes, uploaded_file.name)
            if extracted_data:
                st.session_state["last_file_hash"] = file_hash
                st.session_state["cached_data"] = extracted_data

    # ---------------------------
    # 🔹 Enhanced Data Processing
    # ---------------------------
    
    if extracted_data and "transactions" in extracted_data:
        try:
            statement = Statement.model_validate(extracted_data)
            
            # Display statement header info
            if any([statement.account_holder, statement.account_number, statement.statement_period]):
                st.subheader("📋 Statement Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    if statement.account_holder:
                        st.info(f"**Account Holder:** {statement.account_holder}")
                
                with info_col2:
                    if statement.account_number:
                        st.info(f"**Account Number:** {statement.account_number}")
                
                with info_col3:
                    if statement.statement_period:
                        st.info(f"**Period:** {statement.statement_period}")
            
            # Process transactions
            if statement.transactions:
                st.success(f"✅ Successfully extracted {len(statement.transactions)} transactions!")
                
                # Convert to DataFrame with enhanced processing
                transactions_data = []
                for txn in statement.transactions:
                    processed_txn = {
                        'date': txn.date,
                        'description': txn.description,
                        'debit': clean_currency_value(txn.debit),
                        'credit': clean_currency_value(txn.credit),
                        'balance': clean_currency_value(txn.balance)
                    }
                    transactions_data.append(processed_txn)
                
                df = pd.DataFrame(transactions_data)
                
                # Sort by date if possible
                df['parsed_date'] = df['date'].apply(parse_date)
                df = df.sort_values('parsed_date', na_position='last').reset_index(drop=True)
                
                # Create display DataFrame
                display_df = df.copy()
                display_df['debit'] = display_df['debit'].apply(
                    lambda x: f"{currency_symbol}{x:,.2f}" if x > 0 else ""
                )
                display_df['credit'] = display_df['credit'].apply(
                    lambda x: f"{currency_symbol}{x:,.2f}" if x > 0 else ""
                )
                display_df['balance'] = display_df['balance'].apply(
                    lambda x: f"{currency_symbol}{x:,.2f}"
                )
                
                # Display transactions table
                st.subheader("📊 Transactions Overview")
                st.dataframe(
                    display_df[['date', 'description', 'debit', 'credit', 'balance']],
                    use_container_width=True
                )
                
                # Enhanced Summary Statistics
                st.subheader("📈 Financial Summary")
                
                total_debits = df['debit'].sum()
                total_credits = df['credit'].sum()
                net_flow = total_credits - total_debits
                opening_balance = df['balance'].iloc[0] if not df.empty else 0
                closing_balance = df['balance'].iloc[-1] if not df.empty else 0
                
                # Summary metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "💸 Total Debits",
                        f"{currency_symbol}{total_debits:,.2f}",
                        delta=f"-{total_debits:.2f}" if total_debits > 0 else None
                    )
                
                with metric_col2:
                    st.metric(
                        "💰 Total Credits",
                        f"{currency_symbol}{total_credits:,.2f}",
                        delta=f"+{total_credits:.2f}" if total_credits > 0 else None
                    )
                
                with metric_col3:
                    st.metric(
                        "📊 Net Flow",
                        f"{currency_symbol}{net_flow:,.2f}",
                        delta=f"{'+' if net_flow >= 0 else ''}{net_flow:.2f}"
                    )
                
                with metric_col4:
                    st.metric(
                        "💳 Closing Balance",
                        f"{currency_symbol}{closing_balance:,.2f}"
                    )
                
                # Enhanced Visualizations
                st.subheader("📊 Transaction Analysis")
                
                tab1, tab2, tab3 = st.tabs(["💹 Flow Analysis", "📈 Balance Trend", "🏷️ Categories"])
                
                with tab1:
                    # Debit vs Credit comparison
                    flow_fig = go.Figure(data=[
                        go.Bar(name='Credits', x=['Inflow'], y=[total_credits], marker_color='green'),
                        go.Bar(name='Debits', x=['Outflow'], y=[total_debits], marker_color='red')
                    ])
                    flow_fig.update_layout(
                        title="Cash Flow Overview",
                        yaxis_title=f"Amount ({currency_symbol})",
                        barmode='group'
                    )
                    st.plotly_chart(flow_fig, use_container_width=True)
                
                with tab2:
                    if len(df) > 1:
                        # Balance trend over time
                        balance_fig = px.line(
                            df, 
                            x='date', 
                            y='balance',
                            title='Account Balance Trend',
                            markers=True
                        )
                        balance_fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title=f"Balance ({currency_symbol})"
                        )
                        st.plotly_chart(balance_fig, use_container_width=True)
                    else:
                        st.info("📊 Balance trend requires more than one transaction")
                
                with tab3:
                    # Transaction categories (simple keyword-based)
                    st.info("🔄 Advanced categorization coming soon! Currently showing transaction descriptions.")
                    
                    # Show top descriptions
                    if len(df) > 0:
                        desc_counts = df['description'].value_counts().head(10)
                        if not desc_counts.empty:
                            fig_cat = px.bar(
                                x=desc_counts.values,
                                y=desc_counts.index,
                                orientation='h',
                                title="Top Transaction Types"
                            )
                            fig_cat.update_layout(
                                xaxis_title="Frequency",
                                yaxis_title="Transaction Type"
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                
                # Download functionality
                st.subheader("💾 Export Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            else:
                st.warning("⚠️ No transactions found in the extracted data.")
                
        except Exception as e:
            st.error(f"❌ Error processing statement data: {str(e)}")
            st.json(extracted_data)  # Show raw data for debugging
    
    elif extracted_data:
        st.warning("⚠️ Data extracted but no transactions found.")
        st.json(extracted_data)
    
    else:
        st.error("❌ Failed to extract data from the uploaded file. Please try again or check if the file format is supported.")

# ---------------------------
# 🔹 Footer
# ---------------------------

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🏦 Bank Statement Parser | Built with Streamlit & LlamaExtract</p>
        <p><small>Secure processing - Your data is not stored permanently</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
