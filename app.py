import streamlit as st
from llama_cloud_services import LlamaExtract
from llama_cloud_services.extract import ExtractConfig, ExtractTarget, ExtractMode
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import hashlib
import os
import tempfile
from datetime import datetime
import re
import plotly.express as px
import plotly.graph_objects as go

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

def clean_currency_value(value) -> float:
    """Clean and convert currency strings to float values"""
    if value is None or pd.isna(value) or value == "" or str(value).strip() == "":
        return 0.0
    
    value_str = str(value).strip()
    if not value_str or value_str == "0":
        return 0.0
    
    value_str = re.sub(r'[$£€¥₹,\s]', '', value_str)
    
    if '(' in str(value) and ')' in str(value):
        value_str = re.sub(r'[()]', '', value_str)
        value_str = '-' + value_str
    else:
        value_str = re.sub(r'[()]', '', value_str)
    
    if not value_str or value_str == '-':
        return 0.0
    
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return 0.0

def parse_date(date_str: str) -> datetime:
    """Parse date string with multiple format support"""
    if not date_str or pd.isna(date_str):
        return None
    
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

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**📁 Supported Formats:**")
    st.markdown("- PDF files")
    st.markdown("- PNG images") 
    st.markdown("- JPG/JPEG images")
    
    currency_symbol = st.selectbox(
        "💱 Currency Symbol",
        options=["$", "£", "€", "¥", "₹"],
        index=0
    )
    
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
def extract_statement_data(file_hash: str, *, _file_content: bytes, _filename: str) -> dict:
    try:
        file_extension = os.path.splitext(_filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(_file_content)
            tmp_file_path = tmp_file.name
        
        try:
            result = agent.extract(tmp_file_path)
            return result.data
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    except Exception as e:
        st.error(f"❌ Extraction failed: {str(e)}")
        return None

# ---------------------------
# 🔹 File Upload
# ---------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your bank statement",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, PNG, JPG, JPEG"
)

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Name", uploaded_file.name)
    with col2:
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        st.metric("📊 File Size", f"{file_size_mb:.2f} MB")
    with col3:
        file_type = uploaded_file.type
        st.metric("📋 File Type", file_type)
    
    file_bytes = uploaded_file.getbuffer()
    file_hash = get_file_hash(file_bytes)
    
    if "last_file_hash" in st.session_state and st.session_state["last_file_hash"] == file_hash:
        extracted_data = st.session_state["cached_data"]
        st.info("📋 Using cached extraction results")
    else:
        with st.spinner("🔄 Processing your statement..."):
           extracted_data = extract_statement_data(file_hash, _file_content=file_bytes, _filename=uploaded_file.name)
           if extracted_data:
                st.session_state["last_file_hash"] = file_hash
                st.session_state["cached_data"] = extracted_data

    # ---------------------------
    # 🔹 Data Processing
    # ---------------------------
    
    if extracted_data and "transactions" in extracted_data:
        try:
            statement = Statement.model_validate(extracted_data)
            
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
            
            if statement.transactions:
                st.success(f"✅ Successfully extracted {len(statement.transactions)} transactions!")
                
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
                df['parsed_date'] = df['date'].apply(parse_date)
                df = df.sort_values('parsed_date', na_position='last').reset_index(drop=True)
                
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
                
                st.subheader("📊 Transactions Overview")
                st.dataframe(
                    display_df[['date', 'description', 'debit', 'credit', 'balance']],
                    use_container_width=True
                )
                
                # Financial Summary
                st.subheader("📈 Financial Summary")
                total_debits = df['debit'].sum()
                total_credits = df['credit'].sum()
                net_flow = total_credits - total_debits
                closing_balance = df['balance'].iloc[-1] if not df.empty else 0
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("💸 Total Debits", f"{currency_symbol}{total_debits:,.2f}")
                with metric_col2:
                    st.metric("💰 Total Credits", f"{currency_symbol}{total_credits:,.2f}")
                with metric_col3:
                    st.metric("📊 Net Flow", f"{currency_symbol}{net_flow:,.2f}")
                with metric_col4:
                    st.metric("💳 Closing Balance", f"{currency_symbol}{closing_balance:,.2f}")
                
                # Visualization Tabs
                st.subheader("📊 Transaction Analysis")
                tab1, tab2, tab3 = st.tabs(["💹 Flow Analysis", "📈 Balance Trend", "🏷️ Categories"])
                
                with tab1:
                    st.markdown("**💰 Cash Flow Overview**")
                    flow_fig = go.Figure(data=[
                        go.Bar(name='Credits', x=['Inflow'], y=[total_credits], marker_color='#00CC88'),
                        go.Bar(name='Debits', x=['Outflow'], y=[total_debits], marker_color='#FF6B6B')
                    ])
                    flow_fig.update_layout(
                        title="Cash Flow Comparison",
                        yaxis_title=f"Amount ({currency_symbol})",
                        barmode='group',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(flow_fig, use_container_width=True)
                
                with tab2:
                    if len(df) > 1:
                        balance_fig = go.Figure()
                        balance_fig.add_trace(go.Scatter(
                            x=df['date'],
                            y=df['balance'],
                            mode='lines+markers',
                            name='Balance',
                            line=dict(color='#4A90E2', width=3),
                            marker=dict(size=8)
                        ))
                        balance_fig.update_layout(
                            title='Account Balance Over Time',
                            xaxis_title='Date',
                            yaxis_title=f'Balance ({currency_symbol})',
                            height=400
                        )
                        st.plotly_chart(balance_fig, use_container_width=True)
                    else:
                        st.info("📊 Balance trend requires more than one transaction")
                
                with tab3:
                    if len(df) > 0:
                        desc_counts = df['description'].value_counts().head(10)
                        if not desc_counts.empty:
                            st.markdown("**Most Frequent Transaction Types:**")
                            cat_fig = go.Figure(go.Bar(
                                x=desc_counts.values,
                                y=desc_counts.index,
                                orientation='h',
                                marker_color='#9B59B6'
                            ))
                            cat_fig.update_layout(
                                title="Top Transaction Types",
                                xaxis_title="Frequency",
                                yaxis_title="Transaction Type",
                                height=400
                            )
                            st.plotly_chart(cat_fig, use_container_width=True)
                        
                        debit_count = len(df[df['debit'] > 0])
                        credit_count = len(df[df['credit'] > 0])
                        
                        if debit_count > 0 or credit_count > 0:
                            pie_fig = go.Figure(data=[go.Pie(
                                labels=['Debit Transactions', 'Credit Transactions'],
                                values=[debit_count, credit_count],
                                hole=.3,
                                marker_colors=['#FF6B6B', '#00CC88']
                            )])
                            pie_fig.update_layout(
                                title="Transaction Type Distribution",
                                height=400
                            )
                            st.plotly_chart(pie_fig, use_container_width=True)
                
                # Download Options
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
            with st.expander("🔍 View Raw Data for Debugging"):
                st.json(extracted_data)
    
    elif extracted_data:
        st.warning("⚠️ Data extracted but no transactions found.")
        with st.expander("🔍 View Raw Extracted Data"):
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
