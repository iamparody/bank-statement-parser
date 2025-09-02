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
import plotly.graph_objects as go

# Fallback parser import
try:
    from bank_parser import extract_text_from_pdf, extract_transactions
    FALLBACK_PARSER_AVAILABLE = True
except ImportError:
    FALLBACK_PARSER_AVAILABLE = False
    st.warning("⚠️ Fallback parser not available. Large PDFs may not extract completely.")

# Optional page-splitting fallback if the extractor struggles with very large PDFs.
# Will be used only if installed; otherwise safely ignored.
try:
    from pypdf import PdfReader, PdfWriter  # lightweight, modern alternative to PyPDF2
    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False

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
# 🔹 Utilities (memory-safe + fast)
# ---------------------------

CURRENCY_PATTERN = re.compile(r'[$£€¥₹,\s]')

def clean_currency_value(value) -> float:
    """Clean and convert currency strings to float values."""
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s or s == "0":
        return 0.0
    s_clean = CURRENCY_PATTERN.sub('', s)
    # handle (123.45) negatives
    if '(' in s and ')' in s:
        s_clean = s_clean.replace('(', '').replace(')', '')
        s_clean = '-' + s_clean
    else:
        s_clean = s_clean.replace('(', '').replace(')', '')
    if not s_clean or s_clean == '-':
        return 0.0
    try:
        return float(s_clean)
    except Exception:
        return 0.0

def parse_date(date_str: str):
    """Parse date string with multiple format support; return None if unknown."""
    if not date_str:
        return None
    try:
        import pandas as _pd  # use pandas' tolerant parser
        return _pd.to_datetime(str(date_str), errors='coerce', dayfirst=False, infer_datetime_format=True)
    except Exception:
        return None

def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def validate_transactions_chunked(transactions: List[dict], chunk_size: int = 250, progress=None) -> List[Transaction]:
    """Pydantic-validate transactions in chunks to keep memory + CPU in check."""
    validated: List[Transaction] = []
    total = len(transactions)
    if progress:
        progress.progress(0.0, text="Validating transactions...")
    for i in range(0, total, chunk_size):
        subset = transactions[i:i+chunk_size]
        # Use BaseModel parsing per item; faster & isolates bad rows
        for tx in subset:
            try:
                validated.append(Transaction(**tx))
            except Exception:
                # If a row fails validation, coerce minimally to avoid full-stop
                validated.append(Transaction(
                    date=str(tx.get('date', '')),
                    description=str(tx.get('description', '')),
                    debit=str(tx.get('debit', '0')),
                    credit=str(tx.get('credit', '0')),
                    balance=str(tx.get('balance', '0')),
                ))
        if progress and total:
            progress.progress(min((i + len(subset)) / total, 1.0), text=f"Validated {i + len(subset)}/{total} transactions")
    if progress:
        progress.empty()
    return validated

# ---------------------------
# 🔹 UI
# ---------------------------

st.title("🏦 Advanced Bank Statement Parser")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**📁 Supported Formats:**\n- PDF files\n- PNG images\n- JPG/JPEG images")
    currency_symbol = st.selectbox("💱 Currency Symbol", options=["$", "£", "€", "¥", "₹"], index=0)
    date_format = st.selectbox("📅 Expected Date Format", options=["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD", "Auto-detect"], index=3)
    chunk_pages = st.slider("🧩 PDF chunk size (pages)", 3, 15, 6, help="If needed, we'll split very large PDFs into N-page chunks for extraction.")
    
    # Fallback settings
    st.markdown("### 🔄 Fallback Parser")
    use_fallback_threshold = st.slider("Transaction threshold for fallback", 10, 100, 50, 
                                     help="If extracted transactions are below this number, use fallback parser")
    st.caption("Tip: Larger chunks = fewer API calls, smaller chunks = more resilient on heavy PDFs.")

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
# 🔹 Agent Initialization (cached)
# ---------------------------

@st.cache_resource
def initialize_agent():
    try:
        extractor = LlamaExtract(api_key=api_key)
        # Default config optimized for large docs
        default_config = ExtractConfig(
            extraction_target=ExtractTarget.PER_DOC,
            extraction_mode=ExtractMode.MULTIMODAL
        )
        agent_name = "enhanced_bank_statement_parser"
        try:
            agent_obj = extractor.get_agent(agent_name)
            st.sidebar.success("✅ Agent loaded")
        except Exception:
            agent_obj = extractor.create_agent(name=agent_name, data_schema=Statement, config=default_config)
            st.sidebar.success("✅ Agent created")
        return agent_obj
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {e}")
        return None

agent = initialize_agent()
if not agent:
    st.stop()

# ---------------------------
# 🔹 Extraction (memory-safe, large-PDF aware)
# ---------------------------

@st.cache_data(show_spinner="🔍 Extracting data from your statement...")
def extract_statement_data(file_hash: str, *, _file_content: bytes, _filename: str, per_page: bool, per_page_chunk: int) -> dict:
    """
    Use hash for cache key, skip hashing large bytes via leading underscores.
    If PDF is very large and per_page=True, attempt chunk extraction & merge.
    """
    temp_path = None
    try:
        ext = os.path.splitext(_filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(_file_content)
            temp_path = tmp.name

        # Primary strategy: let the service handle big files (PER_PAGE gives it more leeway)
        if ext == ".pdf" and per_page:
            base_config = ExtractConfig(
                extraction_target=ExtractTarget.PER_PAGE,
                extraction_mode=ExtractMode.MULTIMODAL
            )
            try:
                result = agent.extract(temp_path, config=base_config)
                data = result.data
                # If service returns already-merged, great. If not, we'll merge below.
                if isinstance(data, dict) and "transactions" in data:
                    return {"data": data, "temp_path": temp_path}
            except Exception:
                # fall back to manual chunking below
                pass

            # Fallback path: manual chunking if pypdf available
            if PYPDF_AVAILABLE:
                reader = PdfReader(temp_path)
                n_pages = len(reader.pages)
                merged_transactions = []
                merged_meta = {}
                # page-chunk loop with progress
                for start in range(0, n_pages, per_page_chunk):
                    end = min(start + per_page_chunk, n_pages)
                    writer = PdfWriter()
                    for p in range(start, end):
                        writer.add_page(reader.pages[p])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as chunk_file:
                        writer.write(chunk_file)
                        chunk_path = chunk_file.name
                    try:
                        chunk_result = agent.extract(chunk_path, config=base_config)
                        chunk_data = getattr(chunk_result, "data", None) or {}
                        # merge meta once
                        if not merged_meta:
                            for k in ["account_holder", "account_number", "statement_period", "opening_balance", "closing_balance"]:
                                if k in chunk_data:
                                    merged_meta[k] = chunk_data.get(k)
                        # collect transactions
                        if "transactions" in chunk_data and isinstance(chunk_data["transactions"], list):
                            merged_transactions.extend(chunk_data["transactions"])
                    finally:
                        try:
                            os.unlink(chunk_path)
                        except Exception:
                            pass
                if merged_transactions:
                    merged_meta["transactions"] = merged_transactions
                    return {"data": merged_meta, "temp_path": temp_path}
                # if chunking produced nothing, fall back to doc-level
            # Last resort: PER_DOC
            result = agent.extract(temp_path)
            return {"data": result.data, "temp_path": temp_path}

        # Non-PDF or no per-page requested → simple path
        result = agent.extract(temp_path)
        return {"data": result.data, "temp_path": temp_path}

    except Exception as e:
        st.error(f"❌ Extraction failed: {e}")
        return {"data": None, "temp_path": temp_path}

# ---------------------------
# 🔹 Uploader
# ---------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your bank statement",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF, PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # IMPORTANT: read() returns bytes (hashable, not memoryview)
    file_bytes = uploaded_file.read()
    file_hash = get_file_hash(file_bytes)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File Name", uploaded_file.name)
    with col2:
        st.metric("📊 File Size", f"{len(file_bytes)/(1024*1024):.2f} MB")
    with col3:
        st.metric("📋 File Type", uploaded_file.type)

    # Cache-aware call: underscores prevent hashing of the heavy params
    with st.spinner("🔄 Processing your statement..."):
        extraction_result = extract_statement_data(
            file_hash,
            _file_content=file_bytes,
            _filename=uploaded_file.name,
            per_page=True,                  # enable large-PDF-optimized path
            per_page_chunk=chunk_pages      # sidebar-configurable
        )

    extracted_data = extraction_result.get("data") if extraction_result else None
    temp_path = extraction_result.get("temp_path") if extraction_result else None

    # ---------------------------
    # 🔹 Fallback Parser Logic
    # ---------------------------
    if extracted_data and isinstance(extracted_data, dict) and "transactions" in extracted_data:
        original_count = len(extracted_data.get("transactions", []))
        
        # Check if we should use fallback parser
        if FALLBACK_PARSER_AVAILABLE and original_count < use_fallback_threshold and temp_path:
            with st.spinner("🔄 Using fallback parser for better extraction..."):
                try:
                    text = extract_text_from_pdf(temp_path)
                    fallback_transactions = extract_transactions(text)
                    
                    if len(fallback_transactions) > original_count:
                        st.success(f"✅ Fallback parser found {len(fallback_transactions)} transactions vs {original_count} from primary extraction")
                        extracted_data["transactions"] = fallback_transactions
                    else:
                        st.info(f"ℹ️ Primary extraction ({original_count} transactions) was sufficient")
                        
                except Exception as e:
                    st.warning(f"⚠️ Fallback parser failed: {e}")

    # Clean up temp file
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    # ---------------------------
    # 🔹 Data Processing
    # ---------------------------
    if extracted_data and isinstance(extracted_data, dict) and "transactions" in extracted_data:
        try:
            # Validate in chunks (fast + stable)
            val_prog = st.progress(0.0, text="Preparing transactions...")
            tx_validated = validate_transactions_chunked(extracted_data.get("transactions", []), chunk_size=250, progress=val_prog)

            # Build Statement (avoid validating all at once for speed)
            stmt = Statement(
                account_holder=extracted_data.get("account_holder"),
                account_number=extracted_data.get("account_number"),
                statement_period=extracted_data.get("statement_period"),
                opening_balance=str(extracted_data.get("opening_balance", "") or ""),
                closing_balance=str(extracted_data.get("closing_balance", "") or ""),
                transactions=tx_validated
            )

            # Header
            if any([stmt.account_holder, stmt.account_number, stmt.statement_period]):
                st.subheader("📋 Statement Information")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if stmt.account_holder:
                        st.info(f"**Account Holder:** {stmt.account_holder}")
                with c2:
                    if stmt.account_number:
                        st.info(f"**Account Number:** {stmt.account_number}")
                with c3:
                    if stmt.statement_period:
                        st.info(f"**Period:** {stmt.statement_period}")

            # Transactions → DataFrame (vectorized & memory-friendly)
            st.success(f"✅ Extracted {len(stmt.transactions)} transactions")
            df = pd.DataFrame([{
                "date": t.date,
                "description": t.description,
                "debit": clean_currency_value(t.debit),
                "credit": clean_currency_value(t.credit),
                "balance": clean_currency_value(t.balance)
            } for t in stmt.transactions])

            # Dates
            df["parsed_date"] = df["date"].apply(lambda x: parse_date(x))
            df.sort_values("parsed_date", inplace=True, na_position="last")
            df.reset_index(drop=True, inplace=True)

            # Pretty display table
            display_df = df.copy()
            display_df["debit"]   = display_df["debit"].apply(lambda x: f"{currency_symbol}{x:,.2f}" if x > 0 else "")
            display_df["credit"]  = display_df["credit"].apply(lambda x: f"{currency_symbol}{x:,.2f}" if x > 0 else "")
            display_df["balance"] = display_df["balance"].apply(lambda x: f"{currency_symbol}{x:,.2f}")

            st.subheader("📊 Transactions Overview")
            st.dataframe(display_df[["date", "description", "debit", "credit", "balance"]], use_container_width=True)

            # Summary
            st.subheader("📈 Financial Summary")
            total_debits  = float(df["debit"].sum())
            total_credits = float(df["credit"].sum())
            net_flow      = total_credits - total_debits
            closing_balance = float(df["balance"].iloc[-1]) if not df.empty else 0.0

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("💸 Total Debits",  f"{currency_symbol}{total_debits:,.2f}")
            with m2: st.metric("💰 Total Credits", f"{currency_symbol}{total_credits:,.2f}")
            with m3: st.metric("📊 Net Flow",      f"{currency_symbol}{net_flow:,.2f}")
            with m4: st.metric("💳 Closing Balance", f"{currency_symbol}{closing_balance:,.2f}")

            # Charts
            st.subheader("📊 Transaction Analysis")
            tab1, tab2, tab3 = st.tabs(["💹 Flow Analysis", "📈 Balance Trend", "🏷️ Categories"])

            with tab1:
                flow_fig = go.Figure(data=[
                    go.Bar(name='Credits', x=['Inflow'], y=[total_credits], marker_color='#00CC88',
                           text=[f'{currency_symbol}{total_credits:,.2f}'], textposition='auto'),
                    go.Bar(name='Debits',  x=['Outflow'], y=[total_debits],  marker_color='#FF6B6B',
                           text=[f'{currency_symbol}{total_debits:,.2f}'], textposition='auto')
                ])
                flow_fig.update_layout(title="Cash Flow Comparison", yaxis_title=f"Amount ({currency_symbol})",
                                       barmode='group', height=400, showlegend=True)
                st.plotly_chart(flow_fig, use_container_width=True)

            with tab2:
                if len(df) > 1:
                    balance_fig = go.Figure()
                    balance_fig.add_trace(go.Scatter(
                        x=df["date"], y=df["balance"], mode="lines+markers", name="Balance",
                        line=dict(color='#4A90E2', width=3), marker=dict(size=6)
                    ))
                    balance_fig.update_layout(title="Account Balance Over Time", xaxis_title="Date",
                                              yaxis_title=f"Balance ({currency_symbol})", height=400)
                    st.plotly_chart(balance_fig, use_container_width=True)
                else:
                    st.info("📊 Balance trend requires more than one transaction")

            with tab3:
                if len(df) > 0:
                    desc_counts = df["description"].value_counts().head(10)
                    if not desc_counts.empty:
                        cat_fig = go.Figure(go.Bar(
                            x=desc_counts.values, y=desc_counts.index, orientation='h', marker_color='#9B59B6',
                            text=desc_counts.values, textposition='auto'
                        ))
                        cat_fig.update_layout(title="Top Transaction Types", xaxis_title="Frequency",
                                              yaxis_title="Transaction Type", height=400)
                        st.plotly_chart(cat_fig, use_container_width=True)

                    debit_count  = int((df["debit"]  > 0).sum())
                    credit_count = int((df["credit"] > 0).sum())
                    if debit_count or credit_count:
                        pie_fig = go.Figure(data=[go.Pie(
                            labels=['Debit Transactions', 'Credit Transactions'],
                            values=[debit_count, credit_count], hole=.3,
                            marker_colors=['#FF6B6B', '#00CC88']
                        )])
                        pie_fig.update_layout(title="Transaction Type Distribution", height=400)
                        st.plotly_chart(pie_fig, use_container_width=True)

            # Downloads
            st.subheader("💾 Export Data")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "📥 Download as CSV",
                    data=df.drop(columns=["parsed_date"]).to_csv(index=False),
                    file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with c2:
                st.download_button(
                    "📥 Download as JSON",
                    data=df.drop(columns=["parsed_date"]).to_json(orient='records'),
                    file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"❌ Error processing statement data: {e}")
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
        <p><small>Secure processing — Your data is not stored permanently</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
