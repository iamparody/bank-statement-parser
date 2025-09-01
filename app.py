import streamlit as st
from llama_cloud_services import LlamaExtract
from llama_cloud_services.extract import ExtractConfig, ExtractTarget, ExtractMode
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

# Define schema for a single transaction
class Transaction(BaseModel):
    date: str = Field(description="Date of the transaction")
    description: str = Field(description="Name or type of the transaction")
    debit: str = Field(description="Outflow transactions")
    credit: str = Field(description="Inflow transactions")
    balance: str = Field(description="Remaining balance after the transaction")

# Define schema for a document with multiple transactions
class Statement(BaseModel):
    transactions: List[Transaction] = Field(description="List of transaction entries")

# Streamlit app UI
st.title("🏦 Bank Statement Parser")

# Get API key from Streamlit secrets
api_key = st.secrets["LLAMA_API_KEY"]

# Initialize the extractor and safely get/create the agent
@st.cache_resource
def get_agent():
    extractor = LlamaExtract(api_key=api_key)

    config = ExtractConfig(
        extraction_target=ExtractTarget.PER_DOC,
        extraction_mode=ExtractMode.MULTIMODAL
    )

    agent_name = "bank_statement_parser_final"

    try:
        # Try to create agent
        agent = extractor.create_agent(
            name=agent_name,
            data_schema=Statement,
            config=config
        )
    except Exception as e:
        if "already exists" in str(e):
            # Reuse existing agent
            agent = extractor.get_agent_by_name(agent_name)
        else:
            raise e

    return agent

agent = get_agent()

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your bank statement PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("🔍 Extracting transactions..."):
        # Save uploaded PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract the data
        result = agent.extract("temp.pdf")
        extracted_data = result.data

        if extracted_data and "transactions" in extracted_data:
            statement = Statement.model_validate(extracted_data)

            st.success(f"✅ Found {len(statement.transactions)} transactions!")

            # Convert to DataFrame for display and charts
            df = pd.DataFrame([txn.model_dump() for txn in statement.transactions])

            # Clean numeric columns to floats for plotting
            for col in ["debit", "credit", "balance"]:
                df[col] = df[col].str.replace(",", "").fillna("0").astype(float)

            st.subheader("📊 Transactions Table")
            st.dataframe(df)

            # Summary
            total_debit = df["debit"].sum()
            total_credit = df["credit"].sum()
            closing_balance = df["balance"].iloc[-1]

            st.subheader("📌 Summary")
            st.write(f"- Total Debit: **${total_debit:,.2f}**")
            st.write(f"- Total Credit: **${total_credit:,.2f}**")
            st.write(f"- Closing Balance: **${closing_balance:,.2f}**")

            # Charts
            st.subheader("📈 Debit vs Credit")
            st.bar_chart(data=df[["debit", "credit"]].sum().to_frame().T)

        else:
            st.warning("⚠️ No transactions found in the uploaded PDF.")
