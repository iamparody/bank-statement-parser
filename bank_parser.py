import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def extract_transactions(text: str) -> List[Dict]:
    """Extract transactions from bank statement text"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    transactions = []
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',
    ]
    
    # Skip keywords
    skip_keywords = ['txn date', 'balance', 'money out', 'money in', 'description', 
                    'your transactions', 'opening balance', 'closing balance', 'total',
                    'account', 'statement', 'period', 'branch']
    
    for line in lines:
        # Skip header/summary lines
        if any(keyword in line.lower() for keyword in skip_keywords):
            continue
            
        # Find date
        date_match = None
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                date_str = match.group()
                # Try different formats
                for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y-%m-%d']:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        date_match = parsed_date.strftime('%Y-%m-%d')
                        break
                    except:
                        continue
                if date_match:
                    break
        
        if not date_match:
            continue
            
        # Find amounts - look for numbers that could be money
        amount_matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', line)
        amounts = []
        
        for match in amount_matches:
            try:
                clean_amount = match.replace(',', '')
                amount_val = float(clean_amount)
                if 0.01 <= amount_val <= 10000000:  # Reasonable range
                    amounts.append(amount_val)
            except:
                continue
        
        if len(amounts) < 1:
            continue
            
        # Extract description
        desc_line = line
        desc_line = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', desc_line)
        for match in amount_matches:
            desc_line = desc_line.replace(match, '')
        description = ' '.join(desc_line.split()).strip()
        
        if not description or len(description) < 3:
            continue
            
        # Determine transaction structure
        debit = 0.0
        credit = 0.0
        balance = amounts[-1] if amounts else 0.0  # Last amount usually balance
        
        # Logic for debit/credit based on description context
        credit_keywords = ['deposit', 'credit', 'salary', 'clv', 'incoming', 'received', 'transfer in']
        debit_keywords = ['payment', 'withdrawal', 'fee', 'charge', 'bill', 'purchase']
        
        if len(amounts) >= 2:
            if any(kw in description.lower() for kw in credit_keywords):
                credit = amounts[0] if len(amounts) >= 2 else amounts[0]
                balance = amounts[1] if len(amounts) >= 2 else amounts[0]
            else:
                debit = amounts[0] if len(amounts) >= 2 else amounts[0]
                balance = amounts[1] if len(amounts) >= 2 else amounts[0]
        elif len(amounts) == 1:
            # Single amount - determine type from context
            if any(kw in description.lower() for kw in credit_keywords):
                credit = amounts[0]
            elif any(kw in description.lower() for kw in debit_keywords):
                debit = amounts[0]
            else:
                # Default: if large amount, likely balance; otherwise debit
                if amounts[0] > 10000:
                    balance = amounts[0]
                else:
                    debit = amounts[0]
        
        transaction = {
            "date": date_match,
            "description": description,
            "debit": str(debit),
            "credit": str(credit),
            "balance": str(balance)
        }
        
        transactions.append(transaction)
    
    return transactions
