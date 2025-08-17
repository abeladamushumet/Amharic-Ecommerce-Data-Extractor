"""
Enhanced Telegram Message Preprocessor for Amharic E-commerce Data
- Special handling for financial entities (prices, contacts)
- Improved Amharic text normalization
- Metadata preservation for vendor analysis
"""

import json
import re
import pandas as pd
from pathlib import Path
import emoji
from typing import Dict, Any

# Configuration
RAW_PATH = "data/raw/messages.json"
CLEAN_PATH = "data/processed/clean_messages.csv"
ENTITIES_PATH = "data/processed/entities.json"

class AmharicPreprocessor:
    def __init__(self):
        # Regex patterns for financial entities
        self.price_pattern = re.compile(
            r"(ዋጋ|ብር|birr|price)[:\s]*([\d,]+\.?\d*)", 
            re.IGNORECASE
        )
        self.contact_pattern = re.compile(
            r"(09\d{8}|@[\w]+|[\w\.]+@[\w\.]+)", 
            re.UNICODE
        )

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract financial entities from raw text"""
        entities = {
            "prices": [],
            "contacts": [],
            "has_delivery": False
        }

        # Price extraction
        for match in self.price_pattern.finditer(text):
            amount = match.group(2).replace(",", "")
            entities["prices"].append(float(amount))
        
        # Contact extraction
        entities["contacts"] = self.contact_pattern.findall(text)
        
        # Delivery mentions
        entities["has_delivery"] = bool(re.search(
            r"(መላክ|delivery|ገቢያ ቦታ)", 
            text, 
            re.IGNORECASE
        ))
        
        return entities

    def normalize_amharic(self, text: str) -> str:
        """Enhanced Amharic normalization"""
        # Standardize whitespace and punctuation
        text = re.sub(r"[፡።፣]", " ", text)  # Replace Ethiopic punctuation
        text = re.sub(r"\s+", " ", text).strip()
        
        # Common substitutions
        replacements = {
            r"ሩ": "ር",  # Common typo normalization
            r"ሉ": "ል",
            r"\bብር\b": "ETB"  # Standardize currency
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
            
        return text

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline"""
        if not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.strip()
        text = emoji.replace_emoji(text, replace="")  # Better emoji handling
        text = self.normalize_amharic(text)
        
        return text

def main():
    # Initialize preprocessor
    processor = AmharicPreprocessor()
    
    # Load raw JSON
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Process data
    cleaned_data = []
    entity_records = []
    
    for msg in raw_data:
        # Clean main text
        clean_text = processor.clean_text(msg["text"])
        
        # Skip empty messages
        if not clean_text:
            continue
            
        # Extract entities
        entities = processor.extract_entities(msg["text"])
        
        # Build cleaned record
        cleaned_record = {
            "id": msg["message_id"],
            "channel": msg["channel"],
            "date": msg["date"],
            "views": msg.get("views", 0),
            "text": clean_text,
            "word_count": len(clean_text.split())
        }
        
        # Build entity record
        entity_record = {
            "message_id": msg["message_id"],
            "channel": msg["channel"],
            **entities
        }
        
        cleaned_data.append(cleaned_record)
        entity_records.append(entity_record)

    # Save cleaned messages
    df = pd.DataFrame(cleaned_data)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
    
    # Save extracted entities
    with open(ENTITIES_PATH, "w", encoding="utf-8") as f:
        json.dump(entity_records, f, ensure_ascii=False, indent=2)
    
    print(f"""
    Preprocessing Complete!
    → Cleaned messages: {CLEAN_PATH} ({len(df)} records)
    → Extracted entities: {ENTITIES_PATH}
    → Avg price mentions per message: {df['text'].apply(lambda x: len(processor.price_pattern.findall(x))).mean():.1f}
    """)

if __name__ == "__main__":
    main()