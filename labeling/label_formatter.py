"""
Enhanced Amharic E-commerce Labeling Tool
- Strictly follows the required CoNLL format
- Focuses on PRODUCT, PRICE, and LOCATION entities
- Provides auto-suggestions for common patterns
- Validates IOB tagging consistency
"""

import pandas as pd
from pathlib import Path
import re
from colorama import Fore, Style, init

# Initialize colors
init(autoreset=True)

# Configuration
INPUT_CSV = "data/processed/clean_messages.csv"
OUTPUT_CONLL = "data/processed/labeled_data.conll"
SAVE_EVERY = 5  # Save progress every 5 messages
MIN_LABELS = 30  # Minimum recommended labels
MAX_LABELS = 50  # Target maximum labels

# Entity tags as specified in requirements
ENTITY_TAGS = {
    'PRODUCT': ['B-PRODUCT', 'I-PRODUCT'],
    'PRICE': ['B-PRICE', 'I-PRICE'],
    'LOCATION': ['B-LOC', 'I-LOC'],
    'O': 'O'
}

# Patterns for auto-suggestion
PRICE_PATTERNS = [
    r'\d+\.?\d*\s?(ብር|ETB|birr)',
    r'በ\s?\d+\.?\d*\s?ብር',
    r'ዋጋ\s?\d+\.?\d*'
]

LOCATION_PATTERNS = [
    r'አዲስ\s+አበባ',
    r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # For English location names
    r'[ሀ-ፚ]+\s+[ሀ-ፚ]+'  # For Amharic location names
]

def suggest_label(token, prev_tag=""):
    """Suggest labels based on patterns and previous tag"""
    # Check for prices
    if any(re.search(pattern, token) for pattern in PRICE_PATTERNS):
        if prev_tag.endswith('PRICE'):
            return 'I-PRICE'
        return 'B-PRICE'
    
    # Check for locations
    if any(re.search(pattern, token) for pattern in LOCATION_PATTERNS):
        if prev_tag.endswith('LOC'):
            return 'I-LOC'
        return 'B-LOC'
    
    # Default to outside tag
    return 'O'

def validate_labels(labels):
    """Validate IOB tagging sequence"""
    errors = []
    for i, (current, next_tag) in enumerate(zip(labels, labels[1:] + ['O'])):
        # Check I- tags are preceded by B- or I- of same type
        if current.startswith('I-'):
            expected_prefix = current.replace('I-', 'B-')
            if i == 0 or (labels[i-1] != current and labels[i-1] != expected_prefix):
                errors.append(f"Invalid I- tag at position {i}: {current} without B-")
        
        # Check B- tags aren't followed by I- of different type
        if current.startswith('B-') and next_tag.startswith('I-'):
            if current[2:] != next_tag[2:]:
                errors.append(f"Invalid transition: {current} → {next_tag}")
    
    return errors

def print_labeling_guide():
    """Display the labeling instructions"""
    print(f"\n{Fore.YELLOW}LABELING GUIDE:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}B-PRODUCT{Style.RESET_ALL}: Start of product (e.g., 'Baby bottle')")
    print(f"{Fore.CYAN}I-PRODUCT{Style.RESET_ALL}: Continuation (e.g., 'bottle' in 'Baby bottle')")
    print(f"{Fore.CYAN}B-PRICE{Style.RESET_ALL}: Start of price (e.g., 'ዋጋ 1000 ብር')")
    print(f"{Fore.CYAN}I-PRICE{Style.RESET_ALL}: Price amount (e.g., '1000' in 'ዋጋ 1000 ብር')")
    print(f"{Fore.CYAN}B-LOC{Style.RESET_ALL}: Start of location (e.g., 'Addis abeba')")
    print(f"{Fore.CYAN}I-LOC{Style.RESET_ALL}: Continuation (e.g., 'abeba' in 'Addis abeba')")
    print(f"{Fore.CYAN}O{Style.RESET_ALL}: Outside any entity (default)\n")

def main():
    try:
        # Load data
        df = pd.read_csv(INPUT_CSV)
        text_column = 'text' if 'text' in df.columns else 'clean_text'
        
        labeled_data = []
        count = 0
        
        print(f"{Fore.GREEN}Amharic E-commerce Labeling Tool{Style.RESET_ALL}")
        print(f"Target: Label {MIN_LABELS}-{MAX_LABELS} messages\n")
        print_labeling_guide()

        for _, row in df.iterrows():
            if count >= MAX_LABELS:
                break
                
            text = str(row[text_column]).strip()
            if not text:
                continue
                
            tokens = text.split()
            labels = []
            print(f"\n{Fore.BLUE}Message {count+1}/{MAX_LABELS}{Style.RESET_ALL}")
            print(f"{text}\n")
            
            # Label each token
            for i, token in enumerate(tokens):
                prev_tag = labels[i-1] if i > 0 else ""
                suggestion = suggest_label(token, prev_tag)
                
                while True:
                    user_input = input(f"{i+1}/{len(tokens)} {token} [{suggestion}]: ").strip().upper()
                    label = user_input or suggestion
                    
                    # Validate input
                    valid_tags = []
                    for tags in ENTITY_TAGS.values():
                        if isinstance(tags, list):
                            valid_tags.extend(tags)
                        else:
                            valid_tags.append(tags)
                    
                    if label in valid_tags:
                        labels.append(label)
                        break
                    print(f"{Fore.RED}Invalid tag! Valid: {', '.join(valid_tags)}{Style.RESET_ALL}")

            # Validate the labeling
            if errors := validate_labels(labels):
                print(f"{Fore.RED}Labeling errors detected:{Style.RESET_ALL}")
                for err in errors:
                    print(f" - {err}")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    continue
            
            # Save in CoNLL format
            for token, label in zip(tokens, labels):
                labeled_data.append(f"{token}\t{label}")
            labeled_data.append("")  # Empty line between messages
            count += 1

            # Periodic save
            if count % SAVE_EVERY == 0:
                Path(OUTPUT_CONLL).parent.mkdir(exist_ok=True)
                with open(OUTPUT_CONLL, "w", encoding="utf-8") as f:
                    f.write("\n".join(labeled_data))
                print(f"{Fore.GREEN}Saved {count} messages{Style.RESET_ALL}")
                
        # Final save
        with open(OUTPUT_CONLL, "w", encoding="utf-8") as f:
            f.write("\n".join(labeled_data))
            
        print(f"\n{Fore.GREEN}✅ Done! Labeled {count} messages → {OUTPUT_CONLL}{Style.RESET_ALL}")
        if count < MIN_LABELS:
            print(f"{Fore.YELLOW}Note: Only {count} messages labeled (minimum {MIN_LABELS} recommended){Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()