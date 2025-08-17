# Amharic E-commerce Data Extractor

## Project Overview

This project provides a comprehensive solution for extracting, processing, and analyzing e-commerce related data from Telegram channels, specifically focusing on Amharic content. It leverages natural language processing (NLP) techniques, including Named Entity Recognition (NER), to identify key entities like products, prices, and contact information from unstructured text. The extracted data is then cleaned, structured, and made available for further analysis or visualization through a Streamlit dashboard.

## Features

- **Telegram Data Scraper**: Extracts messages from specified Telegram channels, focusing on e-commerce related discussions.
- **Amharic Text Preprocessing**: Cleans and normalizes Amharic text, including handling of financial entities, contact information, and delivery mentions.
- **Named Entity Recognition (NER)**: Fine-tunes a multilingual transformer model (XLM-RoBERTa) to identify and classify e-commerce specific entities within Amharic text.
- **Data Interpretability**: Utilizes LIME and SHAP for understanding model predictions and feature importance.
- **Interactive Dashboard**: Provides a Streamlit-based dashboard for visualizing extracted data and model insights.
- **Vendor Scorecard**: (Implied by directory structure) A component for evaluating vendor performance based on extracted data.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/abeladamushumet/Amharic-Ecommerce-Data-Extractor.git
    cd Amharic-Ecommerce-Data-Extractor
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file specifies the necessary Python libraries, including `pandas`, `numpy`, `scikit-learn`, `transformers`, `torch`, `seqeval`, `shap`, `lime`, `streamlit`, `pyyaml`, and `pytest`.

4.  **Configure Telegram API access**:

    Obtain your `api_id` and `api_hash` from [my.telegram.org](https://my.telegram.org/). Then, update the `configs/config.yaml` file with your credentials:

    ```yaml
    telegram:
      api_id: YOUR_API_ID
      api_hash: "YOUR_API_HASH"
      output: "data/raw/messages.json"
    ```

## Usage

### 1. Data Ingestion (Scraping Telegram Channels)

To scrape messages from Telegram channels, run the `telegram_scraper.py` script. This script is configured to scrape from 5 randomly selected channels from a predefined list, or you can manually specify channels within the script.

```bash
python data_ingestion/telegram_scraper.py
```

The scraped data will be saved as `data/raw/messages.json`.

### 2. Data Preprocessing

After scraping, preprocess the raw data to clean and normalize the Amharic text and extract initial entities. This step is crucial for preparing the data for NER model training and further analysis.

```bash
python data_ingestion/preprocess.py
```

This will generate `data/processed/clean_messages.csv` (cleaned messages) and `data/processed/entities.json` (extracted entities).

### 3. Named Entity Recognition (NER) Model Training

The project includes a script to fine-tune a multilingual transformer model (XLM-RoBERTa) for Amharic NER. This requires a manually labeled dataset in CoNLL format (`data/processed/labeled_data.conll`).

```bash
python models/train_ner.py
```

The trained model will be saved in the `models/final_ner_model` directory.

### 4. Running the Dashboard

An interactive Streamlit dashboard is available for visualizing the extracted data and model insights. To run the dashboard:

```bash
streamlit run dashboard/app.py
```

This will open the dashboard in your web browser, typically at `http://localhost:8501`.

## Project Structure

```
Amharic-Ecommerce-Data-Extractor/
├── configs/                 # Configuration files (e.g., Telegram API credentials)
│   ├── config.yaml
│   └── training_config.yaml
├── data/                    # Stores raw and processed data
│   ├── processed/
│   │   ├── clean_messages.csv
│   │   ├── entities.json
│   │   └── labeled_data.conll
│   └── raw/
│       └── messages.json
├── data_ingestion/          # Scripts for data scraping and preprocessing
│   ├── preprocess.py
│   └── telegram_scraper.py
├── dashboard/               # Streamlit application for data visualization
│   └── app.py
├── interpretability/        # Scripts for model interpretability (LIME, SHAP)
│   ├── lime_explainer.py
│   └── shap_analysis.py
├── labeling/                # Tools for data labeling
│   └── label_formatter.py
├── models/                  # Scripts for NER model training, evaluation, and utilities
│   ├── evaluate.py
│   ├── model_utils.py
│   └── train_ner.py
├── tests/                   # Unit and integration tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_scorecard.py
├── vendor_scorecard/        # Components for vendor performance analysis
│   ├── scorecard.py
│   └── vendor_metrics.py
├── .github/                 # GitHub Actions workflows
├── .gitignore
├── LICENSE
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── session.session          # Telegram session file
└── setup.py                 # Project setup script
```

## Model Details (Named Entity Recognition)

The NER model is built upon `xlm-roberta-base`, a powerful multilingual transformer model. It is fine-tuned on a custom Amharic dataset to recognize specific e-commerce entities. The `models/train_ner.py` script handles the training process, including tokenization, alignment of labels, and evaluation using precision, recall, and F1-score metrics.

**Label Scheme:**

The model is trained to identify the following entity types:

-   `PRODUCT`: Commercial goods or services being offered.
-   `PRICE`: Monetary values associated with products.
-   `CONTACT`: Phone numbers, usernames, or email addresses for communication.
-   `LINK`: URLs or web addresses.
-   `ADDRESS`: Physical locations.
-   `PHONE`: Phone numbers (specifically identified if distinct from general CONTACT).

*(Note: The `train_ner.py` script defines a `get_label_list` function that includes `B-PRODUCT`, `I-PRODUCT`, `B-PRICE`, `I-PRICE`, `B-PCONTACT`, `I-CONTACT`, `B-LINK`, `I-LINK`, `B-ADDRESS`, `I-ADDRESS`, `B-PHONE`, `I-PHONE`. This indicates a BIO (Beginning, Inside, Outside) tagging scheme for robust entity recognition.)*

## Interpretability

To provide insights into how the NER model makes its predictions, the project incorporates interpretability techniques:

-   **LIME (Local Interpretable Model-agnostic Explanations)**: Used to explain individual predictions of the model by perturbing the input and observing changes in the output.
-   **SHAP (SHapley Additive exPlanations)**: Provides a unified approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using Shapley values from game theory.

These tools, located in the `interpretability/` directory, help in understanding which parts of the input text contribute most to the identification of specific entities.


## License

This project is licensed under the Apache License.
