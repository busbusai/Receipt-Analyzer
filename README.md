# Receipt Analyzer App

This app analyzes uploaded receipt images and extracts structured accounting data.

## Features

- Detects vendor name (merchant)
- Extracts purchase date in `DD.MM.YYYY`
- Detects currency (defaults to `EUR`)
- Extracts gross total (`Brutto`)
- Detects VAT rates (`MwSt`) and amount per rate, including multi-line VAT blocks
- Calculates net amount (`Netto`)
- Categorizes expense type
- Computes confidence score per extracted field
- Marks uncertain or missing fields as `[Incomplete]`
- Outputs Markdown table, CSV, and Excel (`.xlsx`)

## Requirements

- Python 3.10+
- Tesseract OCR installed locally

macOS install:

```bash
brew install tesseract
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL in your browser and upload one or multiple receipt images.

## Easy Usage

1. Add receipt using one of these methods:
	- Camera (Mobile): take photo directly on phone
	- File Upload: select existing image(s)
2. Review extracted structured table.
3. Copy Markdown table or CSV text from the app.
4. Download one of these formats:
	- `.xlsx` for Microsoft Excel
	- `.csv` (comma) for Google Sheets and general use
	- `.csv` (semicolon) for regional Excel delimiter settings

## Mobile App-Like Usage

1. Deploy the app (Streamlit Cloud).
2. Open the app URL on your phone.
3. Use `Add to Home Screen` in your mobile browser.
4. Open the icon like an app and use `Camera (Mobile)` to capture receipts.

## Google Sheets Import

1. Open Google Sheets.
2. Go to `File -> Import -> Upload`.
3. Select the CSV file downloaded from the app.
4. Choose whether to replace current sheet or insert a new sheet.
