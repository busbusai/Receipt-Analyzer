import io
import re
import csv
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from dateutil import parser as date_parser


INCOMPLETE = "[Incomplete]"

CURRENCY_SYMBOLS = {
    "EUR": ["€", "EUR", "EURO"],
    "USD": ["$", "USD"],
    "GBP": ["£", "GBP"],
    "TRY": ["₺", "TRY", "TL"],
}

CATEGORY_KEYWORDS = {
    "Office Supplies": ["office", "paper", "printer", "stationery", "staples", "büro"],
    "Travel": ["taxi", "uber", "flight", "train", "hotel", "booking", "fuel", "parking"],
    "Meals": ["restaurant", "cafe", "coffee", "meal", "food", "bar", "kebab", "pizza"],
    "Marketing": ["ads", "advert", "facebook", "google", "promotion", "campaign"],
}


@dataclass
class ReceiptResult:
    file_name: str
    vendor_name: str
    vendor_confidence: float
    purchase_date: str
    purchase_date_confidence: float
    currency: str
    currency_confidence: float
    gross_amount: str
    gross_amount_confidence: float
    vat_breakdown: str
    vat_breakdown_confidence: float
    net_amount: str
    net_amount_confidence: float
    category: str
    category_confidence: float


def normalize_amount(text: str) -> Optional[float]:
    cleaned = text.strip().replace(" ", "")
    if not cleaned:
        return None

    # Keep only numeric separators and digits.
    cleaned = re.sub(r"[^0-9,.-]", "", cleaned)
    if cleaned.count(",") > 1 and "." not in cleaned:
        return None

    # Handle European format like 1.234,56
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def format_amount(value: Optional[float]) -> str:
    if value is None:
        return INCOMPLETE
    return f"{value:.2f}"


def preprocess_image(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return sharpened


def configure_tesseract() -> None:
    env_cmd = os.getenv("TESSERACT_CMD", "").strip()
    candidates = [
        env_cmd,
        shutil.which("tesseract") or "",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]

    for cmd in candidates:
        if cmd and os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
            return


def is_tesseract_ready() -> Tuple[bool, str]:
    configure_tesseract()
    try:
        version = str(pytesseract.get_tesseract_version())
        return True, version
    except Exception:
        return False, ""


def ocr_with_confidence(img: Image.Image) -> Tuple[str, float]:
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confidences = []

    for i, raw_word in enumerate(data.get("text", [])):
        word = raw_word.strip()
        if not word:
            continue
        conf = data.get("conf", ["-1"])[i]
        try:
            conf_val = float(conf)
        except ValueError:
            conf_val = -1
        if conf_val >= 0:
            confidences.append(conf_val)

    # Keep line layout because VAT blocks are often spread across multiple lines.
    text = pytesseract.image_to_string(img)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    return text, avg_conf


def score_field(found: bool, ocr_conf: float, boost: float = 0.0) -> float:
    if not found:
        return 0.0
    score = max(0.0, min(99.0, ocr_conf + boost))
    return round(score, 1)


def maybe_incomplete(value: str, confidence: float, threshold: float = 40.0) -> str:
    if value == INCOMPLETE or confidence < threshold:
        return INCOMPLETE
    return value


def detect_currency(text: str) -> str:
    upper = text.upper()
    for code, symbols in CURRENCY_SYMBOLS.items():
        if any(sym.upper() in upper for sym in symbols):
            return code
    return "EUR"


def extract_vendor(text: str) -> str:
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if not lines:
        return INCOMPLETE

    # Pick first line with letters and no long numeric sequence.
    for line in lines[:5]:
        if re.search(r"[A-Za-zÄÖÜäöüß]", line) and not re.search(r"\d{5,}", line):
            return line[:80]
    return INCOMPLETE


def extract_date(text: str) -> str:
    date_patterns = [
        r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b",
        r"\b(\d{4}[./-]\d{2}[./-]\d{2})\b",
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                dt = date_parser.parse(m, dayfirst=True)
                return dt.strftime("%d.%m.%Y")
            except (ValueError, date_parser.ParserError):
                continue

    return INCOMPLETE


def extract_gross_amount(text: str) -> Optional[float]:
    patterns = [
        r"(?:total|summe|brutto|betrag|zu zahlen|gesamt)\s*[:]?\s*([0-9][0-9.,]*)",
        r"([0-9][0-9.,]*)\s*(?:€|EUR)",
    ]

    candidates = []
    lower = text.lower()
    for pattern in patterns:
        for m in re.finditer(pattern, lower, flags=re.IGNORECASE):
            val = normalize_amount(m.group(1))
            if val is not None:
                candidates.append(val)

    if not candidates:
        all_numbers = re.findall(r"\b\d+[.,]\d{2}\b", text)
        parsed = [normalize_amount(x) for x in all_numbers]
        parsed = [x for x in parsed if x is not None]
        return max(parsed) if parsed else None

    return max(candidates)


def extract_vat_breakdown(text: str) -> List[Tuple[float, float]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    breakdown: List[Tuple[float, float]] = []

    same_line_patterns = [
        re.compile(r"(?:mwst|vat|ust|tax)?\s*(\d{1,2})\s*%\s*[:]?\s*([0-9][0-9.,]*)", re.IGNORECASE),
        re.compile(r"([0-9][0-9.,]*)\s*(?:€|eur)?\s*(?:mwst|vat|ust|tax)\s*(\d{1,2})\s*%", re.IGNORECASE),
    ]

    for line in lines:
        for pattern in same_line_patterns:
            for m in pattern.finditer(line):
                if pattern is same_line_patterns[0]:
                    rate_raw, amount_raw = m.group(1), m.group(2)
                else:
                    amount_raw, rate_raw = m.group(1), m.group(2)
                amount = normalize_amount(amount_raw)
                if amount is not None:
                    breakdown.append((float(rate_raw), amount))

    rate_pattern = re.compile(r"(\d{1,2})\s*%")
    amount_pattern = re.compile(r"([0-9][0-9.,]*)")

    for i, line in enumerate(lines):
        rate_match = rate_pattern.search(line)
        if not rate_match:
            continue
        rate = float(rate_match.group(1))

        # Attempt multiline parsing: rate line followed by tax amount line.
        has_amount_same_line = any(p.search(line) for p in same_line_patterns)
        if has_amount_same_line:
            continue

        window = lines[i + 1 : i + 3]
        for next_line in window:
            if not re.search(r"mwst|vat|ust|tax|steuer|taxe|iva|tva", next_line, re.IGNORECASE) and not re.search(
                r"[0-9][0-9.,]*\s*(?:€|eur)?$", next_line, re.IGNORECASE
            ):
                continue
            amount_match = amount_pattern.search(next_line)
            if not amount_match:
                continue
            amount = normalize_amount(amount_match.group(1))
            if amount is not None:
                breakdown.append((rate, amount))
                break

    dedup: Dict[float, float] = {}
    for rate, amount in breakdown:
        dedup[rate] = max(dedup.get(rate, 0.0), amount)

    return sorted(dedup.items(), key=lambda x: x[0])


def categorize_expense(text: str) -> str:
    lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in lower)

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "Other"
    return best_category


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # Fallback when optional dependency 'tabulate' is unavailable.
        return df.to_string(index=False)


def to_csv_text(df: pd.DataFrame, delimiter: str = ",") -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(df.columns.tolist())
    for row in df.itertuples(index=False):
        writer.writerow(list(row))
    return output.getvalue()


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Receipts")
    return output.getvalue()


def analyze_receipt(file_name: str, img: Image.Image) -> ReceiptResult:
    processed = preprocess_image(img)
    text, confidence = ocr_with_confidence(processed)

    vendor = extract_vendor(text)
    purchase_date = extract_date(text)
    currency = detect_currency(text) or "EUR"
    gross = extract_gross_amount(text)
    vat_rows = extract_vat_breakdown(text)

    if vat_rows:
        vat_breakdown = "; ".join([f"{int(rate)}%: {amount:.2f}" for rate, amount in vat_rows])
        vat_total = sum(amount for _, amount in vat_rows)
    else:
        vat_breakdown = INCOMPLETE
        vat_total = None

    if gross is not None and vat_total is not None:
        net = max(gross - vat_total, 0)
    else:
        net = None

    category = categorize_expense(text)

    vendor_conf = score_field(vendor != INCOMPLETE, confidence, boost=10)
    date_conf = score_field(purchase_date != INCOMPLETE, confidence, boost=8)
    currency_conf = score_field(True, confidence, boost=4)
    gross_conf = score_field(gross is not None, confidence, boost=12)
    vat_conf = score_field(len(vat_rows) > 0, confidence, boost=10)
    net_conf = score_field(net is not None, confidence, boost=8)

    category_hits = sum(1 for kws in CATEGORY_KEYWORDS.values() for kw in kws if kw in text.lower())
    category_conf = score_field(category != "Other", confidence, boost=min(20.0, category_hits * 4.0))
    if category == "Other":
        category_conf = score_field(True, confidence, boost=-10)

    gross_str = format_amount(gross)
    net_str = format_amount(net)

    vendor = maybe_incomplete(vendor, vendor_conf)
    purchase_date = maybe_incomplete(purchase_date, date_conf)
    gross_str = maybe_incomplete(gross_str, gross_conf)
    vat_breakdown = maybe_incomplete(vat_breakdown, vat_conf)
    net_str = maybe_incomplete(net_str, net_conf)
    category = maybe_incomplete(category, category_conf)

    return ReceiptResult(
        file_name=file_name,
        vendor_name=vendor,
        vendor_confidence=vendor_conf,
        purchase_date=purchase_date,
        purchase_date_confidence=date_conf,
        currency=currency if currency else "EUR",
        currency_confidence=currency_conf,
        gross_amount=gross_str,
        gross_amount_confidence=gross_conf,
        vat_breakdown=vat_breakdown,
        vat_breakdown_confidence=vat_conf,
        net_amount=net_str,
        net_amount_confidence=net_conf,
        category=category,
        category_confidence=category_conf,
    )


def main() -> None:
    st.set_page_config(page_title="Receipt Analyzer", page_icon="🧾", layout="wide", initial_sidebar_state="collapsed")

    st.title("Receipt Analyzer")
    st.write(
        "Upload one or more receipts. The app extracts structured accounting fields and exports Markdown, CSV, and Excel."
    )
    st.caption("Mobile ready: On phone, use the camera option to take a receipt photo directly.")

    tesseract_ok, tesseract_version = is_tesseract_ready()
    if not tesseract_ok:
        st.error(
            "Tesseract OCR bulunamadi. macOS icin terminalde su komutu calistir: brew install tesseract"
        )
        st.info(
            "Kurulumdan sonra uygulamayi yeniden baslat: /Users/busragungor/fis_tarayıcı/.venv/bin/streamlit run /Users/busragungor/fis_tarayıcı/app.py"
        )
        return
    st.caption(f"OCR engine ready (Tesseract {tesseract_version}).")

    with st.expander("Quick Start: How to use + Excel/Google Sheets"):
        st.markdown(
            """
1. Upload one or more receipt images.
2. Wait for extraction results in the structured table.
3. Scroll down to see Markdown and CSV text outputs.
4. Use download buttons:
   - Download Excel (.xlsx) for Microsoft Excel.
   - Download CSV (comma) for Google Sheets and general tools.
   - Download CSV (semicolon) for regional Excel settings.

Google Sheets import:
- Open Google Sheets.
- File -> Import -> Upload -> select downloaded CSV.
- Choose Replace current sheet (or Insert new sheet).
            """
        )

    st.subheader("Add Receipt")
    input_mode = st.radio(
        "Choose input method",
        options=["Camera (Mobile)", "File Upload"],
        horizontal=True,
    )

    sources: List[Tuple[str, Image.Image]] = []

    if input_mode == "Camera (Mobile)":
        camera_file = st.camera_input("Take receipt photo")
        if camera_file is not None:
            camera_image = Image.open(io.BytesIO(camera_file.getvalue()))
            sources.append(("camera_capture.jpg", camera_image))
    else:
        uploaded_files = st.file_uploader(
            "Upload receipt image(s)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for file in uploaded_files:
                sources.append((file.name, Image.open(file)))

    if not sources:
        st.info("Add at least one receipt (camera or file upload).")
        return

    results = []
    for source_name, image in sources:
        result = analyze_receipt(source_name, image)
        results.append(result)

    df = pd.DataFrame(
        [
            {
                "File": r.file_name,
                "Vendor Name": r.vendor_name,
                "Vendor Confidence (%)": r.vendor_confidence,
                "Date of Purchase": r.purchase_date,
                "Date Confidence (%)": r.purchase_date_confidence,
                "Currency": r.currency,
                "Currency Confidence (%)": r.currency_confidence,
                "Total Gross Amount (Brutto)": r.gross_amount,
                "Gross Confidence (%)": r.gross_amount_confidence,
                "VAT (MwSt) Breakdown": r.vat_breakdown,
                "VAT Confidence (%)": r.vat_breakdown_confidence,
                "Net Amount (Netto)": r.net_amount,
                "Net Confidence (%)": r.net_amount_confidence,
                "Expense Category": r.category,
                "Category Confidence (%)": r.category_confidence,
            }
            for r in results
        ]
    )

    st.subheader("Structured Output")
    st.dataframe(df, use_container_width=True)

    md_text = to_markdown_table(df)
    csv_text = to_csv_text(df, delimiter=",")
    csv_text_excel_tr = to_csv_text(df, delimiter=";")
    xlsx_bytes = to_xlsx_bytes(df)

    st.subheader("Markdown Table")
    st.code(md_text, language="markdown")

    st.subheader("CSV")
    st.code(csv_text, language="csv")

    st.download_button(
        "Download CSV",
        data=csv_text,
        file_name=f"receipt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download CSV (Excel regional ; )",
        data=csv_text_excel_tr,
        file_name=f"receipt_analysis_excel_regional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download Excel (.xlsx)",
        data=xlsx_bytes,
        file_name=f"receipt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success("Export ready: You can open the .xlsx directly in Excel, or import the CSV into Google Sheets.")


if __name__ == "__main__":
    main()
