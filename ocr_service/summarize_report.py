import os
from groq import Groq
from dotenv import load_dotenv

from .ocr_reader import read_text
from .doc_classifier import classify_document
from .lab_analyzer import analyze_lab_report

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def interpret_medical_report(image_path):

    # Step 1 — OCR
    raw_text = read_text(image_path)

    # Step 2 — Document Classification (NEW)
    doc_type = classify_document(raw_text)

    # If lab report, stop prescription interpretation
    if "lab" in doc_type:
        analysis = analyze_lab_report(raw_text)
        return {
            "raw_ocr": raw_text,
            "explanation": "📄 Document Type Detected: LAB REPORT\n\n" + analysis
        }

    if "other" in doc_type:
        return {
            "raw_ocr": raw_text,
            "explanation": (
                "📄 Document Type Detected: OTHER DOCUMENT\n\n"
                "The uploaded image does not appear to be a prescription."
            )
        }

    # Step 3 — Prescription Interpretation
    prompt = f"""
You are a clinical safety-critical prescription reader.

The OCR text below is extremely corrupted.
Your job is NOT to guess — your job is to FILTER.

Rules (strict):
1) Only output a medicine if partial readable letters exist
2) If unsure → do not include
3) Never assume antibiotics
4) Fewer medicines is safer than wrong medicines

Return format:

Confirmed Medicines:
- name | dosage | why matched

Rejected Guesses:
- name | why rejected

Patient Advice:
Tell patient to verify with doctor.

OCR TEXT:
{raw_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return {
        "raw_ocr": raw_text,
        "explanation": response.choices[0].message.content
    }