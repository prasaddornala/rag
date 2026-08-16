from docx import Document
import sys

if len(sys.argv) < 2:
    print("Usage: extract_docx.py <docx-path> <out-path (optional)>")
    sys.exit(1)

docx_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/week5_requirements.txt"

doc = Document(docx_path)
with open(out_path, "w", encoding="utf-8") as f:
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            f.write(text + "\n")
print(f"Wrote extracted text to {out_path}")
