# 🧪 Tests: Add tests for OCR image content processing

## 🎯 What
Added comprehensive unit tests for `OCRServer._process_image_content` in `src/ai_workers/workers/ocr.py`. This method parses the OpenAI-compatible content array to extract text prompts and image URLs.

## 📊 Coverage
The new tests in `tests/test_ocr.py` cover:
- Single text extraction
- Single image URL extraction
- Combined text and image extraction
- Empty text handling
- Multiple text/image parts (verifying "last write wins" behavior)
- Unknown content types
- Missing fields in content parts

## ✨ Result
Improved test coverage for the OCR worker logic, ensuring robust handling of various input formats.
