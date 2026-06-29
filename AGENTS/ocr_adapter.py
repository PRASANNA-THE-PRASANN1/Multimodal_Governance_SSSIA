"""
OCR Engine Adapter — Swappable OCR backend for the PIID pipeline.
=================================================================
Provides a unified interface for OCR engines (EasyOCR, PaddleOCR, Tesseract).
The classifier code calls only the abstract interface, never a specific engine.

Usage:
    from ocr_adapter import OCREngineFactory

    engine = OCREngineFactory.create("easyocr", languages=["en"])
    text, confidence = engine.run("path/to/image.png")
"""

import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def run(self, image_path: str) -> Tuple[str, float]:
        """
        Run OCR on a single image.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            Tuple of (extracted_text, mean_confidence).
            extracted_text: All detected text concatenated with spaces.
            mean_confidence: Mean confidence score across all detections (0.0–1.0).
                             Returns 0.0 if no text is detected.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the engine name for logging."""
        pass


class EasyOCREngine(OCREngine):
    """EasyOCR-based OCR engine (primary engine)."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = True):
        """
        Initialize EasyOCR reader.

        Args:
            languages: List of language codes (default: ["en"]).
            gpu: Whether to use GPU acceleration (default: True, falls back to CPU).
        """
        import easyocr
        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def run(self, image_path: str) -> Tuple[str, float]:
        """Run EasyOCR on a single image and return (text, confidence)."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.reader.readtext(image_path)

        if not results:
            return ("", 0.0)

        # Each result is (bbox, text, confidence)
        texts = []
        confidences = []
        for (_bbox, text, conf) in results:
            texts.append(text.strip())
            confidences.append(conf)

        concatenated_text = " ".join(texts)
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return (concatenated_text, round(mean_confidence, 4))

    def name(self) -> str:
        return "EasyOCR"


class PaddleOCREngine(OCREngine):
    """PaddleOCR-based OCR engine (future implementation)."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = True):
        raise NotImplementedError(
            "PaddleOCR engine is not yet implemented. "
            "Install paddleocr and implement this class when ready."
        )

    def run(self, image_path: str) -> Tuple[str, float]:
        raise NotImplementedError

    def name(self) -> str:
        return "PaddleOCR"


class TesseractOCREngine(OCREngine):
    """Tesseract-based OCR engine (future implementation)."""

    def __init__(self, languages: Optional[List[str]] = None, **kwargs):
        raise NotImplementedError(
            "Tesseract engine is not yet implemented. "
            "Install pytesseract and implement this class when ready."
        )

    def run(self, image_path: str) -> Tuple[str, float]:
        raise NotImplementedError

    def name(self) -> str:
        return "Tesseract"


class OCREngineFactory:
    """Factory for creating OCR engine instances."""

    _registry = {
        "easyocr": EasyOCREngine,
        "paddleocr": PaddleOCREngine,
        "tesseract": TesseractOCREngine,
    }

    @classmethod
    def create(cls, engine_name: str, **kwargs) -> OCREngine:
        """
        Create an OCR engine by name.

        Args:
            engine_name: One of "easyocr", "paddleocr", "tesseract".
            **kwargs: Passed to the engine constructor (e.g., languages, gpu).

        Returns:
            An initialized OCREngine instance.
        """
        engine_name = engine_name.lower().strip()
        if engine_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown OCR engine '{engine_name}'. Available: {available}"
            )
        return cls._registry[engine_name](**kwargs)

    @classmethod
    def available_engines(cls) -> List[str]:
        """Return list of registered engine names."""
        return list(cls._registry.keys())


if __name__ == "__main__":
    print(f"Available OCR engines: {OCREngineFactory.available_engines()}")
    print("Use OCREngineFactory.create('easyocr') to instantiate.")
