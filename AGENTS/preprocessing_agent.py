import re

class PreprocessingAgent:
    """
    Preprocessing Agent
    Responsible for cleaning, normalizing text data and preparing image data for downstream models.
    """
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Performs basic text cleanup (lowercasing, whitespace trimming, etc.)
        """
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_image(self, image_path: str):
        """
        Placeholder for image loading and preprocessing (resizing, normalization).
        """
        print(f"[PreprocessingAgent] Loading and preprocessing image from: {image_path}")
        # Image processing logic here
        return image_path

    def process(self, data: dict) -> dict:
        """
        Preprocesses incoming raw payload data.
        """
        processed_data = data.copy()
        if "text" in data:
            processed_data["text"] = self.clean_text(data["text"])
        if "image_path" in data:
            processed_data["image_path"] = self.preprocess_image(data["image_path"])
        return processed_data

if __name__ == "__main__":
    agent = PreprocessingAgent()
    sample_text = "   Hello!  My email is test@example.com.    "
    print("Cleaned text:", agent.process({"text": sample_text}))
