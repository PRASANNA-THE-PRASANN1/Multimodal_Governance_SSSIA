from preprocessing_agent import PreprocessingAgent
from vision_agent import VisionAgent
from prompt_agent import PromptAgent
from risk_agent import RiskAgent
from governance_agent import GovernanceAgent

class DecisionAgent:
    """
    Decision Agent
    Orchestrates the entire multi-agent PII workflow.
    Aggregates responses and produces the final actionable decision.
    """
    def __init__(self, rules_path: str = None):
        self.preprocessor = PreprocessingAgent()
        self.vision = VisionAgent()
        self.prompt = PromptAgent()
        self.risk = RiskAgent()
        self.governance = GovernanceAgent(rules_path=rules_path)

    def redact_text(self, text: str, pii_items: list) -> str:
        """
        Simple redact helper that replaces detected PII values with [REDACTED].
        """
        redacted = text
        # Sort items by start index descending to prevent offset shifts
        sorted_items = sorted(pii_items, key=lambda x: x.get("start_index", 0), reverse=True)
        
        for item in sorted_items:
            val = item.get("value")
            if val and val in redacted:
                redacted = redacted.replace(val, f"[{item.get('type')}]")
        return redacted

    def evaluate_payload(self, raw_payload: dict) -> dict:
        """
        Orchestrates full analysis of text and/or images.
        """
        # 1. Preprocessing
        preprocessed = self.preprocessor.process(raw_payload)
        text = preprocessed.get("text", "")
        image_path = preprocessed.get("image_path", None)

        pii_items = []
        vision_results = {}

        # 2. Vision analysis (if image is present)
        if image_path:
            vision_results = self.vision.process(image_path)
            # Add text extracted via OCR back into main processing pipeline
            ocr_text = vision_results.get("extracted_text", "")
            if ocr_text:
                text += " " + ocr_text

        # 3. Prompt/LLM PII Identification
        if text:
            prompt_results = self.prompt.process(text)
            pii_items.extend(prompt_results.get("pii_items", []))

        # 4. Risk Assessment
        risk_info = self.risk.process(pii_items)

        # 5. Governance Evaluation
        gov_info = self.governance.process(risk_info, pii_items)

        # 6. Redaction
        redacted_text = self.redact_text(text, pii_items) if text else ""

        # Final decision payload
        return {
            "decision": gov_info.get("governance_action", "ALLOW"),
            "risk_evaluation": risk_info,
            "governance_evaluation": gov_info,
            "detected_pii_count": len(pii_items),
            "pii_items": pii_items,
            "redacted_text": redacted_text,
            "vision_metadata": vision_results
        }

if __name__ == "__main__":
    agent = DecisionAgent()
    payload = {
        "text": "Call me at 555-0199 or email user@google.com."
    }
    decision = agent.evaluate_payload(payload)
    print("Decision output:", decision)
