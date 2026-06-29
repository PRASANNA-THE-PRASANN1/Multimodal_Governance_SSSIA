class RiskAgent:
    """
    Risk Agent
    Assesses privacy risks, calculates risk scores based on type and volume of detected PII,
    and classifies risk levels (Low, Medium, High, Critical).
    """
    def __init__(self):
        # Risk weights for different types of PII
        self.risk_weights = {
            "NAME": 1.0,
            "EMAIL": 2.0,
            "PHONE": 2.0,
            "ADDRESS": 3.0,
            "IP_ADDRESS": 1.5,
            "SSN": 10.0,
            "CREDIT_CARD": 10.0,
            "PASSWORD": 8.0
        }

    def calculate_risk(self, pii_items: list) -> dict:
        """
        Calculates cumulative risk score and returns risk level classification.
        """
        score = 0.0
        for item in pii_items:
            item_type = item.get("type", "unknown").upper()
            weight = self.risk_weights.get(item_type, 1.0)
            score += weight

        if score == 0:
            level = "None"
        elif score <= 2.0:
            level = "Low"
        elif score <= 5.0:
            level = "Medium"
        elif score <= 10.0:
            level = "High"
        else:
            level = "Critical"

        return {
            "risk_score": score,
            "risk_level": level
        }

    def process(self, pii_items: list) -> dict:
        return self.calculate_risk(pii_items)

if __name__ == "__main__":
    agent = RiskAgent()
    detected_pii = [
        {"type": "NAME", "value": "Alice"},
        {"type": "SSN", "value": "000-12-3456"}
    ]
    print(agent.process(detected_pii))
