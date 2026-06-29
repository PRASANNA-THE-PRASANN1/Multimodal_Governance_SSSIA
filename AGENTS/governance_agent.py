import json
import os

class GovernanceAgent:
    """
    Governance Agent
    Applies organizational rules and regulatory standards (GDPR, HIPAA, CCPA)
    to decide on data sharing, consent, and storage policies.
    """
    def __init__(self, rules_path: str = None):
        self.rules = {}
        if rules_path and os.path.exists(rules_path):
            self.load_rules(rules_path)
        else:
            # Default fallbacks
            self.rules = {
                "block_critical_risk": True,
                "allow_redacted": True,
                "requires_consent": ["EMAIL", "PHONE", "ADDRESS"]
            }

    def load_rules(self, rules_path: str):
        try:
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)
            print(f"[GovernanceAgent] Rules loaded from {rules_path}")
        except Exception as e:
            print(f"[GovernanceAgent] Error loading rules: {e}")

    def evaluate_compliance(self, risk_info: dict, pii_items: list) -> dict:
        """
        Evaluates detected PII and risk against governance rules.
        """
        risk_level = risk_info.get("risk_level", "None")
        risk_score = risk_info.get("risk_score", 0.0)

        action = "ALLOW"
        reasons = []

        if self.rules.get("block_critical_risk") and risk_level in ["Critical", "High"]:
            action = "BLOCK"
            reasons.append(f"Risk level is {risk_level} which exceeds allowable threshold.")

        for item in pii_items:
            item_type = item.get("type", "").upper()
            if item_type in self.rules.get("requires_consent", []):
                reasons.append(f"Consent check required for field: {item_type}")
                if action != "BLOCK":
                    action = "SUSPEND_UNTIL_CONSENT"

        return {
            "governance_action": action,
            "reasons": reasons,
            "compliant": action == "ALLOW"
        }

    def process(self, risk_info: dict, pii_items: list) -> dict:
        return self.evaluate_compliance(risk_info, pii_items)

if __name__ == "__main__":
    agent = GovernanceAgent()
    print(agent.process({"risk_level": "High", "risk_score": 11.0}, [{"type": "SSN"}]))
