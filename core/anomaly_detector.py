class AnomalyDetector:
    def __init__(self):
        self.anomaly_keywords = ["sospechoso", "arma", "fuego", "intruso", "caida", "violencia", "peligro"]

    def evaluate(self, vlm_text, person_count):
        is_anomaly = False
        classification = "NORMAL"

        text_lower = vlm_text.lower()
        if any(keyword in text_lower for keyword in self.anomaly_keywords):
            is_anomaly = True
            classification = "ANOMALY_DETECTED"
        elif person_count > 3:
            is_anomaly = True
            classification = "CROWD_ANOMALY"
        elif person_count > 0:
            classification = "PERSON_PRESENCE"

        return is_anomaly, classification
