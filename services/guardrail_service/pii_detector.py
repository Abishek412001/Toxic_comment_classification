"""
Enterprise PII Detection & Redaction Engine using High-Precision Regex & Pattern Matchers.
"""

import re
from typing import List, Tuple
from services.guardrail_service.schemas import PIIEntity, PIIMaskingResponse

PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "IP_ADDRESS": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "SSN_PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b|\b\d{3}-\d{2}-\d{4}\b",
}


class PIIDetector:
    """Enterprise PII Detection & Redaction Engine."""

    def detect_pii(self, text: str) -> List[PIIEntity]:
        """Detects PII entities within text string."""
        entities: List[PIIEntity] = []

        for entity_type, pattern in PII_PATTERNS.items():
            for match in re.finditer(pattern, text):
                val = match.group()

                # Basic validation filtering for credit card / phone length
                if entity_type == "CREDIT_CARD":
                    digits_only = re.sub(r"\D", "", val)
                    if len(digits_only) < 13 or len(digits_only) > 16:
                        continue

                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        value=val,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

        return entities

    def mask_pii(self, text: str, mask_label: str = "[REDACTED]") -> Tuple[str, List[PIIEntity]]:
        """Masks detected PII entities within text string."""
        entities = self.detect_pii(text)
        masked_text = text

        for entity_type, pattern in PII_PATTERNS.items():
            replacement = f"[{entity_type}_REDACTED]"
            masked_text = re.sub(pattern, replacement, masked_text)

        return masked_text, entities


pii_detector = PIIDetector()
