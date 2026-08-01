"""
Adversarial Prompt Injection & Jailbreak Attack Detector Engine.
"""

import re
from typing import Tuple, Optional

INJECTION_PATTERNS = [
    (r"ignore (all )?previous instructions", "IGNORE_PREVIOUS_INSTRUCTIONS"),
    (r"system prompt (leak|show|display|reveal)", "SYSTEM_PROMPT_LEAK"),
    (r"disregard (all )?rules", "RULE_DISREGARD"),
    (r"you are now in (dan|developer) mode", "DAN_JAILBREAK"),
    (r"pretend to be an unrestricted", "ROLE_PLAY_MANIPULATION"),
    (r"<script>.*?</script>", "HTML_SCRIPT_INJECTION"),
    (r"\[SYSTEM NOTE:", "HIDDEN_DELIMITER_INJECTION"),
]


class InjectionDetector:
    """Detects adversarial prompt injections and jailbreak attacks."""

    def detect_injection_or_jailbreak(self, prompt: str) -> Tuple[bool, Optional[str], bool, Optional[str]]:
        """Returns (has_injection, injection_type, has_jailbreak, jailbreak_type)."""
        prompt_lower = prompt.lower()

        has_injection = False
        inj_type = None
        has_jailbreak = False
        jb_type = None

        for pattern, attack_name in INJECTION_PATTERNS:
            if re.search(pattern, prompt_lower):
                if "dan" in attack_name.lower() or "role" in attack_name.lower():
                    has_jailbreak = True
                    jb_type = attack_name
                else:
                    has_injection = True
                    inj_type = attack_name

        return has_injection, inj_type, has_jailbreak, jb_type


injection_detector = InjectionDetector()
