import unittest
from src.preprocessing.email_cleaner import EmailCleaner

class TestEmailCleaner(unittest.TestCase):
    def test_email_removal(self):
        cleaner = EmailCleaner(replacement_token="[EMAIL]")
        res = cleaner.transform("Contact user@example.com for support")
        self.assertIn("[EMAIL]", res)
        self.assertNotIn("user@example.com", res)

if __name__ == "__main__":
    unittest.main()
