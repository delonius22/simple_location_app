"""
Unit tests for the RSA-TOTP encryption system configured for Windows desktop usage.
"""

import os
import json
import unittest
import shutil
from pathlib import Path

# Import modules from our package
from src.main import RSATOTPCrypto
from src.key_manager import KeyManager
from src.qr_generator import TOTPQRGenerator
import pyotp


class TestRSATOTPEncryptionWindows(unittest.TestCase):
    """Test the RSA-TOTP encryption system using a standard Windows desktop location."""
    
    def setUp(self):
        """Set up the test environment on Windows desktop."""
        # Define test directory on Windows desktop
        self.desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.test_dir_name = "RSA_TOTP_Test"
        self.test_dir = os.path.join(self.desktop_path, self.test_dir_name)
        
        # Create test directory if it doesn't exist
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize the crypto system
        self.crypto = RSATOTPCrypto(str(self.config_dir))
        
        # Initialize with a test issuer and user
        self.issuer = "TestCompany"
        self.user = "test@example.com"
        self.qr_path = self.crypto.initialize_system(self.issuer, self.user)
        
        # Get the TOTP secret for verification
        totp_config_path = self.config_dir / "totp_secret.json"
        with open(totp_config_path, 'r') as f:
            self.totp_config = json.load(f)
            self.totp_secret = self.totp_config["secret"]
        
        # Test data
        self.test_data = {
            "id": 12345,
            "name": "Test User",
            "email": "test@example.com",
            "metadata": {
                "created": "2025-04-06T12:00:00Z",
                "last_login": "2025-04-06T13:00:00Z"
            },
            "sensitive_data": "This is confidential information that needs to be protected"
        }
        
        print(f"Test environment set up on desktop at: {self.test_dir}")
        print(f"TOTP QR code generated at: {self.qr_path}")
        print("You can scan this QR code with an authenticator app for manual testing")
    
    def tearDown(self):
        """Clean up after tests - optionally comment out to keep test files."""
        # Uncomment this line to clean up test files automatically
        # shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Keep test files by default for manual verification
        print(f"Test files preserved at: {self.test_dir}")
        print("You can manually delete this directory when no longer needed")
    
    def test_key_generation(self):
        """Test RSA key generation."""
        key_manager = KeyManager(self.config_dir)
        version = key_manager.get_current_key_version()
        
        self.assertIsNotNone(version)
        self.assertTrue(version.startswith("rsa-4096"))
        
        private_key = key_manager.load_private_key()
        public_key = key_manager.load_public_key()
        
        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)
        
        print(f"Key generation test passed. Key version: {version}")
    
    def test_qr_generation(self):
        """Test TOTP QR code generation."""
        self.assertTrue(os.path.exists(self.qr_path))
        print(f"QR code generation test passed. QR saved at: {self.qr_path}")
    
    def test_encryption_decryption(self):
        """Test the encryption and decryption process."""
        # Encrypt the test data
        encrypted_data = self.crypto.encrypt_json(self.test_data)
        
        # Save encrypted data to a file for manual inspection
        encrypted_file_path = Path(self.test_dir) / "encrypted_test_data.json"
        with open(encrypted_file_path, 'w') as f:
            json.dump(encrypted_data, f, indent=2)
        
        # Verify the encrypted data structure
        self.assertIn("ciphertext", encrypted_data)
        self.assertIn("iv", encrypted_data)
        self.assertIn("encrypted_key", encrypted_data)
        self.assertIn("metadata", encrypted_data)
        
        # Generate a valid TOTP code
        totp = pyotp.TOTP(
            self.totp_secret,
            digits=self.totp_config["digits"],
            interval=self.totp_config["interval"]
        )
        valid_totp = totp.now()
        
        print(f"Current valid TOTP code: {valid_totp}")
        print("You can use this code for manual testing in the next 90 seconds")
        
        # Decrypt the data with a valid TOTP code
        decrypted_data = self.crypto.decrypt_json(encrypted_data, valid_totp)
        
        # Save decrypted data to a file for manual inspection
        decrypted_file_path = Path(self.test_dir) / "decrypted_test_data.json"
        with open(decrypted_file_path, 'w') as f:
            json.dump(decrypted_data, f, indent=2)
        
        # Verify the decryption was successful
        self.assertIsNotNone(decrypted_data)
        self.assertEqual(decrypted_data, self.test_data)
        
        # Try decryption with an invalid TOTP code
        invalid_totp = "12345678"  # 8 digits but invalid
        failed_decryption = self.crypto.decrypt_json(encrypted_data, invalid_totp)
        
        # Verify the decryption failed
        self.assertIsNone(failed_decryption)
        
        print("Encryption/decryption test passed")
        print(f"Encrypted data saved to: {encrypted_file_path}")
        print(f"Decrypted data saved to: {decrypted_file_path}")
    
    def test_manual_workflow(self):
        """
        Set up files for manual testing of the full workflow.
        This test doesn't assert anything but prepares files for manual verification.
        """
        # Create a sample JSON file
        sample_data = {
            "project_id": "PRJ-2025-042",
            "client": "Strategic Client Inc.",
            "confidential": True,
            "budget": 250000,
            "team_members": [
                {"name": "Alice Smith", "role": "Project Lead"},
                {"name": "Bob Johnson", "role": "Developer"},
                {"name": "Carol Davis", "role": "Analyst"}
            ],
            "notes": "This is a test file for manual verification."
        }
        
        sample_file_path = Path(self.test_dir) / "sample_data.json"
        with open(sample_file_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        # Encrypt the sample data
        encrypted_data = self.crypto.encrypt_json(sample_data)
        
        encrypted_file_path = Path(self.test_dir) / "encrypted_sample.json"
        with open(encrypted_file_path, 'w') as f:
            json.dump(encrypted_data, f, indent=2)
        
        # Generate a current valid TOTP code
        totp = pyotp.TOTP(
            self.totp_secret,
            digits=self.totp_config["digits"],
            interval=self.totp_config["interval"]
        )
        valid_totp = totp.now()
        
        # Create a README with instructions for manual testing
        readme_path = Path(self.test_dir) / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("RSA-TOTP Encryption System Manual Test Files\n")
            f.write("===========================================\n\n")
            f.write("This directory contains files for manually testing the encryption system.\n\n")
            
            f.write("QR Code for Authenticator App:\n")
            f.write(f"  {self.qr_path}\n")
            f.write("  Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)\n\n")
            
            f.write("Sample Files:\n")
            f.write(f"  Original data: {sample_file_path}\n")
            f.write(f"  Encrypted data: {encrypted_file_path}\n\n")
            
            f.write("Current TOTP Code (valid briefly):\n")
            f.write(f"  {valid_totp}\n\n")
            
            f.write("Manual Test Commands:\n")
            f.write("  # To decrypt using command line:\n")
            f.write(f"  python -m src.main decrypt {encrypted_file_path} --output decrypted_output.json --totp YOUR_CURRENT_TOTP_CODE\n\n")
            
            f.write("  # To encrypt a new file:\n")
            f.write(f"  python -m src.main encrypt your_data.json --output new_encrypted.json\n\n")
            
            f.write("Note: These test files will remain on your desktop for manual verification.\n")
            f.write("You can delete this directory when you've completed testing.\n")
        
        print("Manual test files prepared:")
        print(f"  README with instructions: {readme_path}")
        print(f"  Sample data: {sample_file_path}")
        print(f"  Encrypted sample: {encrypted_file_path}")
        print(f"  Current TOTP code: {valid_totp} (valid for a short time)")


if __name__ == "__main__":
    unittest.main()