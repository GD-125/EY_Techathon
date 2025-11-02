"""
Encryption Service Module
Provides AES-256 encryption/decryption for sensitive data
Complies with PCI-DSS and GDPR requirements
"""

import base64
import hashlib
import secrets
from typing import Union
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


class CryptoService:
    """
    AES-256-CBC Encryption Service
    Provides encryption/decryption for PII and sensitive data
    """

    def __init__(self, encryption_key: str, salt: str):
        """
        Initialize crypto service with encryption key

        Args:
            encryption_key: 32-character encryption key
            salt: Salt value for key derivation
        """
        self.salt = salt.encode()

        # Derive a proper 32-byte key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        self.key = kdf.derive(encryption_key.encode())

    def encrypt(self, plaintext: Union[str, bytes]) -> str:
        """
        Encrypt plaintext using AES-256-CBC

        Args:
            plaintext: Data to encrypt (string or bytes)

        Returns:
            Base64-encoded encrypted data with IV prepended
        """
        # Convert string to bytes if needed
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        # Generate random IV (Initialization Vector)
        iv = secrets.token_bytes(16)

        # Pad the plaintext to block size (128 bits = 16 bytes)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Create cipher and encrypt
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Prepend IV to ciphertext and encode as base64
        encrypted_data = iv + ciphertext
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext using AES-256-CBC

        Args:
            ciphertext: Base64-encoded encrypted data

        Returns:
            Decrypted plaintext string
        """
        try:
            # Decode base64
            encrypted_data = base64.b64decode(ciphertext.encode('utf-8'))

            # Extract IV and ciphertext
            iv = encrypted_data[:16]
            actual_ciphertext = encrypted_data[16:]

            # Create cipher and decrypt
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()

            # Remove padding
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

            return plaintext.decode('utf-8')

        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256

        Args:
            password: Plain password

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(
            (password + self.salt.decode()).encode('utf-8')
        ).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify password against hash

        Args:
            password: Plain password
            hashed: Hashed password

        Returns:
            True if password matches
        """
        return self.hash_password(password) == hashed

    def encrypt_dict(self, data: dict, fields_to_encrypt: list) -> dict:
        """
        Encrypt specific fields in a dictionary

        Args:
            data: Dictionary containing data
            fields_to_encrypt: List of field names to encrypt

        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()

        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
                encrypted_data[f"{field}_encrypted"] = True

        return encrypted_data

    def decrypt_dict(self, data: dict, fields_to_decrypt: list) -> dict:
        """
        Decrypt specific fields in a dictionary

        Args:
            data: Dictionary containing encrypted data
            fields_to_decrypt: List of field names to decrypt

        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()

        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data.get(f"{field}_encrypted"):
                try:
                    decrypted_data[field] = self.decrypt(decrypted_data[field])
                    decrypted_data.pop(f"{field}_encrypted", None)
                except Exception:
                    # If decryption fails, leave encrypted
                    pass

        return decrypted_data

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate cryptographically secure random token

        Args:
            length: Token length in bytes

        Returns:
            Hexadecimal token string
        """
        return secrets.token_hex(length)

    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """
        Mask sensitive data for logging/display

        Args:
            data: Sensitive string (e.g., PAN, phone)
            visible_chars: Number of characters to keep visible

        Returns:
            Masked string
        """
        if not data or len(data) <= visible_chars:
            return "*" * len(data) if data else ""

        return data[:visible_chars] + "*" * (len(data) - visible_chars)


# Sensitive fields that should always be encrypted
SENSITIVE_FIELDS = [
    "pan",
    "aadhaar",
    "account_number",
    "credit_score",
    "salary",
    "monthly_salary",
    "annual_income",
    "address",
    "email",
    "phone",
    "date_of_birth",
    "bank_account",
    "ifsc_code",
]
