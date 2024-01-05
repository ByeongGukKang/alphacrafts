
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import alphacraft

def encrypt_file(salt: int, password: str, file):
    """
    salt: 0~9999
    """
    with open(f"{alphacraft.location}\\data\\save\\salt_list.txt", 'r', encoding='utf-8') as f:
        salt_list = f.read().split("\n")    

    password = password.encode('utf-8')
    file = file.encode('utf-8')

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_list[salt].encode('utf-8'),
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))

    cipher_suite = Fernet(key)
    ciphertext = cipher_suite.encrypt(file)
    ciphertext = ciphertext.decode('utf-8')
    return ciphertext

def decrypt_file(salt: int, password: str, file):
    """
    salt: 0~9999
    """
    with open(f"{alphacraft.location}\\data\\save\\salt_list.txt", 'r', encoding='utf-8') as f:
        salt_list = f.read().split("\n")

    password = password.encode('utf-8')
    file = file.encode('utf-8')

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_list[salt].encode('utf-8'),
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))

    cipher_suite = Fernet(key)
    plaintext = cipher_suite.decrypt(file)
    plaintext = plaintext.decode('utf-8')
    return plaintext
