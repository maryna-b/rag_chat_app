"""
Fix encoding for main.py file specifically.
"""

def fix_main_py():
    """Fix encoding for main.py file."""
    file_path = "main.py"
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    print(f"Fixing encoding for: {file_path}")
    
    for encoding in encodings_to_try:
        try:
            # Try to read with this encoding
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # If successful, write back as UTF-8
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[OK] Successfully converted from {encoding} to UTF-8")
            return True
            
        except UnicodeDecodeError:
            print(f"[FAIL] Failed with {encoding}")
            continue
        except Exception as e:
            print(f"[ERROR] Error with {encoding}: {e}")
            continue
    
    print(f"[ERROR] Could not read {file_path} with any encoding")
    return False

if __name__ == "__main__":
    fix_main_py()