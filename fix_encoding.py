"""
Script to fix text encoding issues in document files.
"""
import os

def fix_encoding(file_path):
    """Try to read file with different encodings and convert to UTF-8."""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    print(f"Processing: {file_path}")
    
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

def main():
    """Fix encoding for all text files in the data directory."""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Checking and fixing text file encodings...")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            total_count += 1
            if fix_encoding(file_path):
                success_count += 1
            print()  # Add spacing between files
    
    print("=" * 50)
    print(f"Processed {success_count}/{total_count} files successfully")
    print("All files should now be UTF-8 encoded.")

if __name__ == "__main__":
    main()