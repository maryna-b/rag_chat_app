"""
Simple test script to verify the basic setup without requiring all dependencies.
"""
import os
import sys

def test_project_structure():
    """Test that all required files and directories exist."""
    print("Testing project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        ".env.example",
        "src/document_processor.py",
        "src/vector_store.py"
    ]
    
    required_dirs = [
        "src",
        "data"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"[OK] {file_path}")
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"[OK] {dir_path}/")
    
    if missing_files:
        print(f"\n[ERROR] Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"\n[ERROR] Missing directories: {missing_dirs}")
        return False
    
    print("\n[SUCCESS] All required files and directories exist!")
    return True

def test_sample_documents():
    """Test that sample documents exist and can be read."""
    print("\nTesting sample documents...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False
    
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print("[ERROR] No .txt files found in data directory")
        return False
    
    total_content = 0
    for filename in txt_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                total_content += len(content)
                print(f"[OK] {filename} ({len(content)} characters)")
        except Exception as e:
            print(f"[ERROR] Could not read {filename}: {e}")
            return False
    
    print(f"\n[SUCCESS] Found {len(txt_files)} documents with {total_content} total characters")
    return True

def test_python_syntax():
    """Test that Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = [
        "src/document_processor.py",
        "src/vector_store.py"
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, file_path, 'exec')
            print(f"[OK] {file_path} - syntax valid")
            
        except SyntaxError as e:
            print(f"[ERROR] {file_path} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] {file_path} - error: {e}")
            return False
    
    print("\n[SUCCESS] All Python files have valid syntax!")
    return True

def main():
    """Run all tests."""
    print("RAG Chat App - Setup Test")
    print("=" * 40)
    
    tests = [
        test_project_structure,
        test_sample_documents, 
        test_python_syntax
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"[ERROR] Test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("[SUCCESS] All tests passed! Project setup is ready.")
        print("\nNext steps:")
        print("1. Set up your OpenAI API key in .env file")
        print("2. Install dependencies: pip install -r requirements.txt") 
        print("3. Run the application: streamlit run main.py")
    else:
        print("[FAILED] Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()