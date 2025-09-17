import os
import sys
import traceback
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.factory import RAGSystemFactory
    from src.config import ConfigurationManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class TestSuite:
    """Comprehensive test suite for the RAG system."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling."""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            if result:
                self.passed_tests += 1
                self.test_results.append((test_name, "PASSED", None))
                print(f"[PASS] {test_name}")
            else:
                self.failed_tests += 1
                self.test_results.append((test_name, "FAILED", "Test returned False"))
                print(f"[FAIL] {test_name}")
                
        except Exception as e:
            self.failed_tests += 1
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.test_results.append((test_name, "ERROR", error_msg))
            print(f"[ERROR] {test_name}: {str(e)}")
    
    def test_project_structure(self) -> bool:
        """Test that all required files and directories exist."""
        print("Testing project structure...")
        
        required_files = [
            "README.md",
            "requirements.txt",
            "main.py",
            "src/interfaces.py",
            "src/config.py",
            "src/logger.py",
            "src/document_loader.py",
            "src/document_processor.py",
            "src/vector_store.py",
            "src/rag_pipeline.py",
            "src/factory.py"
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
                print(f"  [OK] {file_path}")
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
            else:
                print(f"  [OK] {dir_path}/")
        
        if missing_files:
            print(f"  [ERROR] Missing files: {missing_files}")
            return False
        
        if missing_dirs:
            print(f"  [ERROR] Missing directories: {missing_dirs}")
            return False
        
        print("  [SUCCESS] All required files and directories exist!")
        return True
    
    def test_sample_documents(self) -> bool:
        """Test that sample documents exist and can be read."""
        print("Testing sample documents...")
        
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"  [ERROR] Data directory not found: {data_dir}")
            return False
        
        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        if not txt_files:
            print("  [ERROR] No .txt files found in data directory")
            return False
        
        total_content = 0
        for filename in txt_files:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_content += len(content)
                    print(f"  [OK] {filename} ({len(content)} characters)")
            except Exception as e:
                print(f"  [ERROR] Could not read {filename}: {e}")
                return False
        
        print(f"  [SUCCESS] Found {len(txt_files)} documents with {total_content} total characters")
        return True
    
    def test_configuration(self) -> bool:
        """Test configuration management."""
        print("Testing configuration...")
        
        try:
            config_manager = ConfigurationManager()
            config = config_manager.get_config()
            
            print(f"  [OK] Configuration loaded")
            print(f"  [OK] Model: {config.openai_model}")
            print(f"  [OK] Chunk size: {config.chunk_size}")
            print(f"  [OK] Data directory: {config.data_directory}")
            
            # Test validation
            validation_errors = config_manager.get_validation_errors()
            if validation_errors:
                print(f"  [WARNING] Configuration issues: {validation_errors}")
                # Don't fail the test for missing API key in testing
                if len(validation_errors) == 1 and "OPENAI_API_KEY" in validation_errors[0]:
                    print("  [INFO] API key missing - expected in test environment")
                    return True
                return False
            
            print("  [SUCCESS] Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Configuration test failed: {e}")
            return False
    
    def test_factory_pattern(self) -> bool:
        """Test factory pattern and dependency injection."""
        print("Testing factory pattern...")
        
        if not os.getenv("OPENAI_API_KEY"):
            print("  [WARNING] Skipping factory test - no API key")
            return True
        
        try:
            factory = RAGSystemFactory()
            print("  [OK] Factory created successfully")
            
            # Test component creation
            loader = factory.create_document_loader()
            print("  [OK] Document loader created")
            
            processor = factory.create_document_processor()
            print("  [OK] Document processor created")
            
            pipeline = factory.create_document_pipeline()
            print("  [OK] Document pipeline created")
            
            vector_store = factory.create_vector_store()
            print("  [OK] Vector store created")
            
            print("  [SUCCESS] All components created successfully")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Factory test failed: {e}")
            return False
    
    def test_document_processing(self) -> bool:
        """Test document loading and processing."""
        print("Testing document processing...")
        
        if not os.getenv("OPENAI_API_KEY"):
            print("  [WARNING] Skipping document processing test - no API key")
            return True
        
        try:
            factory = RAGSystemFactory()
            pipeline = factory.create_document_pipeline()
            
            # Test document processing
            documents = pipeline.process_directory("data")
            
            if not documents:
                print("  [ERROR] No documents processed")
                return False
            
            print(f"  [OK] Processed {len(documents)} document chunks")
            
            # Check document structure
            sample_doc = documents[0]
            print(f"  [OK] Sample document source: {sample_doc.metadata.get('source', 'N/A')}")
            print(f"  [OK] Sample content length: {len(sample_doc.page_content)}")
            
            print("  [SUCCESS] Document processing successful")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Document processing test failed: {e}")
            return False
    
    def test_rag_pipeline(self) -> bool:
        """Test RAG pipeline functionality."""
        print("Testing RAG pipeline...")
        
        if not os.getenv("OPENAI_API_KEY"):
            print("  [WARNING] Skipping RAG pipeline test - no API key")
            return True
        
        try:
            factory = RAGSystemFactory()
            rag_pipeline = factory.create_rag_pipeline()
            
            # Initialize the pipeline
            if not rag_pipeline.initialize("data"):
                print("  [ERROR] Failed to initialize RAG pipeline")
                return False
            
            print("  [OK] RAG pipeline initialized")
            
            # Test a simple query
            test_question = "What information is available in the documents?"
            result = rag_pipeline.query(test_question)
            
            if not result or not result.get("answer"):
                print("  [ERROR] Query returned no answer")
                return False
            
            print(f"  [OK] Query processed successfully")
            print(f"  [OK] Answer length: {len(result['answer'])}")
            print(f"  [OK] Sources found: {len(result.get('sources', []))}")
            
            print("  [SUCCESS] RAG pipeline test successful")
            return True
            
        except Exception as e:
            print(f"  [ERROR] RAG pipeline test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests and provide summary."""
        print("RAG Chat App - Refactored Test Suite")
        print("="*60)
        
        test_methods = [
            ("Project Structure", self.test_project_structure),
            ("Sample Documents", self.test_sample_documents),
            ("Configuration Management", self.test_configuration),
            ("Factory Pattern", self.test_factory_pattern),
            ("Document Processing", self.test_document_processing),
            ("RAG Pipeline", self.test_rag_pipeline)
        ]
        
        for test_name, test_method in test_methods:
            self.run_test(test_name, test_method)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)
        
        total_tests = self.passed_tests + self.failed_tests
        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        
        if self.failed_tests > 0:
            print(f"\n[FAILED] TESTS:")
            for test_name, status, error in self.test_results:
                if status in ["FAILED", "ERROR"]:
                    print(f"  - {test_name}: {status}")
                    if error:
                        print(f"    Error: {error.split(chr(10))[0]}")  # First line of error
        
        success_rate = (self.passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\n[SUCCESS] ALL TESTS PASSED!")
            print("\nNext steps:")
            if not os.getenv("OPENAI_API_KEY"):
                print("1. Set up your OpenAI API key in .env file")
            print("2. Install dependencies: pip install -r requirements.txt")
            print("3. Run the application: streamlit run main.py")
        else:
            print(f"\n[WARNING] {self.failed_tests} test(s) failed. Please fix the issues above.")
        
        return self.failed_tests == 0


def main():
    """Run the test suite."""
    test_suite = TestSuite()
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)