#!/usr/bin/env python3
"""
Validation script to verify the benchmark structure and imports.
This script checks that all modules can be imported without errors.
"""

import sys

def validate_imports():
    """Validate that all modules can be imported."""
    print("Validating module imports...")
    
    modules = [
        'config',
        'db_connection',
        'embedding_models',
        'vector_store',
        'evaluation',
        'sample_data',
        'benchmark'
    ]
    
    errors = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            # Expected for packages not installed yet
            if 'langchain' in str(e) or 'tidb_vector' in str(e) or 'openai' in str(e) or 'sentence_transformers' in str(e):
                print(f"⚠ {module_name} - dependencies not installed (expected)")
            else:
                print(f"✗ {module_name} - {e}")
                errors.append((module_name, e))
        except Exception as e:
            print(f"✗ {module_name} - {e}")
            errors.append((module_name, e))
    
    return len(errors) == 0

def validate_structure():
    """Validate project structure."""
    print("\nValidating project structure...")
    
    import os
    
    required_files = [
        'benchmark.py',
        'config.py',
        'db_connection.py',
        'embedding_models.py',
        'vector_store.py',
        'evaluation.py',
        'sample_data.py',
        'requirements.txt',
        '.env.example',
        'README.md',
        '.gitignore'
    ]
    
    missing = []
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - MISSING")
            missing.append(filename)
    
    return len(missing) == 0

def validate_sample_data():
    """Validate sample data is available."""
    print("\nValidating sample data...")
    
    try:
        from sample_data import get_documents, get_test_queries
        
        docs = get_documents()
        queries = get_test_queries()
        
        print(f"✓ Sample documents: {len(docs)} documents")
        print(f"✓ Test queries: {len(queries)} queries")
        
        # Check document structure
        if docs:
            doc = docs[0]
            if 'content' in doc and 'metadata' in doc:
                print(f"✓ Document structure is valid")
            else:
                print(f"✗ Document structure is invalid")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating sample data: {e}")
        return False

def main():
    """Run all validation checks."""
    print("="*80)
    print("TiDB Vector LLM Testbed - Validation Script")
    print("="*80)
    
    results = []
    
    # Check structure
    results.append(("Project Structure", validate_structure()))
    
    # Check imports (will show warnings about missing dependencies, which is expected)
    results.append(("Module Imports", validate_imports()))
    
    # Check sample data
    results.append(("Sample Data", validate_sample_data()))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure .env file with your TiDB credentials")
        print("3. Run benchmark: python benchmark.py")
        return 0
    else:
        print("\n✗ Some validation checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
