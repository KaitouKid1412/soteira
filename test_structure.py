#!/usr/bin/env python3
"""
Test script to verify the project structure and basic imports.
Run this before installing heavy dependencies.
"""

import sys
import importlib.util
from pathlib import Path


def test_module_structure(module_path, module_name):
    """Test if a module can be parsed without importing heavy dependencies."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False, f"Could not create spec for {module_name}"
        
        # Try to compile the module without executing it
        with open(module_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, module_path, 'exec')
            return True, "OK"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Test all modules in the project."""
    
    print("Testing project structure...")
    print("=" * 50)
    
    # Test modules
    modules_to_test = [
        ("main.py", "main"),
        ("gates/motion.py", "motion"),
        ("gates/scene.py", "scene"),
        ("gates/objects.py", "objects"),
        ("llm_sink.py", "llm_sink"),
        ("utils/timing.py", "timing"),
        ("utils/io.py", "io_utils"),
    ]
    
    all_passed = True
    
    for module_path, module_name in modules_to_test:
        if Path(module_path).exists():
            success, message = test_module_structure(module_path, module_name)
            status = "✓" if success else "✗"
            print(f"{status} {module_path:25} - {message}")
            if not success:
                all_passed = False
        else:
            print(f"✗ {module_path:25} - File not found")
            all_passed = False
    
    print("=" * 50)
    
    # Test directory structure
    required_dirs = ["gates", "utils", "events"]
    print("\nTesting directory structure...")
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            all_passed = False
    
    # Test required files
    required_files = ["requirements.txt", "README.md"]
    print("\nTesting required files...")
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("✓ All structure tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test with webcam: python main.py --source 0 --show --dry-run")
        print("3. Check help: python main.py --help")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())