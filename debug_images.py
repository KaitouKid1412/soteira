#!/usr/bin/env python3
"""
Debug script to test image saving functionality
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_opencv_import():
    """Test if OpenCV can be imported"""
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  OpenCV version: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

def test_directory_permissions():
    """Test if we can write to events directory"""
    events_dir = Path("events")
    test_file = events_dir / "test_write.txt"
    
    try:
        events_dir.mkdir(exist_ok=True)
        with open(test_file, 'w') as f:
            f.write("test")
        
        if test_file.exists():
            test_file.unlink()  # Delete test file
            print("✓ Directory write permissions OK")
            return True
        else:
            print("✗ Could not create test file")
            return False
            
    except Exception as e:
        print(f"✗ Directory write test failed: {e}")
        return False

def test_image_saving():
    """Test if we can save a simple image"""
    if not test_opencv_import():
        return False
    
    import cv2
    
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [0, 255, 0]  # Green image
        
        # Test file path
        test_path = "events/debug_test.jpg"
        
        # Try to save
        success = cv2.imwrite(test_path, test_image)
        
        if success and os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            print(f"✓ Image saved successfully: {test_path} ({file_size} bytes)")
            
            # Clean up
            os.remove(test_path)
            return True
        else:
            print(f"✗ Image save failed - success={success}, exists={os.path.exists(test_path)}")
            return False
            
    except Exception as e:
        print(f"✗ Image saving test failed: {e}")
        return False

def test_utils_io():
    """Test the utils.io.safe_write_image function"""
    try:
        from utils.io import safe_write_image
        import cv2
        
        # Create test image
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[:, :] = [255, 0, 0]  # Red image
        
        test_path = "events/utils_test.jpg"
        
        # Test safe_write_image
        success = safe_write_image(test_path, test_image)
        
        if success and os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            print(f"✓ utils.io.safe_write_image works: {test_path} ({file_size} bytes)")
            
            # Clean up
            os.remove(test_path)
            return True
        else:
            print(f"✗ utils.io.safe_write_image failed - success={success}, exists={os.path.exists(test_path)}")
            return False
            
    except Exception as e:
        print(f"✗ utils.io test failed: {e}")
        return False

def main():
    print("Debugging image saving functionality...")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_opencv_import,
        test_directory_permissions, 
        test_image_saving,
        test_utils_io
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    if all(results):
        print("✓ All tests passed! Image saving should work.")
        print("\nThe issue might be:")
        print("- Images being saved with different names")
        print("- Images being saved in a different directory")
        print("- Main script not calling save functions")
    else:
        print("✗ Some tests failed. This explains why images aren't being saved.")
        print("\nNext steps:")
        print("- Install missing dependencies")
        print("- Fix permission issues")
        print("- Debug the specific failing test")

if __name__ == "__main__":
    main()