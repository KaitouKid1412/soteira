#!/usr/bin/env python3
"""
Test script for Gemini Flash 2.5 streaming functionality.
This tests the streaming implementation without the full video pipeline.
"""

import os
import time
from llm_sink import call_gemini_streaming
import google.generativeai as genai

# Test configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TEST_IMAGE_PATH = "processing_output"  # Look for any recent image

def test_streaming():
    """Test Gemini streaming with a sample image."""
    print("🧪 Testing Gemini Flash 2.5 Streaming")
    print("=" * 50)
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your-key-here'")
        return False
    
    try:
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini Flash 2.5 initialized")
        
        # Find a test image
        from pathlib import Path
        output_dir = Path(TEST_IMAGE_PATH)
        if output_dir.exists():
            images = list(output_dir.glob("*.jpg"))
            if images:
                test_image = str(images[0])
                print(f"📸 Using test image: {test_image}")
            else:
                print("❌ No test images found in processing_output/")
                return False
        else:
            print("❌ processing_output directory not found")
            return False
        
        # Test prompt
        prompt = "Describe this scene for a blind person in detail, including people, objects, and activities."
        
        print(f"🎯 Prompt: {prompt}")
        print("🚀 Starting streaming test...")
        print("-" * 30)
        
        # Mock web display for token capture
        class MockWebDisplay:
            def __init__(self):
                self.tokens = []
                self.server = self
                self.pending_token_queue = []
        
        mock_display = MockWebDisplay()
        
        # Start timing
        start_time = time.time()
        
        # Call streaming function
        result = call_gemini_streaming(test_image, prompt, model, mock_display)
        
        # End timing
        elapsed = time.time() - start_time
        
        print("-" * 30)
        print(f"⏱️  Total time: {elapsed:.2f} seconds")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print(f"✅ Success! Response length: {len(result['choices'][0]['message']['content'])} chars")
        print(f"🎭 Tokens captured: {len(mock_display.pending_token_queue)}")
        print(f"📊 Usage: {result['usage']}")
        
        # Show first few tokens
        if mock_display.pending_token_queue:
            print("\n🎪 First few streaming tokens:")
            for i, token_data in enumerate(mock_display.pending_token_queue[:5]):
                print(f"  {i+1}: '{token_data['token']}'")
        
        print(f"\n📄 Full response preview:")
        response_text = result['choices'][0]['message']['content']
        print(f"  {response_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming()
    if success:
        print("\n🎉 Streaming test PASSED!")
        print("💡 You can now use Speed Mode in the frontend")
    else:
        print("\n💥 Streaming test FAILED!")
        print("🔧 Check your GEMINI_API_KEY and try again")