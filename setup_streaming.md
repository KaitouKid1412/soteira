# 🚀 Streaming Mode Setup Guide

This guide helps you set up Gemini Flash 2.5 streaming for ultra-fast real-time descriptions.

## 1. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## 2. Set Environment Variable

### For current session:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### For permanent setup (add to ~/.bashrc or ~/.zshrc):
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 3. Test Streaming Setup

Run the test script to verify everything works:

```bash
python test_streaming.py
```

You should see:
- ✅ Gemini Flash 2.5 initialized
- 🚀 Starting streaming test...
- 🎪 Streaming tokens appearing in real-time
- 🎉 Streaming test PASSED!

## 4. Use Speed Mode

1. Start the API server: `python api_server.py`
2. Open the frontend at `http://localhost:8000`
3. Select "Real-time Description (Accessibility)" mode
4. Enable "⚡ Speed Mode" checkbox
5. Enable "🔊 Text-to-Speech"
6. Start analysis with phone stream

## ⚡ Performance Comparison

| Mode | Response Time | Experience |
|------|---------------|------------|
| Standard GPT-4o | 2-3 seconds | Complete response, then speak |
| Speed Mode (Gemini) | 200-500ms | Progressive streaming + TTS |

## 🎯 Expected Benefits

- **4-6x faster** initial response time
- **Progressive narration** starts immediately
- **Real-time feel** for blind user accessibility
- **Smoother experience** with sentence-by-sentence delivery

## 🔧 Troubleshooting

**"GEMINI_API_KEY not set"**
- Make sure you exported the environment variable
- Restart your terminal after adding to bashrc/zshrc

**"No test images found"**
- Run the main system once to generate test images
- Or place any .jpg image in `processing_output/` folder

**"Gemini streaming error"**
- Check your API key is valid
- Ensure you have credits/quota remaining
- Try again in a few minutes (rate limiting)