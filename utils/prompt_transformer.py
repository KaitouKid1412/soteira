"""
LLM-based prompt transformation system for converting user queries 
into specific image analysis prompts.
"""

import json
import openai
from typing import Dict, Optional


class LLMPromptTransformer:
    """Uses LLM to transform user queries into image-specific analysis prompts."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the transformer with OpenAI API.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for prompt transformation
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def transform_prompt(self, user_query: str) -> Dict:
        """
        Transform user query into image analysis prompt using LLM.
        
        Args:
            user_query: Original user query
            
        Returns:
            Dict with transformed prompt and metadata
        """
        
        system_prompt = """You are a prompt transformation expert. Your job is to convert user queries about video/image analysis into specific, actionable prompts for a multimodal AI that will analyze individual images.

User queries might be like:
- "fetch all pictures of humans"
- "check if there are any cigarettes in the feed" 
- "find cars in the video"
- "detect weapons"

Transform these into specific image analysis prompts that:
1. Ask about ONE specific image (not multiple/video)
2. Request JSON response format
3. Are clear and actionable

Respond ONLY with valid JSON in this format:
{
  "image_prompt": "The specific prompt to send with each image",
  "intent": "detection|counting|presence|analysis", 
  "target_objects": ["list", "of", "objects"],
  "response_format": "Description of expected JSON response format",
  "confidence_required": true/false
}

Examples:
User: "fetch all pictures of humans"
Response: {"image_prompt": "Is there a human/person visible in this image? Respond in JSON format: {'detected': true/false, 'count': number, 'confidence': 0-1, 'description': 'brief description of what you see'}", "intent": "detection", "target_objects": ["human", "person"], "response_format": "detected, count, confidence, description", "confidence_required": true}

User: "check if there are cigarettes"  
Response: {"image_prompt": "Analyze this image for cigarettes or smoking-related items. Respond in JSON format: {'detected': true/false, 'items_found': ['list of items'], 'confidence': 0-1, 'description': 'what you observed'}", "intent": "detection", "target_objects": ["cigarette", "smoking"], "response_format": "detected, items_found, confidence, description", "confidence_required": true}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=500
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON if wrapped in code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
                
            transformation = json.loads(response_text)
            
            # Add metadata
            transformation["original_query"] = user_query
            transformation["transformation_success"] = True
            
            return transformation
            
        except json.JSONDecodeError as e:
            print(f"[TRANSFORMER] JSON parsing error: {e}")
            print(f"[TRANSFORMER] Raw response: {response_text}")
            return self._fallback_transformation(user_query)
            
        except Exception as e:
            print(f"[TRANSFORMER] API error: {e}")
            return self._fallback_transformation(user_query)
    
    def _fallback_transformation(self, user_query: str) -> Dict:
        """Fallback transformation if LLM call fails."""
        return {
            "image_prompt": f"Analyze this image based on the request: '{user_query}'. Respond in JSON format with your findings: {{'analysis': 'description', 'confidence': 0-1}}",
            "intent": "analysis",
            "target_objects": [],
            "response_format": "analysis, confidence", 
            "confidence_required": True,
            "original_query": user_query,
            "transformation_success": False,
            "fallback_used": True
        }
    
    def test_transformation(self, query: str) -> None:
        """Test the transformation with a sample query."""
        print(f"Testing query: '{query}'")
        result = self.transform_prompt(query)
        print(f"Transformed prompt: {result['image_prompt']}")
        print(f"Intent: {result['intent']}")
        print(f"Target objects: {result['target_objects']}")
        print("-" * 50)


# Simple transformer without API calls for testing
class SimplePromptTransformer:
    """Simple rule-based transformer for testing without API calls."""
    
    def transform_prompt(self, user_query: str) -> Dict:
        """Simple transformation for testing."""
        
        # Basic pattern matching
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ['human', 'person', 'people']):
            return {
                "image_prompt": "Is there a human/person visible in this image? Respond in JSON format: {'detected': true/false, 'count': number, 'confidence': 0-1, 'description': 'brief description'}",
                "intent": "detection",
                "target_objects": ["human", "person"],
                "response_format": "detected, count, confidence, description",
                "confidence_required": True,
                "original_query": user_query,
                "transformation_success": True
            }
        elif any(word in query_lower for word in ['cigarette', 'smoking', 'smoke']):
            return {
                "image_prompt": "Look for cigarettes or smoking-related items in this image. Respond in JSON format: {'detected': true/false, 'items_found': [], 'confidence': 0-1, 'description': 'what you see'}",
                "intent": "detection", 
                "target_objects": ["cigarette", "smoking"],
                "response_format": "detected, items_found, confidence, description",
                "confidence_required": True,
                "original_query": user_query,
                "transformation_success": True
            }
        else:
            return {
                "image_prompt": f"Analyze this image for: '{user_query}'. Respond in JSON format: {{'analysis': 'description', 'relevant_objects': [], 'confidence': 0-1}}",
                "intent": "analysis",
                "target_objects": [],
                "response_format": "analysis, relevant_objects, confidence",
                "confidence_required": True,
                "original_query": user_query,
                "transformation_success": True
            }