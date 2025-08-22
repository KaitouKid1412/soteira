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
        
    def transform_prompt(self, user_query: str, mode: str = "summary") -> Dict:
        """
        Transform user query into image analysis prompt using LLM.
        
        Args:
            user_query: Original user query
            mode: Processing mode ('summary', 'alert', or 'realtime_description')
            
        Returns:
            Dict with transformed prompt and metadata
        """
        
        if mode == "alert":
            system_prompt = """You are a prompt transformation expert. Your job is to convert user queries into image analysis prompts for ALERT monitoring systems.

Transform user queries into prompts that:
1. Analyze ONE specific image 
2. Return JSON with an 'alert' field (true/false)
3. Focus on detection/identification for alerting

Respond ONLY with valid JSON:
{
  "image_prompt": "The prompt to send with each image",
  "intent": "detection|analysis", 
  "target_objects": ["objects to look for"],
  "alert_condition": "When to set alert=true"
}

Examples:
User: "detect weapons"
Response: {"image_prompt": "Analyze this image for weapons or dangerous objects. Respond in JSON: {'detected': true/false, 'items': [], 'confidence': 0-1, 'description': 'what you see', 'alert': true/false}. Set alert=true if weapons are detected.", "intent": "detection", "target_objects": ["weapon", "gun", "knife"], "alert_condition": "Weapons or dangerous objects detected"}

User: "check for smoking"
Response: {"image_prompt": "Look for smoking or cigarettes in this image. Respond in JSON: {'detected': true/false, 'items': [], 'confidence': 0-1, 'description': 'observation', 'alert': true/false}. Set alert=true if smoking is detected.", "intent": "detection", "target_objects": ["cigarette", "smoking"], "alert_condition": "Smoking or cigarettes detected"}"""
        elif mode == "realtime_description":
            system_prompt = """You are a prompt transformation expert. Your job is to convert user queries into image analysis prompts for REAL-TIME ACCESSIBILITY descriptions.

Transform user queries into prompts that:
1. Provide immediate, useful scene descriptions for blind users
2. Focus on spatial awareness, movement, and context
3. Return clear, conversational descriptions in JSON

Respond ONLY with valid JSON:
{
  "image_prompt": "The prompt to send with each image",
  "intent": "accessibility_description", 
  "target_objects": ["elements to describe"],
  "description_focus": "What to emphasize for accessibility"
}

Examples:
User: "Describe what you see in the scene"
Response: {"image_prompt": "This is a frame from a live video stream. Describe this scene for a blind person in 1-2 sentences. Respond ONLY in JSON: {'analysis': 'brief description of people, activities, and key objects', 'confidence': 0.8}. Keep the analysis under 50 words. Focus on what's happening NOW in this moment.", "intent": "accessibility_description", "target_objects": ["people", "objects", "environment", "activities", "spatial_layout"], "description_focus": "Clear spatial awareness and activity description"}

User: "Help me understand my surroundings"
Response: {"image_prompt": "Provide a comprehensive scene description for accessibility. Respond in JSON: {'analysis': 'description covering: immediate surroundings, people and their actions, objects and their positions, potential navigation information, current activities', 'confidence': 0-1}. Use clear, helpful language focusing on what's useful for spatial understanding.", "intent": "accessibility_description", "target_objects": ["environment", "navigation", "people", "obstacles", "activities"], "description_focus": "Navigation assistance and environmental awareness"}"""
        else:  # summary mode
            system_prompt = """You are a prompt transformation expert. Your job is to convert user queries into image analysis prompts for SUMMARY generation systems.

Transform user queries into prompts that:
1. Analyze ONE specific image thoroughly
2. Return detailed JSON for later summarization
3. Focus on comprehensive observation

Respond ONLY with valid JSON:
{
  "image_prompt": "The prompt to send with each image",
  "intent": "analysis|description", 
  "target_objects": ["things to observe"],
  "summary_focus": "What to emphasize for summary"
}

Examples:
User: "What actions are users performing?"
Response: {"image_prompt": "Analyze this image and describe what action or activity is happening. Respond in JSON: {'action': 'main activity', 'objects_used': [], 'people_count': number, 'confidence': 0-1, 'detailed_description': 'comprehensive observation'}. Focus on actions and interactions.", "intent": "analysis", "target_objects": ["person", "action", "interaction"], "summary_focus": "User actions and activities over time"}

User: "monitor office activity"
Response: {"image_prompt": "Describe the office activity in this image. Respond in JSON: {'activity_type': 'description', 'people_present': number, 'objects_visible': [], 'environment': 'setting description', 'confidence': 0-1}. Provide detailed observations.", "intent": "analysis", "target_objects": ["office", "people", "activity"], "summary_focus": "Office environment and work patterns"}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0,  # Zero temperature for fastest responses
                max_tokens=2000  # Reduced tokens for faster transformation
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
            transformation["mode"] = mode
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
    
    def transform_prompt(self, user_query: str, mode: str = "summary") -> Dict:
        """Simple transformation for testing."""
        
        # Basic pattern matching
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ['human', 'person', 'people']):
            if mode == "alert":
                return {
                    "image_prompt": "Is there a human/person visible in this image? Respond in JSON format: {'detected': true/false, 'count': number, 'confidence': 0-1, 'description': 'brief description', 'alert': true/false}. Set alert=true if humans are detected.",
                    "intent": "detection",
                    "target_objects": ["human", "person"],
                    "alert_condition": "Humans detected",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
            elif mode == "realtime_description":
                return {
                    "image_prompt": "This is a frame from a live video stream. Describe what you see for a blind person in 1-2 short sentences. Respond ONLY in JSON: {'analysis': 'brief description of people and their activities', 'confidence': 0.8}. Keep under 30 words. Focus on current moment and actions.",
                    "intent": "accessibility_description",
                    "target_objects": ["people", "activities", "spatial_layout"],
                    "description_focus": "People-focused accessibility description",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
            else:  # summary mode
                return {
                    "image_prompt": "Describe any people visible in this image in detail. Respond in JSON format: {'people_count': number, 'activities': ['list of activities'], 'descriptions': ['detailed descriptions'], 'confidence': 0-1}",
                    "intent": "analysis", 
                    "target_objects": ["human", "person"],
                    "summary_focus": "People and their activities",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
        elif any(word in query_lower for word in ['cigarette', 'smoking', 'smoke']):
            if mode == "alert":
                return {
                    "image_prompt": "Look for cigarettes or smoking-related items in this image. Respond in JSON format: {'detected': true/false, 'items_found': [], 'confidence': 0-1, 'description': 'what you see', 'alert': true/false}. Set alert=true if smoking is detected.",
                    "intent": "detection", 
                    "target_objects": ["cigarette", "smoking"],
                    "alert_condition": "Smoking or cigarettes detected",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
            else:  # summary mode
                return {
                    "image_prompt": "Observe and describe any smoking-related activities or items in this image. Respond in JSON format: {'smoking_present': true/false, 'items_observed': [], 'context': 'environmental context', 'confidence': 0-1}",
                    "intent": "analysis",
                    "target_objects": ["cigarette", "smoking"],
                    "summary_focus": "Smoking behavior patterns",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
        else:
            if mode == "alert":
                return {
                    "image_prompt": f"Analyze this image for: '{user_query}'. Respond in JSON format: {{'analysis': 'description', 'relevant_objects': [], 'confidence': 0-1, 'alert': true/false}}. Set alert=true if anything noteworthy or concerning is found.",
                    "intent": "analysis",
                    "target_objects": [],
                    "alert_condition": "Noteworthy or concerning content detected",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
            elif mode == "realtime_description":
                return {
                    "image_prompt": f"This is a frame from a live video stream. Describe the current scene for a blind person in 1-2 sentences. Respond ONLY in JSON: {{'analysis': 'brief description of what is happening right now', 'confidence': 0.8}}. Keep under 40 words. Focus on: '{user_query}'",
                    "intent": "accessibility_description",
                    "target_objects": ["environment", "objects", "people", "activities"],
                    "description_focus": "General accessibility description",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }
            else:  # summary mode
                return {
                    "image_prompt": f"Provide a comprehensive analysis of this image for: '{user_query}'. Respond in JSON format: {{'analysis': 'detailed description', 'relevant_objects': [], 'context': 'environmental context', 'confidence': 0-1}}",
                    "intent": "analysis",
                    "target_objects": [],
                    "summary_focus": "Comprehensive scene understanding",
                    "original_query": user_query,
                    "mode": mode,
                    "transformation_success": True
                }