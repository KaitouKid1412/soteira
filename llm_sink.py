"""
LLM sink for processing video events with OpenAI GPT-4 Vision.
Runs in background thread to avoid blocking main video processing loop.
"""

import json
import time
import queue
import threading
import base64
from datetime import datetime
from pathlib import Path
import openai
from typing import Optional
import cv2
import numpy as np

from utils.prompt_transformer import LLMPromptTransformer, SimplePromptTransformer


def extract_image_features_fast(image_source) -> np.ndarray:
    """
    Fast feature extraction using simple pixel downsampling.
    
    Args:
        image_source: Either path to image file (str) or numpy array (cv2 image)
        
    Returns:
        Feature vector as numpy array (small and fast)
    """
    try:
        # Load image from path or use provided array
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                return np.zeros(36)  # Ultra small for speed  # Much smaller fallback
        else:
            # Assume it's already a cv2 image array
            img = image_source
            if img is None or img.size == 0:
                return np.zeros(36)  # Ultra small for speed
        
        # Ultra small resize for maximum speed (6x6 = 36 pixels)
        img_tiny = cv2.resize(img, (6, 6))
        
        # Convert to grayscale for speed
        img_gray = cv2.cvtColor(img_tiny, cv2.COLOR_BGR2GRAY)
        
        # Flatten to feature vector
        features = img_gray.flatten().astype(np.float32)
        
        # Proper L2 normalization for cosine similarity
        features = features / 255.0  # Scale to [0,1]
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm  # L2 normalize to unit length
        
        return features
        
    except Exception as e:
        return np.zeros(36)  # Ultra small for speed


def encode_image_to_base64(image_source, max_size: int = 512, quality: int = 75) -> str:
    """Encode image to base64 for OpenAI API with compression and resizing.
    
    Args:
        image_source: Either file path (str) or cv2 image array (numpy.ndarray)
        max_size: Maximum dimension (width or height) for resizing
        quality: JPEG compression quality (1-100, lower = smaller file)
    """
    import cv2
    
    # Load image
    if isinstance(image_source, str):
        # File path provided
        img = cv2.imread(image_source)
        if img is None:
            raise Exception(f"Failed to load image from {image_source}")
    else:
        # Numpy array provided (cv2 image)
        img = image_source
        if img is None:
            raise Exception("Invalid image array provided")
    
    # Resize image if too large (for faster API processing)
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Encode with compression
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode('.jpg', img, encode_params)
    if not success:
        raise Exception("Failed to encode image to JPEG")
    
    return base64.b64encode(buffer).decode('utf-8')


def send_to_openai_vision(prompt: str, image_source, api_key: str, 
                         model: str = "gpt-4o", use_fast_model: bool = False) -> dict:
    """
    Send image and prompt to OpenAI GPT-4 Vision API.
    
    Args:
        prompt: Text prompt for analysis
        image_source: Path to image file (str) or cv2 image array (numpy.ndarray)
        api_key: OpenAI API key
        model: Model to use (default: gpt-4o)
        
    Returns:
        API response dict
    """
    try:
        # Encode image with optimization for speed
        base64_image = encode_image_to_base64(image_source, max_size=512, quality=75)
        
        # Use faster model for high-load scenarios
        if use_fast_model:
            model = "gpt-4o-mini"  # Faster, cheaper alternative
        
        # Initialize OpenAI client with connection pooling and optimizations
        client = openai.OpenAI(
            api_key=api_key,
            timeout=10.0,  # Reduced timeout for faster failure detection
            max_retries=2   # Reduced retries for speed
        )
        
        # Prepare payload
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150,  # Reduced for faster responses
            temperature=0.0  # Zero temperature for fastest, most consistent responses
        )
        
        # Convert to dict format
        result = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        print(f"[LLM] ✓ OpenAI API call successful")
        print(f"[LLM] Response: {result['choices'][0]['message']['content'][:200]}...")
        return result
        
    except Exception as e:
        print(f"[LLM] ✗ OpenAI API error: {e}")
        # Return error response
        return {
            "error": str(e),
            "id": f"error-{int(time.time())}",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "message": {
                        "role": "assistant", 
                        "content": f"Error analyzing image: {e}"
                    },
                    "finish_reason": "error"
                }
            ]
        }


def send_to_gpt_stub(prompt: str, image_path: str) -> dict:
    """
    Stub function for testing without API calls.
    """
    # Simulate processing time
    time.sleep(0.2)
    
    # Generate simulated response based on prompt
    if "human" in prompt.lower() or "person" in prompt.lower():
        simulated_content = '{"detected": true, "count": 1, "confidence": 0.85, "description": "One person visible in the image"}'
    elif "cigarette" in prompt.lower() or "smoking" in prompt.lower():
        simulated_content = '{"detected": false, "items_found": [], "confidence": 0.90, "description": "No cigarettes or smoking items detected"}'
    else:
        simulated_content = '{"analysis": "General image analysis completed", "confidence": 0.75}'
    
    return {
        "id": f"stub-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o-stub",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": simulated_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


class LLMSink:
    """Background processor for LLM API calls with parallel processing and rate limit handling."""
    
    def __init__(self, user_query: str, openai_api_key: Optional[str] = None, 
                 dry_run: bool = False, max_queue_size: int = 200, num_workers: int = 8,
                 similarity_threshold: float = 0.75, debug_mode: bool = False,
                 mode: str = "summary", web_display=None):
        """
        Initialize LLM sink.
        
        Args:
            user_query: Original user query (e.g., "find humans", "check for cigarettes")
            openai_api_key: OpenAI API key (None for stub mode)
            dry_run: If True, don't actually call LLM (just log)
            max_queue_size: Maximum number of queued requests (increased for better throughput)
            num_workers: Number of parallel worker threads (optimized for real-time processing)
            similarity_threshold: Cosine similarity threshold (0.75 = 75% similar = skip, optimized for speed)
            debug_mode: Enable verbose debug logging
            mode: Processing mode ('summary' or 'alert')
        """
        self.user_query = user_query
        self.openai_api_key = openai_api_key
        self.dry_run = dry_run
        self.max_queue_size = max_queue_size
        self.similarity_threshold = similarity_threshold
        self.debug_mode = debug_mode
        self.mode = mode
        self.web_display = web_display
        
        # Response collection for summary mode
        self.collected_responses = []  # List of (timestamp, image_path, response_content)
        self.response_lock = threading.Lock()  # Thread-safe access to responses
        
        # Alert tracking
        self.alert_count = 0
        
        # Image similarity tracking - maintain history of sent images
        self.sent_image_features = []  # List of all sent image features
        self.max_history_size = 100    # Keep last 100 sent images for comparison
        
        # Temporal deduplication - skip frames too close in time
        self.last_sent_timestamp = 0.0  # Timestamp of last sent frame
        self.temporal_threshold_ms = 500  # Minimum time between frames (ms)
        
        # Motion-based prioritization
        self.motion_threshold_high = 0.1   # High motion threshold for priority
        self.motion_threshold_low = 0.02   # Low motion threshold for skipping
        self.similarity_stats = {
            'total_requested': 0,
            'skipped_similar': 0,
            'sent_to_llm': 0
        }
        
        # Thread-safe lock for similarity feature updates
        self.similarity_lock = threading.Lock()
        self.num_workers = num_workers
        
        # Initialize prompt transformer
        if openai_api_key and not dry_run:
            self.prompt_transformer = LLMPromptTransformer(openai_api_key)
            self.use_real_api = True
        else:
            self.prompt_transformer = SimplePromptTransformer()
            self.use_real_api = False
        
        # Transform the user query once with mode
        self.transformation = self.prompt_transformer.transform_prompt(user_query, mode)
        self.image_prompt = self.transformation["image_prompt"]
        
        print(f"[LLM] User query: '{user_query}'")
        print(f"[LLM] Transformed prompt: '{self.image_prompt[:100]}...'")
        print(f"[LLM] Intent: {self.transformation['intent']}")
        print(f"[LLM] Target objects: {self.transformation['target_objects']}")
        
        # Thread-safe queue for LLM requests
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        
        # Multiple worker threads for parallel processing
        self.worker_threads = []
        self.running = False
        
        # Rate limiting and retry tracking
        self.rate_limit_backoff = 0  # Seconds to wait due to rate limiting
        self.last_rate_limit = 0     # Timestamp of last rate limit
        self.rate_limit_lock = threading.Lock()
        
        # Model fallback for high load scenarios
        self.queue_high_threshold = max_queue_size * 0.8  # 80% full = use fast model
        self.use_fast_model_fallback = True
        
        # Statistics
        self.requests_processed = 0
        self.requests_failed = 0
        self.requests_rate_limited = 0
        self.start_time = time.time()
        
        # Cost tracking (for gpt-4o default model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        
        # GPT-4o pricing per 1M tokens (as of 2024)
        self.gpt4o_input_cost = 2.50    # $2.50 per 1M input tokens
        self.gpt4o_output_cost = 10.00  # $10.00 per 1M output tokens
        
        # Create analysis directory
        self.analysis_dir = Path("events") / "llm_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Create directory for tracking LLM-sent images
        self.llm_sent_dir = Path("events") / "llm_sent"
        self.llm_sent_dir.mkdir(exist_ok=True)
        
        # Start worker thread
        self.start()
    
    def start(self):
        """Start the background worker threads."""
        if self.running:
            return
            
        self.running = True
        
        # Start multiple worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            self.worker_threads.append(worker)
            worker.start()
        
        print(f"[LLM] Started {self.num_workers} parallel worker threads")
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop processing LLM requests.
        
        Args:
            worker_id: Unique ID for this worker thread
        """
        if self.debug_mode:
            print(f"[LLM-{worker_id}] Worker thread started")
        
        while self.running:
            try:
                # Get request from queue (with timeout)
                try:
                    request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                _, image_path, frame_data = request  # We now ignore the original prompt and use transformed one
                
                if self.debug_mode:
                    print(f"[LLM-{worker_id}] Processing request for: {image_path}")
                
                # Check similarity and skip if too similar to previous images
                if not self._check_similarity_and_add(image_path, frame_data):
                    # Image too similar, skip processing
                    self.request_queue.task_done()
                    continue
                
                # Log that image was sent to LLM (for tracking purposes)
                self._log_sent_image(image_path, frame_data)
                
                # Check for rate limiting backoff
                with self.rate_limit_lock:
                    if self.rate_limit_backoff > 0:
                        backoff_remaining = self.rate_limit_backoff - (time.time() - self.last_rate_limit)
                        if backoff_remaining > 0:
                            print(f"[LLM-{worker_id}] Rate limit backoff: waiting {backoff_remaining:.1f}s")
                            time.sleep(min(backoff_remaining, 1.0))  # Sleep max 1s at a time
                            continue
                        else:
                            self.rate_limit_backoff = 0  # Clear backoff
                
                # Process request with retry logic
                try:
                    if self.dry_run:
                        print(f"[LLM-{worker_id}] DRY RUN - Would analyze image:")
                        print(f"  Image: {image_path}")
                        print(f"  With prompt: {self.image_prompt[:100]}...")
                        
                        # Show running total for dry run
                        total = self.similarity_stats['total_requested']
                        sent = self.similarity_stats['sent_to_llm'] 
                        skipped = self.similarity_stats['skipped_similar']
                        print(f"  [DRY-RUN TOTAL] Would send {sent} images to LLM (filtered {skipped}/{total} similar)")
                        
                        response = send_to_gpt_stub(self.image_prompt, image_path)
                    else:
                        # Use real API or stub based on setup with smart model selection
                        if self.use_real_api and self.openai_api_key:
                            # Check if we should use fast model due to high queue load
                            current_queue_size = self.get_queue_size()
                            use_fast = self.use_fast_model_fallback and current_queue_size > self.queue_high_threshold
                            if use_fast and self.debug_mode:
                                print(f"[FAST_MODEL] Using gpt-4o-mini due to high queue ({current_queue_size}/{self.max_queue_size})")
                            response = self._call_openai_with_retry(image_path, frame_data, worker_id, use_fast_model=use_fast)
                        else:
                            response = send_to_gpt_stub(self.image_prompt, image_path)
                    
                    # Update cost tracking (only for real API calls)
                    if self.use_real_api and self.openai_api_key and not self.dry_run:
                        self._update_cost_tracking(response)
                    
                    # Process response based on mode
                    self._process_response(image_path, response)
                    
                    # Save response and analysis
                    self._save_analysis(image_path, response)
                    self.requests_processed += 1
                    
                except Exception as e:
                    print(f"[LLM-{worker_id}] Error processing request: {e}")
                    self.requests_failed += 1
                
                # Mark task as done
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"[LLM-{worker_id}] Worker loop error: {e}")
                time.sleep(0.1)
    
    def _call_openai_with_retry(self, image_path: str, frame_data, worker_id: int, max_retries: int = 2, use_fast_model: bool = False) -> dict:
        """Call OpenAI API with retry logic for rate limiting.
        
        Args:
            image_path: Path to image file (for logging)
            frame_data: Image data as numpy array (preferred) or None to use file path
            worker_id: Worker thread ID for logging
            max_retries: Maximum number of retries
            use_fast_model: Whether to use faster gpt-4o-mini model for speed
            
        Returns:
            dict: API response
        """
        for attempt in range(max_retries + 1):
            try:
                # Use frame_data if available, otherwise fall back to image_path
                image_source = frame_data if frame_data is not None else image_path
                response = send_to_openai_vision(self.image_prompt, image_source, self.openai_api_key, use_fast_model=use_fast_model)
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limiting errors
                if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                    self.requests_rate_limited += 1
                    
                    # Extract wait time from error if possible, otherwise use faster backoff
                    wait_time = min(1.0 + (attempt * 0.5), 3.0)  # Linear backoff: 1.5s, 2s, 2.5s (max 3s)
                    
                    # Try to extract wait time from error message
                    if "try again in" in error_str:
                        try:
                            # Look for patterns like "try again in 20s" or "try again in 1m"
                            import re
                            match = re.search(r'try again in (\\d+)([sm]?)', error_str)
                            if match:
                                time_val = int(match.group(1))
                                unit = match.group(2)
                                if unit == 'm':
                                    wait_time = time_val * 60
                                else:
                                    wait_time = time_val
                        except:
                            pass  # Fall back to exponential backoff
                    
                    print(f"[LLM-{worker_id}] Rate limited (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time}s")
                    
                    # Set global backoff to coordinate all workers
                    with self.rate_limit_lock:
                        self.rate_limit_backoff = max(self.rate_limit_backoff, wait_time)
                        self.last_rate_limit = time.time()
                    
                    if attempt < max_retries:
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[LLM-{worker_id}] Max retries exceeded for rate limiting")
                        raise
                
                # For other errors, retry with shorter backoff
                elif attempt < max_retries:
                    wait_time = 0.5 * (attempt + 1)  # 0.5s, 1s, 1.5s
                    print(f"[LLM-{worker_id}] API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"[LLM-{worker_id}] Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"[LLM-{worker_id}] Max retries exceeded for API error: {e}")
                    raise
        
        # Should never reach here
        raise Exception("Unexpected end of retry loop")
    
    def generate_synthesis_summary(self) -> str:
        """
        Generate a comprehensive answer to the user's original question by synthesizing all LLM responses.
        
        Returns:
            str: LLM-synthesized answer to the original user question
        """
        if self.mode != "summary":
            return f"Summary not available in {self.mode} mode"
        
        if not self.collected_responses:
            return "No responses collected for summary"
            
        try:
            # Sort responses by timestamp
            sorted_responses = sorted(self.collected_responses, key=lambda x: x['timestamp'])
            
            if self.debug_mode:
                print(f"[SYNTHESIS] Processing {len(sorted_responses)} responses to answer: '{self.user_query}'")
            
            # Prepare data for synthesis
            response_texts = []
            for i, item in enumerate(sorted_responses):
                response = item['response']
                raw_content = item.get('raw_content', '')
                
                if self.debug_mode:
                    print(f"[SYNTHESIS DEBUG] Frame {i+1} response type: {type(response)}")
                    print(f"[SYNTHESIS DEBUG] Frame {i+1} response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
                
                # Convert response to readable text
                if isinstance(response, dict) and response:
                    # Extract meaningful fields
                    response_parts = []
                    
                    # Add action/activity info
                    if 'action' in response:
                        response_parts.append(f"Action: {response['action']}")
                    if 'detailed_description' in response:
                        response_parts.append(f"Description: {response['detailed_description']}")
                    if 'analysis' in response:
                        response_parts.append(f"Analysis: {response['analysis']}")
                    
                    # Add object info
                    if 'objects_used' in response and response['objects_used']:
                        response_parts.append(f"Objects: {response['objects_used']}")
                    if 'people_count' in response:
                        response_parts.append(f"People: {response['people_count']}")
                    
                    # If no structured fields, add all non-metadata fields
                    if not response_parts:
                        for key, value in response.items():
                            if key not in ['confidence', 'raw_response', 'alert'] and value:
                                response_parts.append(f"{key}: {value}")
                    
                    if response_parts:
                        response_texts.append(" | ".join(response_parts))
                    else:
                        response_texts.append(f"Raw response: {raw_content[:200]}...")
                        
                elif raw_content:
                    # Use raw content if structured response failed
                    response_texts.append(f"Analysis: {raw_content[:300]}...")
                else:
                    response_texts.append("No clear response data")
            
            if self.debug_mode:
                print(f"[SYNTHESIS DEBUG] Prepared {len(response_texts)} response summaries")
                for i, text in enumerate(response_texts[:3]):  # Show first 3
                    print(f"[SYNTHESIS DEBUG] Sample {i+1}: {text[:100]}...")
            
            # Check if we have meaningful data
            meaningful_responses = [text for text in response_texts if text and "No clear response data" not in text]
            if not meaningful_responses:
                return "No meaningful frame analysis data available for synthesis"
            
            # Create synthesis prompt
            frame_analyses = chr(10).join([f"Frame {i+1}: {text}" for i, text in enumerate(response_texts) if text and "No clear response data" not in text])
            
            synthesis_prompt = f"""You are analyzing video content. A user asked: "{self.user_query}"

You have analyzed {len(meaningful_responses)} individual frames from the video. Here are the frame-by-frame observations:

{frame_analyses}

Based on these individual frame analyses, provide a comprehensive answer to the user's original question: "{self.user_query}"

Focus on:
1. The main activities/actions observed
2. Patterns and sequences over time  
3. Overall behavior and context
4. Direct answer to what the user asked

Respond in a natural, conversational way as if answering the user directly."""

            if self.debug_mode:
                print(f"[SYNTHESIS DEBUG] Created prompt with {len(meaningful_responses)} meaningful responses")
                print(f"[SYNTHESIS DEBUG] Prompt preview: {synthesis_prompt[:300]}...")

            # Call LLM for synthesis (if available)
            if self.use_real_api and self.openai_api_key and not self.dry_run:
                try:
                    client = openai.OpenAI(
                        api_key=self.openai_api_key,
                        timeout=10.0,  # Faster timeout for synthesis
                        max_retries=1   # Single retry for synthesis to maintain speed
                    )
                    response = client.chat.completions.create(
                        model="gpt-4",  # Use GPT-4 for better synthesis
                        messages=[
                            {"role": "system", "content": "You are a helpful video analysis assistant that synthesizes multiple frame observations into coherent answers."},
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        max_tokens=200,  # Slightly higher for synthesis but still optimized
                        temperature=0.0
                    )
                    
                    synthesis_answer = response.choices[0].message.content.strip()
                    
                    # Format the final answer
                    final_answer = f"""ANSWER TO: "{self.user_query}"
{'='*60}
{synthesis_answer}

[Analysis based on {len(sorted_responses)} frames over {self._get_timespan_string(sorted_responses)}]
{'='*60}"""
                    
                    return final_answer
                    
                except Exception as e:
                    print(f"[SYNTHESIS] Error calling LLM for synthesis: {e}")
                    return self._fallback_synthesis(sorted_responses)
            else:
                return self._fallback_synthesis(sorted_responses)
                
        except Exception as e:
            return f"Error generating synthesis: {e}"
    
    def _fallback_synthesis(self, sorted_responses) -> str:
        """Fallback synthesis without LLM."""
        activities = []
        descriptions = []
        
        for item in sorted_responses:
            response = item['response']
            if isinstance(response, dict):
                if 'action' in response:
                    activities.append(response['action'])
                if 'detailed_description' in response:
                    descriptions.append(response['detailed_description'])
        
        # Create basic synthesis
        unique_activities = list(set(activities))
        
        fallback_answer = f"""ANSWER TO: "{self.user_query}"
{'='*60}
Based on analysis of {len(sorted_responses)} frames, the main activities observed were:

{chr(10).join([f"• {activity}" for activity in unique_activities[:10]])}

The user appears to be engaged in {', '.join(unique_activities[:3]) if unique_activities else 'various activities'}.

[Basic analysis - LLM synthesis not available]
{'='*60}"""
        
        return fallback_answer
    
    def _get_timespan_string(self, sorted_responses) -> str:
        """Get formatted timespan string."""
        if len(sorted_responses) <= 1:
            return "single moment"
        
        start_time = datetime.fromtimestamp(sorted_responses[0]['timestamp'])
        end_time = datetime.fromtimestamp(sorted_responses[-1]['timestamp'])
        duration = end_time - start_time
        
        return f"{duration.total_seconds():.0f} seconds"
    
    def _update_cost_tracking(self, response: dict):
        """
        Update cost tracking based on API response usage.
        
        Args:
            response: API response dict containing usage information
        """
        try:
            usage = response.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            
            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            
            # Calculate cost for this request (convert from per-1M-tokens to per-token)
            prompt_cost = (prompt_tokens / 1_000_000) * self.gpt4o_input_cost
            completion_cost = (completion_tokens / 1_000_000) * self.gpt4o_output_cost
            request_cost = prompt_cost + completion_cost
            
            self.total_cost_usd += request_cost
            
            if self.debug_mode:
                print(f"[LLM COST] Request: ${request_cost:.4f} "
                      f"(input: {prompt_tokens} tokens, output: {completion_tokens} tokens)")
                      
        except Exception as e:
            if self.debug_mode:
                print(f"[LLM COST] Error updating cost tracking: {e}")
    
    def _process_response(self, image_path: str, response: dict):
        """
        Process LLM response based on mode (summary or alert).
        
        Args:
            image_path: Path to the image that was analyzed
            response: LLM API response
        """
        try:
            # Extract response content
            llm_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if self.debug_mode:
                print(f"[ALERT DEBUG] Raw LLM content: {llm_content[:200]}...")
            
            # Try to parse as JSON
            try:
                # Clean JSON content (remove markdown code blocks if present)
                clean_content = llm_content.strip()
                if "```json" in clean_content:
                    clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_content:
                    clean_content = clean_content.split("```")[1].strip()
                
                parsed_response = json.loads(clean_content)
                if self.debug_mode:
                    print(f"[ALERT DEBUG] Parsed JSON successfully: {parsed_response}")
            except Exception as e:
                # If not JSON, treat as plain text
                parsed_response = {"raw_response": llm_content}
                if self.debug_mode:
                    print(f"[ALERT DEBUG] JSON parsing failed: {e}")
                    print(f"[ALERT DEBUG] Cleaned content was: {clean_content[:100] if 'clean_content' in locals() else 'N/A'}...")
            
            timestamp = time.time()
            
            if self.mode == "alert":
                # Check for alerts - handle both boolean and string values
                alert_value = parsed_response.get("alert", False)
                if isinstance(alert_value, str):
                    alert_triggered = alert_value.lower() in ['true', 'yes', '1']
                else:
                    alert_triggered = bool(alert_value)
                
                if self.debug_mode:
                    print(f"[ALERT DEBUG] Raw alert field: {alert_value} (type: {type(alert_value)})")
                    print(f"[ALERT DEBUG] Alert triggered: {alert_triggered}")
                
                if alert_triggered:
                    self.alert_count += 1
                    print(f"\n🚨 ALERT TRIGGERED! ({self.alert_count})")
                    print(f"📷 Image: {Path(image_path).name}")
                    analysis_text = parsed_response.get('description', parsed_response.get('analysis', 'No description'))
                    print(f"🔍 Analysis: {analysis_text}")
                    if 'items_found' in parsed_response:
                        print(f"🎯 Items: {parsed_response['items_found']}")
                    if 'items' in parsed_response:
                        print(f"🎯 Items: {parsed_response['items']}")
                    confidence = parsed_response.get('confidence', 'N/A')
                    print(f"📊 Confidence: {confidence}")
                    print("=" * 60)
                    
                    # Send notification to web display
                    print(f"[LLM DEBUG] Checking web_display: {self.web_display} (type: {type(self.web_display)})")
                    if self.web_display:
                        notification_message = f"{analysis_text}"
                        if 'items_found' in parsed_response and parsed_response['items_found']:
                            notification_message += f" | Items: {parsed_response['items_found']}"
                        elif 'items' in parsed_response and parsed_response['items']:
                            notification_message += f" | Items: {parsed_response['items']}"
                        if confidence != 'N/A':
                            notification_message += f" | Confidence: {confidence}"
                        print(f"[LLM DEBUG] Calling web_display.add_notification with: {notification_message}")
                        self.web_display.add_notification(notification_message)
                        print(f"[LLM DEBUG] Notification sent successfully")
                    
            elif self.mode == "summary":
                # Collect responses for summary
                with self.response_lock:
                    self.collected_responses.append({
                        'timestamp': timestamp,
                        'image_path': image_path,
                        'response': parsed_response,
                        'raw_content': llm_content
                    })
                    
                    if self.debug_mode:
                        print(f"[SUMMARY] Collected response #{len(self.collected_responses)} from {Path(image_path).name}")
                        
        except Exception as e:
            if self.debug_mode:
                print(f"[RESPONSE] Error processing response: {e}")
    
    def _save_analysis(self, image_path: str, response: dict):
        """
        Save comprehensive analysis including original query, transformed prompt, and response.
        
        Args:
            image_path: Path to analyzed image
            response: LLM API response
        """
        try:
            # Generate analysis filename based on image
            image_name = Path(image_path).stem
            analysis_file = self.analysis_dir / f"{image_name}_analysis.json"
            
            # Parse LLM response content if it's JSON
            llm_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed_response = None
            try:
                parsed_response = json.loads(llm_content)
            except:
                # If not valid JSON, keep as string
                parsed_response = llm_content
            
            # Comprehensive analysis data
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "user_query": self.user_query,
                "transformation": self.transformation,
                "image_path": str(image_path),
                "image_prompt": self.image_prompt,
                "raw_response": response,
                "parsed_response": parsed_response,
                "api_used": "openai" if self.use_real_api else "stub",
                "dry_run": self.dry_run
            }
            
            # Save analysis
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            # Print summary
            if parsed_response and isinstance(parsed_response, dict):
                if "detected" in parsed_response:
                    detected = parsed_response.get("detected", False)
                    confidence = parsed_response.get("confidence", 0)
                    print(f"[LLM] Analysis: {image_name} - Detected: {detected}, Confidence: {confidence}")
                else:
                    print(f"[LLM] Analysis: {image_name} - Response: {str(parsed_response)[:100]}...")
            else:
                print(f"[LLM] Analysis: {image_name} - Saved to {analysis_file.name}")
                
        except Exception as e:
            print(f"[LLM] Error saving analysis: {e}")
    
    def _log_sent_image(self, image_path: str, frame_data=None):
        """
        Log and optionally save copy of image that was sent to LLM.
        
        Args:
            image_path: Original image path
            frame_data: Optional frame data (cv2 image array)
        """
        try:
            # Create a copy in the llm_sent directory with timestamp and sequence number
            original_name = Path(image_path).stem
            timestamp = int(time.time() * 1000)
            sent_filename = f"{original_name}_sent_{self.similarity_stats['sent_to_llm']:03d}_{timestamp}.jpg"
            sent_path = self.llm_sent_dir / sent_filename
            
            # Save the image copy
            if frame_data is not None:
                # Use the frame data directly (more reliable)
                import cv2
                cv2.imwrite(str(sent_path), frame_data)
                print(f"[LLM] 📁 Saved LLM copy: {sent_path.name}")
            else:
                # Fall back to copying the original file
                import shutil
                if Path(image_path).exists():
                    shutil.copy2(image_path, sent_path)
                    print(f"[LLM] 📁 Copied LLM image: {sent_path.name}")
                else:
                    print(f"[LLM] ⚠️  Could not copy LLM image: {image_path} (file not found)")
                    
        except Exception as e:
            print(f"[LLM] Error logging sent image: {e}")
    
    def enqueue_to_llm(self, prompt, image_path, frame_data=None, motion_ratio=None):
        """
        Enqueue an LLM request for background processing with smart prioritization.
        
        Args:
            prompt: Text prompt to send
            image_path: Path to image file
            frame_data: Optional cv2 image array (to avoid race condition with file I/O)
            motion_ratio: Optional motion ratio for prioritization (0.0-1.0)
            
        Returns:
            bool: True if successfully queued, False if dropped (queue full, temporal skip, or low priority)
        """
        self.similarity_stats['total_requested'] += 1
        
        # Smart motion-based filtering
        if motion_ratio is not None:
            # Skip very low motion frames to reduce processing load
            if motion_ratio < self.motion_threshold_low:
                if self.debug_mode and self.similarity_stats['total_requested'] <= 10:
                    print(f"[MOTION] Skipped low motion frame ({motion_ratio:.3f} < {self.motion_threshold_low})")
                return False
            
            # For high motion frames, reduce temporal threshold for more responsive processing
            temporal_threshold = self.temporal_threshold_ms
            if motion_ratio > self.motion_threshold_high:
                temporal_threshold = max(250, self.temporal_threshold_ms // 2)  # Half the time for high motion
        else:
            temporal_threshold = self.temporal_threshold_ms
        
        # Temporal deduplication - skip frames too close in time
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self.last_sent_timestamp < temporal_threshold:
            if self.debug_mode and self.similarity_stats['total_requested'] <= 10:
                time_diff = current_time - self.last_sent_timestamp
                print(f"[TEMPORAL] Skipped frame - too soon ({time_diff:.0f}ms < {temporal_threshold:.0f}ms)")
            return False
        
        # Update timestamp for temporal tracking
        self.last_sent_timestamp = current_time
        
        # DEBUG: Always print first few calls
        if self.debug_mode and self.similarity_stats['total_requested'] <= 5:
            print(f"[DEBUG] enqueue_to_llm called #{self.similarity_stats['total_requested']}: {image_path}")
        
        # Similarity checking moved to worker threads for better performance
        
        try:
            # Queue immediately - similarity checking happens in worker threads
            self.request_queue.put((prompt, image_path, frame_data), block=False)
            
            if self.debug_mode and self.similarity_stats['total_requested'] <= 3:
                print(f"[DEBUG] Successfully queued #{self.similarity_stats['total_requested']}: {image_path}")
            
            return True
            
        except queue.Full:
            print(f"[LLM] Queue full, dropping request for {image_path}")
            return False
    
    def _check_similarity_and_add(self, image_path, frame_data):
        """
        Check similarity against previously sent images and add to history if unique.
        This runs in worker threads to avoid blocking main video processing.
        
        Args:
            image_path: Path to image file
            frame_data: Optional cv2 image array
            
        Returns:
            bool: True if image is unique enough to process, False if too similar
        """
        # Extract features from new image
        image_source = frame_data if frame_data is not None else image_path
        new_features = extract_image_features_fast(image_source)
        
        # Thread-safe similarity checking
        with self.similarity_lock:
            # Check similarity against ALL previously sent images
            if len(self.sent_image_features) > 0:
                # Calculate similarities to all previous images
                similarities = [np.dot(new_features, prev_features) for prev_features in self.sent_image_features]
                max_similarity = max(similarities)
                
                if self.debug_mode and self.similarity_stats['sent_to_llm'] <= 10:
                    print(f"[WORKER] Max similarity: {max_similarity:.4f}, threshold: {self.similarity_threshold}")
                
                if max_similarity > self.similarity_threshold:
                    self.similarity_stats['skipped_similar'] += 1
                    similar_idx = similarities.index(max_similarity)
                    if self.debug_mode:
                        print(f"[WORKER] SKIPPED: Image too similar ({max_similarity:.3f} > {self.similarity_threshold}) to image #{similar_idx+1} - {image_path}")
                    return False
            
            # Image is unique enough - add to history
            self.sent_image_features.append(new_features)
            
            # Keep only the most recent images (to prevent memory issues)
            if len(self.sent_image_features) > self.max_history_size:
                self.sent_image_features.pop(0)  # Remove oldest
            
            self.similarity_stats['sent_to_llm'] += 1
            
            if self.debug_mode:
                print(f"[WORKER] ACCEPTED: Image unique enough for processing ({self.similarity_stats['sent_to_llm']}/{self.similarity_stats['total_requested']})")
            
            return True
    
    def get_queue_size(self):
        """Get current queue size."""
        return self.request_queue.qsize()
    
    def get_stats(self):
        """Get processing statistics including similarity deduplication."""
        elapsed = time.time() - self.start_time
        total_requested = self.similarity_stats['total_requested']
        skipped_similar = self.similarity_stats['skipped_similar']
        efficiency_percent = (skipped_similar / total_requested * 100) if total_requested > 0 else 0
        
        return {
            "processed": self.requests_processed,
            "failed": self.requests_failed,
            "rate_limited": self.requests_rate_limited,
            "queue_size": self.get_queue_size(),
            "elapsed_seconds": elapsed,
            "rate_per_minute": (self.requests_processed / elapsed * 60) if elapsed > 0 else 0,
            "active_workers": self.num_workers,
            "similarity_dedup": {
                "total_requested": total_requested,
                "skipped_similar": skipped_similar,
                "sent_to_llm": self.similarity_stats['sent_to_llm'],
                "efficiency_percent": efficiency_percent,
                "threshold": self.similarity_threshold
            },
            "cost_tracking": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_cost_usd": self.total_cost_usd,
                "model": "gpt-4o",
                "input_cost_per_1m": self.gpt4o_input_cost,
                "output_cost_per_1m": self.gpt4o_output_cost
            }
        }
    
    def wait_for_completion(self, timeout=30):
        """
        Wait for all queued requests to be processed.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if all requests processed, False if timeout
        """
        try:
            # Wait for queue to be empty
            start_time = time.time()
            while not self.request_queue.empty() and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            # Wait for all tasks to be marked as done
            self.request_queue.join()
            return True
            
        except Exception as e:
            print(f"[LLM] Error waiting for completion: {e}")
            return False
    
    def shutdown(self, wait_for_completion=True, timeout=10):
        """
        Shutdown the LLM sink gracefully.
        
        Args:
            wait_for_completion: Whether to wait for pending requests
            timeout: Maximum time to wait for completion
        """
        if not self.running:
            return
        
        print("[LLM] Shutting down...")
        
        # Wait for pending requests if requested
        if wait_for_completion and not self.request_queue.empty():
            print(f"[LLM] Waiting for {self.get_queue_size()} pending requests...")
            self.wait_for_completion(timeout)
        
        # Stop worker threads
        self.running = False
        
        # Wait for all workers to finish
        for i, worker in enumerate(self.worker_threads):
            if worker.is_alive():
                worker.join(timeout=2)
                if worker.is_alive():
                    print(f"[LLM] Worker {i} did not shut down gracefully")
        
        # Print final statistics
        stats = self.get_stats()
        dedup_stats = stats['similarity_dedup']
        cost_stats = stats['cost_tracking']
        print(f"[LLM] Final stats: {stats['processed']} processed, "
              f"{stats['failed']} failed, "
              f"{stats.get('rate_limited', 0)} rate limited, "
              f"{stats['rate_per_minute']:.1f} req/min")
        print(f"[LLM] Similarity dedup: {dedup_stats['skipped_similar']}/{dedup_stats['total_requested']} "
              f"({dedup_stats['efficiency_percent']:.1f}%) skipped as duplicates (checked against {len(self.sent_image_features)} unique images)")
        if cost_stats['total_cost_usd'] > 0:
            print(f"[LLM] Total cost: ${cost_stats['total_cost_usd']:.4f} ({cost_stats['total_tokens']:,} tokens)")
        
        print("[LLM] Shutdown complete")