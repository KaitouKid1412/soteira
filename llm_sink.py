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

from utils.prompt_transformer import LLMPromptTransformer, SimplePromptTransformer


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def send_to_openai_vision(prompt: str, image_path: str, api_key: str, 
                         model: str = "gpt-4o") -> dict:
    """
    Send image and prompt to OpenAI GPT-4 Vision API.
    
    Args:
        prompt: Text prompt for analysis
        image_path: Path to image file
        api_key: OpenAI API key
        model: Model to use (default: gpt-4o)
        
    Returns:
        API response dict
    """
    try:
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
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
            max_tokens=500,
            temperature=0.1  # Low temperature for consistent analysis
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
                 dry_run: bool = False, max_queue_size: int = 100, num_workers: int = 4):
        """
        Initialize LLM sink.
        
        Args:
            user_query: Original user query (e.g., "find humans", "check for cigarettes")
            openai_api_key: OpenAI API key (None for stub mode)
            dry_run: If True, don't actually call LLM (just log)
            max_queue_size: Maximum number of queued requests
            num_workers: Number of parallel worker threads
        """
        self.user_query = user_query
        self.openai_api_key = openai_api_key
        self.dry_run = dry_run
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Initialize prompt transformer
        if openai_api_key and not dry_run:
            self.prompt_transformer = LLMPromptTransformer(openai_api_key)
            self.use_real_api = True
        else:
            self.prompt_transformer = SimplePromptTransformer()
            self.use_real_api = False
        
        # Transform the user query once
        self.transformation = self.prompt_transformer.transform_prompt(user_query)
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
        
        # Statistics
        self.requests_processed = 0
        self.requests_failed = 0
        self.requests_rate_limited = 0
        self.start_time = time.time()
        
        # Create analysis directory
        self.analysis_dir = Path("events") / "llm_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
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
        while self.running:
            try:
                # Get request from queue (with timeout)
                try:
                    request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                _, image_path = request  # We now ignore the original prompt and use transformed one
                
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
                        response = send_to_gpt_stub(self.image_prompt, image_path)
                    else:
                        # Use real API or stub based on setup
                        if self.use_real_api and self.openai_api_key:
                            response = self._call_openai_with_retry(image_path, worker_id)
                        else:
                            response = send_to_gpt_stub(self.image_prompt, image_path)
                    
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
    
    def _call_openai_with_retry(self, image_path: str, worker_id: int, max_retries: int = 3) -> dict:
        """Call OpenAI API with retry logic for rate limiting.
        
        Args:
            image_path: Path to image file
            worker_id: Worker thread ID for logging
            max_retries: Maximum number of retries
            
        Returns:
            dict: API response
        """
        for attempt in range(max_retries + 1):
            try:
                response = send_to_openai_vision(self.image_prompt, image_path, self.openai_api_key)
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limiting errors
                if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                    self.requests_rate_limited += 1
                    
                    # Extract wait time from error if possible, otherwise use exponential backoff
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    
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
    
    def enqueue_to_llm(self, prompt, image_path):
        """
        Enqueue an LLM request for background processing.
        
        Args:
            prompt: Text prompt to send
            image_path: Path to image file
            
        Returns:
            bool: True if successfully queued, False if queue is full
        """
        try:
            # Non-blocking queue put
            self.request_queue.put((prompt, image_path), block=False)
            return True
            
        except queue.Full:
            print(f"[LLM] Queue full, dropping request for {image_path}")
            return False
    
    def get_queue_size(self):
        """Get current queue size."""
        return self.request_queue.qsize()
    
    def get_stats(self):
        """Get processing statistics."""
        elapsed = time.time() - self.start_time
        return {
            "processed": self.requests_processed,
            "failed": self.requests_failed,
            "rate_limited": self.requests_rate_limited,
            "queue_size": self.get_queue_size(),
            "elapsed_seconds": elapsed,
            "rate_per_minute": (self.requests_processed / elapsed * 60) if elapsed > 0 else 0,
            "active_workers": self.num_workers
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
        print(f"[LLM] Final stats: {stats['processed']} processed, "
              f"{stats['failed']} failed, "
              f"{stats.get('rate_limited', 0)} rate limited, "
              f"{stats['rate_per_minute']:.1f} req/min")
        
        print("[LLM] Shutdown complete")