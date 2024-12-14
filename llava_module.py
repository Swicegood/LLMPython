# llava_module.py

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import base64
import io
import logging
import re
import gc

logger = logging.getLogger(__name__)

class LLaVAModel:
    def __init__(self):
        # Configure GPU memory settings
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # Load model with optimized memory settings
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            device_map="auto"  # Automatically handle multi-GPU or CPU fallback
        )
        
        # Set default batch size and sequence length
        self.max_batch_size = 1
        self.max_sequence_length = 512

    def _ensure_gpu_memory(self):
        """Ensure sufficient GPU memory is available"""
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get current GPU memory usage
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            memory_free = torch.cuda.get_device_properties(current_device).total_memory / 1024**3 - memory_allocated
            
            logger.info(f"GPU Memory Status - Allocated: {memory_allocated:.2f}GB, "
                       f"Reserved: {memory_reserved:.2f}GB, Free: {memory_free:.2f}GB")
            
            return memory_free > 2.0  # Ensure at least 2GB free
        return False

    def process_image_and_text(self, image, prompt):
        try:
            if not self._ensure_gpu_memory():
                logger.warning("Low GPU memory, attempting to optimize...")
                torch.cuda.empty_cache()
                gc.collect()
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt[:self.max_sequence_length]},  # Truncate if needed
                        {"type": "image"},
                    ],
                },
            ]
            
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(
                    formatted_prompt, 
                    image, 
                    return_tensors="pt"
                ).to("cuda:0", non_blocking=True)  # Non-blocking transfer
                
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,  # Disable sampling for memory efficiency
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                raw_response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Clear memory after processing
            del inputs, output
            torch.cuda.empty_cache()
            
            return self.clean_response(raw_response)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM error in process_image_and_text: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            return "Error: Insufficient GPU memory. Please try again with a shorter prompt or reduced image size."
        except Exception as e:
            logger.error(f"Error in process_image_and_text: {str(e)}")
            return None

    def process_text_only(self, prompt):
        try:
            if not self._ensure_gpu_memory():
                logger.warning("Low GPU memory, attempting to optimize...")
                torch.cuda.empty_cache()
                gc.collect()
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt[:self.max_sequence_length]},
                    ],
                },
            ]
            
            with torch.cuda.amp.autocast():
                formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(
                    formatted_prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length
                ).to("cuda:0", non_blocking=True)
                
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                raw_response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Clear memory after processing
            del inputs, output
            torch.cuda.empty_cache()
            
            return self.clean_response(raw_response)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM error in process_text_only: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            return "Error: Insufficient GPU memory. Please try again with a shorter prompt."
        except Exception as e:
            logger.error(f"Error in process_text_only: {str(e)}")
            return None

    def clean_response(self, response):
        pattern = r'.*\[/INST\]'
        cleaned = re.sub(pattern, '', response, flags=re.DOTALL).strip()
        return cleaned if cleaned else response

    def process_request(self, messages):
        try:
            text_prompt = None
            image = None
            
            if not isinstance(messages, list) or not messages:
                raise ValueError("Messages should be a non-empty list")

            for message in messages:
                if not isinstance(message, dict) or 'content' not in message:
                    raise ValueError("Each message should be a dictionary with a 'content' key")
                
                content = message['content']
                if isinstance(content, str):
                    text_prompt = content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_prompt = item.get('text')
                            elif item.get('type') == 'image_url':
                                image_url = item.get('image_url', {}).get('url')
                                if image_url:
                                    if image_url.startswith('data:image'):
                                        image_data = base64.b64decode(image_url.split(',')[1])
                                        image = Image.open(io.BytesIO(image_data))
                                    else:
                                        image = self.load_image_from_url(image_url)

            if not text_prompt:
                raise ValueError("Text prompt is required")

            if image:
                result = self.process_image_and_text(image, text_prompt)
            else:
                result = self.process_text_only(text_prompt)
                
            return result, 0.0  # Returning 0.0 as a placeholder for token usage
        except Exception as e:
            logger.error(f"Request processing error: {str(e)}")
            return None, None

    @staticmethod
    def load_image_from_url(url):
        try:
            return Image.open(requests.get(url, stream=True).raw)
        except Exception as e:
            logger.error(f"Error loading image from URL: {str(e)}")
            return None

# Example usage (can be commented out or removed in production)
if __name__ == "__main__":
    llava_model = LLaVAModel()
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = LLaVAModel.load_image_from_url(url)
    result = llava_model.process_image_and_text(image, "What is shown in this image?")
    print(result)