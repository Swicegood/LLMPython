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
        try:
            # Configure GPU memory settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            
            # Log GPU information
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
            
            # Load processor first
            logger.info("Loading LLaVA processor...")
            self.processor = None  # Initialize to None for checking
            try:
                self.processor = LlavaNextProcessor.from_pretrained(
                    "llava-hf/llava-v1.6-mistral-7b-hf"
                )
                logger.info("Processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load processor: {str(e)}")
                raise
                
            if self.processor is None:
                raise RuntimeError("Processor is None after loading")
                
            # Load model with explicit verification
            logger.info("Loading LLaVA model...")
            self.model = None  # Initialize to None for checking
            try:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-v1.6-mistral-7b-hf",
                    device_map="auto",
                    quantization_config={
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": "float16",
                        "bnb_4bit_quant_type": "fp4",
                        "bnb_4bit_use_double_quant": True
                    }
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
                
            if self.model is None:
                raise RuntimeError("Model is None after loading")
            
            # Verify model components with correct names
            logger.info("Verifying model components...")
            if not hasattr(self.model, 'vision_tower'):
                raise RuntimeError("Model missing vision_tower component")
            if not hasattr(self.model.vision_tower, 'vision_model'):
                raise RuntimeError("Model missing vision_tower.vision_model component")
            if not hasattr(self.model, 'language_model'):
                raise RuntimeError("Model missing language_model component")
            
            
            # Check if model is on CUDA
            logger.info(f"Model device mapping: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'Unknown'}")
            
            # Print model's memory usage
            if hasattr(self.model, 'get_memory_footprint'):
                logger.info(f"Model memory footprint: {self.model.get_memory_footprint() / 1024**2:.2f}MB")
            
            # Force model to eval mode
            self.model.eval()
            logger.info("Model set to eval mode")
            
            # Set default parameters
            self.max_batch_size = 1
            self.max_sequence_length = 512
            
            logger.info("Initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            # Clean up if initialization fails
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            torch.cuda.empty_cache()
            raise

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

    def _verify_model_dtype(self):
        """Verify model dtype configuration"""
        try:
            # Get model's parameter dtype
            param_dtype = next(self.model.parameters()).dtype
            logger.info(f"Model parameter dtype: {param_dtype}")
            
            # Get model's config dtype
            config_dtype = getattr(self.model.config, 'torch_dtype', None)
            logger.info(f"Model config dtype: {config_dtype}")
            
            # Check vision model dtype
            vision_dtype = next(self.model.vision_model.parameters()).dtype
            logger.info(f"Vision model dtype: {vision_dtype}")
            
            # Check language model dtype
            lang_dtype = next(self.model.language_model.parameters()).dtype
            logger.info(f"Language model dtype: {lang_dtype}")
            
            return True
        except Exception as e:
            logger.error(f"Error checking model dtypes: {str(e)}")
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

    def verify_model_state(self):
        """Verify the current state of the model"""
        try:
            if self.model is None:
                logger.error("Model is None")
                return False
                
            # Try to access key attributes
            logger.info("Verifying model state...")
            logger.info(f"Model type: {type(self.model)}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model requires grad: {any(p.requires_grad for p in self.model.parameters())}")
            
            # Create a proper dummy input sequence
            dummy_text = "Hello"
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dummy_text},
                    ],
                },
            ]
            
            # Process through the pipeline
            try:
                formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                dummy_inputs = self.processor(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length
                )
                
                # Move to CUDA
                cuda_inputs = {
                    k: v.to('cuda:0') for k, v in dummy_inputs.items() if torch.is_tensor(v)
                }
                
                logger.info("Attempting dummy forward pass...")
                with torch.no_grad():
                    outputs = self.model(**cuda_inputs)
                    logger.info("Dummy forward pass successful")
                    logger.info(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
                
                del dummy_inputs, cuda_inputs, outputs
                torch.cuda.empty_cache()
                return True
                
            except Exception as e:
                logger.error(f"Failed to process dummy input: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Model state verification failed: {str(e)}")
            return False

    def process_text_only(self, prompt):
        try:
            # Verify model state first
            logger.info("Verifying model state before processing...")
            if not self.verify_model_state():
                logger.error("Model verification failed")
                return "Error: Model not in valid state"
            
            if prompt is None:
                logger.error("Prompt is None")
                return "Error: Prompt cannot be None"

            logger.info(f"Processing prompt: {prompt[:100]}...")  # Log first 100 chars
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt[:self.max_sequence_length]},
                    ],
                },
            ]

            # Format the prompt
            formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            logger.info("Chat template applied successfully")
            
            # Process inputs
            inputs = self.processor(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length
            )
            logger.info(f"Input tensor shapes: {{k: v.shape for k, v in inputs.items() if torch.is_tensor(v)}}")
            
            # Move to CUDA
            cuda_inputs = {
                k: v.to('cuda:0') for k, v in inputs.items() if torch.is_tensor(v)
            }
            
            # Generate
            try:
                with torch.inference_mode():
                    self.model.eval()  # Ensure eval mode
                    outputs = self.model.generate(
                        **cuda_inputs,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                    logger.info("Generation successful")
                    
                    # Decode the output
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                    response = self.clean_response(response)
                    logger.info(f"Response length: {len(response)}")
                    
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return f"Error in generation: {str(e)}"
            finally:
                # Clean up
                del inputs, cuda_inputs
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_text_only: {str(e)}")
            return f"Error: {str(e)}"
        
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