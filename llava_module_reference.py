# llava_module.py

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

class LLaVAModel:
    def __init__(self):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )

    def process_image_and_text(self, image, prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(formatted_prompt, image, return_tensors="pt").to("cuda:0")
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)

    @staticmethod
    def load_image_from_url(url):
        return Image.open(requests.get(url, stream=True).raw)

# Example usage (can be commented out or removed in production)
if __name__ == "__main__":
    llava_model = LLaVAModel()
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = LLaVAModel.load_image_from_url(url)
    result = llava_model.process_image_and_text(image, "What is shown in this image?")
    print(result)