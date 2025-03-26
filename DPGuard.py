import base64
import os
from http.client import responses

from Model import *
from PIL import Image
import requests
import time
import torch
from torchvision import transforms
from google import genai
from google.genai import types

def load_parallel_model(model_path,devices="cpu"):
    # Load the entire model
    parallel_model = torch.load(model_path, map_location=torch.device(devices))

    # Check if it's a DataParallel model
    if isinstance(parallel_model, torch.nn.DataParallel):
        model = parallel_model.module
    else:
        model = parallel_model

    model.to(devices)
    model.eval()

    return model

def encode_images(image_path_list):
    res = []
    base64_images =[base64.b64encode(open(each, "rb").read()).decode("utf-8") for each in image_path_list]
    for each in base64_images:
        data = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{each}"
                    }
        }
        res.append(data)
    return res

def inquiry_GPT(image_path_list,text_prompt,model_name="gpt-4o"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key == None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable via export OPENAI_API_KEY='your-key'")

    system_prompt = open("system_prompt.txt", "r").read()
    image_prompt = encode_images(image_path_list)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
        }
    ]
    prompt_data[-1]["content"] += image_prompt

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            }
        ],
        "max_tokens": 512,
    }

    payload["messages"] += prompt_data

    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            out = response.json()['choices'][0]['message']['content']
            return out
        except KeyError:
            time.sleep(0.5)

def inquiry_gemini(image_path_list,text_prompt,model_name="gemini-2.5-pro-exp-03-25"):
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if api_key == None:
        raise ValueError("Please set the GOOGLE_GEMINI_API_KEY environment variable via export GOOGLE_GEMINI_API_KEY='your-key'")

    system_prompt = open("system_prompt.txt", "r").read()

    client = genai.Client(api_key = api_key)
    image_prompt = [Image.open(each) for each in image_path_list]

    while True:
        try:
            response = client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt),
                contents=[text_prompt]+image_prompt
            )
            return response.text
        except:
            time.sleep(60)


class DPGuard():
    def __init__(self,binary_model_path,mllm_model="gpt-4o",devices="cuda" if torch.cuda.is_available() else "cpu"):
        self.binary_model = load_parallel_model(binary_model_path,devices)
        self.mllm_model = mllm_model
        self.transformer = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.devices = devices
        self.text_prompt = open("text_prompt.txt", "r").read()


    def detect(self,image_path):
        if self.binary_model(self.transformer(Image.open(image_path)).unsqueeze(0).to(self.devices))>0.5:
            if "gpt" in self.mllm_model:
                response = inquiry_GPT([image_path],self.text_prompt,self.mllm_model).lower()
            elif "gemini" in self.mllm_model:
                response = inquiry_gemini([image_path],self.text_prompt,self.mllm_model).lower()
            '''
            Todo: Feel free to add more model in here
            '''

            if "no dp" in response:
                return """```\n(no dp, no reason as no deceptive pattern detected)\n```"""
            else:
                return f"""```\n{response}\n```"""
        else:
            return """```\n(no dp, no reason as no deceptive pattern detected)\n```"""