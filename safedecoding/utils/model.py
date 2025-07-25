from typing import Any
from openai import OpenAI
from tenacity import retry, wait_chain, wait_fixed
import google.generativeai as genai
import boto3
import json

class GPT:
    def __init__(self, model_name, api=None, temperature=0, seed=0):
        self.model_name = model_name
        # self.client = OpenAI(
        #     api_key=api
        # )
        self.api_key = api
        
        self.T = temperature
        self.seed=seed

        # New client-style API
        self.client = OpenAI(api_key=self.api_key)
        

    def __call__(self, prompt,  n:int=1, debug=False, **kwargs: Any) -> Any:
        prompt = [{'role':'user', 'content':prompt}]
        if debug:
            return self.client.chat.completions.create(messages=prompt, n=n, model=self.model_name, temperature=self.T, seed=self.seed, **kwargs)
            
        else:
            return self.call_wrapper(messages=prompt, n=n, model=self.model_name,temperature=self.T, seed=self.seed,  **kwargs)
    

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                [wait_fixed(5) for i in range(2)] +
                [wait_fixed(10)])) # not use for debug
    def call_wrapper(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            logging.warning(f"Rate limit hit: {e}")
            raise
        except openai.APIError as e:
            logging.warning(f"API error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
    
    def resp_parse(self, response)->list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]


    

def load_model(model_name, api_idx, **kwargs):
    if "gpt" in model_name and "gpt2" not in model_name:
        return GPT(model_name, **kwargs)

    else:
        raise ValueError(f"model_name invalid")


