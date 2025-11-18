
from langchain_anthropic import ChatAnthropic
import os
import dotenv 

dotenv.load_dotenv()

def create(): 
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    timeout = 60
    claude_model = ChatAnthropic(api_key=anthropic_api_key, model='claude-3-opus-20240229', timeout=timeout)

    model_dict = {}
    model_dict['claude'] = claude_model

    return model_dict