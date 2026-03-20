from langchain_deepseek import ChatDeepSeek
from zai import ZhipuAiClient
from agent.env_utils import DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY,ZHIPU_API_KEY

deepseek_llm=ChatDeepSeek(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
)

zhipu_ai_client = ZhipuAiClient(api_key=ZHIPU_API_KEY)