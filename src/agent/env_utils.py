from dotenv import load_dotenv
import os

load_dotenv(override=True)

DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")

ZHIPU_API_KEY=os.getenv("ZHIPU_API_KEY")