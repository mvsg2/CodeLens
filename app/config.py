from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables from a .env file
# IMPORTANT: Call this function before because os.environ 
# by default does not load the .env file automatically

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", None)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "codelens-bucket")