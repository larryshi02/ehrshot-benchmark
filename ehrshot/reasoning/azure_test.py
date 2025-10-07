import os
import json
from openai import AzureOpenAI

# Try to load from config file first, then fall back to environment variables
config_file = os.path.join(os.path.dirname(__file__), "azure_config.json")

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    endpoint = config["endpoint"]
    deployment = config["deployment"]
    subscription_key = config["api_key"]
    api_version = config["api_version"]
    model_name = config["model"]
else:
    # Fall back to environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://clinicalml-cloudbank-openai.openai.azure.com/")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

# Check if we have a valid API key
if subscription_key == "your-api-key-here" or not subscription_key:
    print("⚠️  WARNING: No valid Azure OpenAI API key found!")
    print("Please either:")
    print("1. Create azure_config.json with your credentials, or")
    print("2. Set the AZURE_OPENAI_API_KEY environment variable")
    print("See azure_config_template.json for the expected format.")
    exit(1)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)