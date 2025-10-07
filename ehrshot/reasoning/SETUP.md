# Azure Reasoning Pipeline Setup

This guide explains how to securely configure your Azure OpenAI credentials for the reasoning pipeline.

## 🔐 Security Best Practices

**Never commit API keys to git!** This repository includes `.gitignore` to prevent accidental exposure of sensitive files.

## 🚀 Setup Options

### Option 1: Environment Variables (Recommended)

Set these environment variables in your shell:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-actual-api-key-here"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_MODEL="gpt-4o"
```

To make them permanent, add to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export AZURE_OPENAI_API_KEY="your-actual-api-key-here"' >> ~/.bashrc
echo 'export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"' >> ~/.bashrc
```

### Option 2: Configuration File

1. Copy the template:
   ```bash
   cp azure_config_template.json azure_config.json
   ```

2. Edit `azure_config.json` with your actual credentials:
   ```json
   {
     "endpoint": "https://your-actual-endpoint.openai.azure.com/",
     "api_key": "your-actual-api-key-here",
     "api_version": "2024-12-01-preview",
     "deployment": "gpt-4o",
     "model": "gpt-4o",
     "max_tokens": 4096,
     "temperature": 0.7,
     "top_p": 1.0
   }
   ```

3. The file `azure_config.json` is automatically ignored by git.

## ✅ Test Your Setup

Run the connection test:

```bash
python azure_test.py
```

You should see a successful response from Azure OpenAI.

## 🚨 Security Notes

- ✅ `azure_config.json` is in `.gitignore` - won't be committed
- ✅ Environment variables are secure
- ✅ Template file has placeholder values only
- ❌ Never put real API keys in `azure_test.py` or any other code files
- ❌ Never commit `azure_config.json` to git

## 🔧 Troubleshooting

### "No valid Azure OpenAI API key found!"

This means neither environment variables nor `azure_config.json` contain valid credentials.

**Solutions:**
1. Check environment variables: `echo $AZURE_OPENAI_API_KEY`
2. Verify `azure_config.json` exists and has correct format
3. Ensure API key is not the placeholder "your-api-key-here"

### Connection Errors

- Verify your endpoint URL is correct
- Check that your API key has access to the specified deployment
- Ensure the deployment name matches your Azure setup
