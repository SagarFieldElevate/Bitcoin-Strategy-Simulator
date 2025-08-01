# Deployment Guide for Bitcoin Strategy Simulator

This guide covers deploying the Bitcoin Strategy Simulator to Render.

## Pre-Deployment Checklist

### 1. Environment Variables
**CRITICAL**: Remove API keys from `.streamlit/secrets.toml` before pushing to any public repository!

The app requires the following environment variables:
- `PINECONE_API_KEY`: Your Pinecone API key
- `OPENAI_API_KEY`: Your OpenAI API key

### 2. Files Created for Deployment
- `requirements.txt` - Python dependencies
- `render.yaml` - Render configuration
- `.env.example` - Example environment variables
- `.gitignore` - Excludes sensitive files
- `runtime.txt` - Python version specification

## Deployment Steps for Render

### 1. Prepare Your Repository

```bash
# Remove sensitive files from tracking
git rm --cached .streamlit/secrets.toml
git rm --cached check_connections.py

# Add all deployment files
git add requirements.txt render.yaml .env.example .gitignore runtime.txt DEPLOYMENT.md
git commit -m "Add deployment configuration for Render"
git push origin main
```

### 2. Create Render Account
1. Sign up at [https://render.com](https://render.com)
2. Connect your GitHub account

### 3. Deploy on Render

#### Option A: Using render.yaml (Recommended)
1. Click "New +" → "Blueprint"
2. Connect your GitHub repository
3. Render will automatically detect `render.yaml`
4. Review the configuration and click "Apply"

#### Option B: Manual Setup
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: bitcoin-strategy-simulator
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### 4. Configure Environment Variables
In Render dashboard:
1. Go to your service → "Environment"
2. Add the following variables:
   - `PINECONE_API_KEY`: [Your Pinecone API key]
   - `OPENAI_API_KEY`: [Your OpenAI API key]

### 5. Deploy
1. Click "Create Web Service" or "Apply" (if using Blueprint)
2. Wait for the build to complete (5-10 minutes)
3. Your app will be available at: `https://your-app-name.onrender.com`

## Important Security Notes

### API Keys
**WARNING**: Your `.streamlit/secrets.toml` file contains exposed API keys. You should:
1. Regenerate these API keys immediately
2. Never commit API keys to version control
3. Use environment variables for production

### Secure Your Repository
If your repository is public:
```bash
# Check git history for sensitive data
git log --all --full-history -- .streamlit/secrets.toml

# If keys were committed, you need to:
# 1. Rotate all API keys immediately
# 2. Consider using git-filter-branch or BFG Repo-Cleaner to remove from history
```

## Configuration Details

### Streamlit Configuration
The app is configured in `.streamlit/config.toml`:
- Runs on port specified by `$PORT` environment variable
- Accepts connections from any address (0.0.0.0)
- Dark theme with custom colors

### Resource Requirements
- **Memory**: ~512MB minimum (1GB recommended for larger simulations)
- **CPU**: Scales with number of simulations
- **Storage**: Minimal (app doesn't store persistent data)

## Monitoring and Logs

### View Logs
In Render dashboard:
1. Go to your service
2. Click "Logs" tab
3. Monitor for errors during startup

### Common Issues

1. **Port Binding Error**
   - Ensure using `$PORT` environment variable
   - Don't hardcode port numbers

2. **Module Import Errors**
   - Check all dependencies in requirements.txt
   - Verify Python version compatibility

3. **API Connection Errors**
   - Verify environment variables are set
   - Check API key validity
   - Ensure Pinecone index exists

## Performance Optimization

For production deployment:

1. **Caching**: Streamlit's `@st.cache_data` is already implemented
2. **Concurrent Users**: Consider upgrading Render plan for more resources
3. **Simulation Limits**: May need to limit max simulations on free tier

## Alternative Deployment Options

### Streamlit Community Cloud
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from GitHub
4. Add secrets in Streamlit Cloud dashboard

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Local Testing
Before deploying:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PINECONE_API_KEY="your_key"
export OPENAI_API_KEY="your_key"

# Run locally
streamlit run app.py
```

## Post-Deployment

1. **Test All Features**
   - Load Bitcoin data
   - Run single strategy simulation
   - Run portfolio simulation
   - Check all visualizations

2. **Monitor Usage**
   - Watch Render metrics
   - Monitor API usage (OpenAI/Pinecone)
   - Check error logs

3. **Set Up Alerts**
   - Configure uptime monitoring
   - Set up error notifications

## Support

For deployment issues:
- Render: [render.com/docs](https://render.com/docs)
- Streamlit: [docs.streamlit.io](https://docs.streamlit.io)

For app issues:
- Check logs in Render dashboard
- Verify all environment variables
- Ensure API services are accessible