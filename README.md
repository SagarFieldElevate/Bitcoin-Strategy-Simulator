# Bitcoin Strategy Simulator

An advanced Monte Carlo simulation engine for Bitcoin trading strategies with GARCH+jumps modeling and multi-factor correlation analysis.

## Features

- **Monte Carlo Simulation**: Advanced simulation with GARCH(1,1) + jump diffusion model
- **Multi-Factor Analysis**: Correlate Bitcoin with 40+ macro-economic variables
- **Portfolio Optimization**: Combine multiple strategies with custom weights
- **Market Scenarios**: Test strategies under different market conditions
- **Real-time Data**: Integration with Pinecone vector database for strategy data

## Quick Start

### Local Development

1. Clone the repository
```bash
git clone <your-repo-url>
cd Bitcoin-Strategy-Simulator-Improvements
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set environment variables
```bash
export PINECONE_API_KEY="your_pinecone_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

4. Run the app
```bash
streamlit run app.py
```

### Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions on Render.

## Configuration

The app requires two API keys:
- **Pinecone API Key**: For accessing strategy data
- **OpenAI API Key**: For AI-powered features

Create a `.env` file based on `.env.example` and add your keys.

## Usage

1. **Load Bitcoin Data**: Click "Load Bitcoin Data" to fetch historical prices
2. **Select Strategy**: Choose from 600+ correlation strategies
3. **Configure Simulation**:
   - Number of simulations (default: 1000)
   - Simulation period in days (default: 30)
   - Market scenario (baseline or stress conditions)
4. **Run Simulation**: Execute Monte Carlo simulation
5. **Analyze Results**: View price paths, CAGR distribution, and risk metrics

## Technical Details

### Simulation Engine
- GARCH(1,1) model for volatility clustering
- Jump diffusion for extreme events
- Vectorized NumPy implementation
- Parallel processing with ThreadPoolExecutor

### Risk Metrics
- CAGR (Compound Annual Growth Rate)
- Maximum Drawdown
- Sharpe Ratio
- Win Rate
- Percentile statistics

## Future Improvements

See [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) for planned enhancements including:
- Sophisticated regime detection
- Enhanced data quality monitoring
- Adaptive model parameters
- Better strategy validation
- Performance profiling

## License

This project is proprietary software. All rights reserved.

## Support

For issues or questions, please open an issue in the repository.