# Integrating AgentStack with Algorithmic Trading

This guide provides information on how to use AgentStack to enhance the Know-Defeat algorithmic trading system.

## Why AgentStack for Trading?

AgentStack provides a solid foundation for building multi-agent trading systems with the following benefits:

1. **Rapid Agent Deployment**: Quickly create specialized trading agents for different strategies, symbols, or market conditions
2. **Tool Integration**: Easily incorporate custom trading tools like technical indicators, risk management, and order execution
3. **Task Management**: Create and organize trading tasks like market analysis, signal generation, and trade execution
4. **Framework Agnostic**: Works with existing Python trading libraries and frameworks

## Getting Started with AgentStack for Know-Defeat

### Installation

Follow the standard installation instructions in the main README.md file. For Know-Defeat, it's recommended to install within the Autogen conda environment:

```bash
conda activate Autogen
pip install agentstack
```

### Project Initialization

Create a new AgentStack project for trading:

```bash
agentstack init know_defeat_agents --wizard
```

During the wizard setup:
1. Select "Trading" as the domain
2. Choose Python-based agents
3. Set up initial configuration

### Creating Trading Agents

Create specialized agents for different trading functions:

```bash
# Create a market data analyzer
agentstack generate agent market_analyzer \
  --role "Trading Market Analyzer" \
  --goal "Analyze market data and identify potential trading opportunities" \
  --backstory "An expert trading analyst with years of experience in pattern recognition"

# Create a strategy executor
agentstack generate agent strategy_executor \
  --role "Trading Strategy Executor" \
  --goal "Execute trading strategies based on signals from the analyzer" \
  --backstory "An execution specialist that can optimize order placement"

# Create a risk manager
agentstack generate agent risk_manager \
  --role "Trading Risk Manager" \
  --goal "Monitor positions and manage risk parameters" \
  --backstory "An expert in risk assessment and position sizing"
```

### Creating Trading Tasks

Set up specific tasks for your trading agents:

```bash
# Market analysis task
agentstack generate task analyze_market \
  --description "Analyze current market conditions for trading signals" \
  --expected_output "JSON object with identified patterns and signal strength" \
  --agent market_analyzer

# Order execution task
agentstack generate task execute_trade \
  --description "Execute a trade based on the provided signal" \
  --expected_output "JSON object with execution details" \
  --agent strategy_executor

# Risk assessment task
agentstack generate task assess_risk \
  --description "Evaluate risk for a potential trade" \
  --expected_output "JSON object with risk metrics" \
  --agent risk_manager
```

## Integrating with Know-Defeat

### Custom Trading Tools

Add custom tools to enhance your trading agents:

```bash
# Add technical indicator tools
agentstack tools add technical_indicators --agent market_analyzer

# Add order execution tools
agentstack tools add order_executor --agent strategy_executor

# Add risk management tools
agentstack tools add risk_calculator --agent risk_manager
```

### Creating Custom Trading Tools

Create custom tools for Know-Defeat in the `tools/` directory:

#### Example: Technical Indicator Tool

```python
# tools/technical_indicators.py
import pandas as pd
import numpy as np
import ta

def calculate_indicators(data, indicators=None):
    """
    Calculate technical indicators for the provided data
    
    Args:
        data (pd.DataFrame): OHLCV data
        indicators (list): List of indicators to calculate
        
    Returns:
        dict: Dictionary with calculated indicators
    """
    if indicators is None:
        indicators = ['sma', 'ema', 'rsi']
    
    result = {}
    
    for indicator in indicators:
        if indicator == 'sma':
            result['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
            result['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        elif indicator == 'ema':
            result['ema_20'] = ta.trend.ema_indicator(data['close'], window=20)
            result['ema_50'] = ta.trend.ema_indicator(data['close'], window=50)
        elif indicator == 'rsi':
            result['rsi_14'] = ta.momentum.rsi(data['close'], window=14)
            
    return result
```

### Database Integration

Create a tool to interact with the Know-Defeat PostgreSQL database:

```python
# tools/database.py
import asyncpg
import pandas as pd

async def fetch_market_data(ticker, timeframe, limit=1000):
    """
    Fetch market data from the database
    
    Args:
        ticker (str): Symbol to fetch data for
        timeframe (str): Timeframe (1m, 5m, 15m, etc.)
        limit (int): Number of candles to fetch
        
    Returns:
        pd.DataFrame: Dataframe with OHLCV data
    """
    conn = await asyncpg.connect(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )
    
    query = """
    SELECT timestamp, price, volume 
    FROM tick_data
    WHERE ticker = $1
      AND timestamp > NOW() - INTERVAL '1 day'
    ORDER BY timestamp DESC
    LIMIT $2;
    """
    
    result = await conn.fetch(query, ticker, limit)
    await conn.close()
    
    df = pd.DataFrame(result, columns=['timestamp', 'price', 'volume'])
    return df
```

## Integrating with Know-Defeat Bot System

Link your AgentStack agents with the existing Know-Defeat bot system:

### Example Integration

```python
# main.py
import asyncio
from agents.market_analyzer import analyze_market
from agents.strategy_executor import execute_trade
from agents.risk_manager import assess_risk
from tools.database import fetch_market_data

async def main():
    # Fetch data from Know-Defeat database
    tsla_data = await fetch_market_data('TSLA', '5m', 200)
    
    # Run market analysis
    analysis_result = await analyze_market(tsla_data)
    
    # Assess risk
    risk_assessment = await assess_risk(analysis_result)
    
    # Execute trade if conditions met
    if analysis_result.get('signal') == 'BUY' and risk_assessment.get('risk_score') < 0.5:
        execution_result = await execute_trade({
            'ticker': 'TSLA', 
            'direction': 'BUY',
            'quantity': risk_assessment.get('suggested_position_size'),
            'order_type': 'MARKET'
        })
        print(f"Trade executed: {execution_result}")
    
if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices for Trading Agents

1. **Model Selection**: Use models optimized for structured data and decision-making
2. **Prompt Engineering**: Craft prompts specific to trading scenarios
3. **Tool Organization**: Organize tools by function (analysis, execution, risk)
4. **Data Flow**: Design clear data flows between agents
5. **Evaluation**: Set up evaluation metrics specific to trading performance

## Deployment Considerations

When deploying AgentStack with Know-Defeat:

1. **Environment Management**: Use the Autogen conda environment
2. **Database Connectivity**: Ensure proper configuration to the tick_data database
3. **API Credentials**: Securely store any trading API credentials
4. **Monitoring**: Implement monitoring for agent performance
5. **Logging**: Set up comprehensive logging for debugging

## Example: Complete Trading Agent Workflow

1. **Market Data Agent**: Fetches and processes market data
2. **Analysis Agent**: Identifies patterns and generates signals
3. **Risk Management Agent**: Assesses risk and determines position sizing
4. **Execution Agent**: Places orders and monitors fills
5. **Performance Agent**: Tracks and reports on trading performance

Using this workflow with AgentStack and Know-Defeat creates a modular, maintainable multi-agent trading system.

## Resources

- [AgentStack Documentation](https://docs.agentstack.sh/)
- [Know-Defeat Database Schema](../know-defeat-db-schema.md)
- [Trading Algorithms](../../algorithms/) 