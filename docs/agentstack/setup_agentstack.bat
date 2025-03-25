@echo off
REM AgentStack Setup Script for Know-Defeat (Windows version)
REM This script helps set up AgentStack for use with the Know-Defeat trading system

REM Print header
echo ================================================================
echo           AgentStack Setup for Know-Defeat                     
echo ================================================================
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed or not in PATH. Please install Anaconda or Miniconda.
    exit /b 1
)

REM Activate the Autogen environment
echo Activating Autogen conda environment...
call conda activate Autogen

if %ERRORLEVEL% neq 0 (
    echo Failed to activate Autogen environment. Please ensure it exists.
    echo You can create it with: conda create -n Autogen python=3.10
    exit /b 1
)

echo Successfully activated Autogen environment.
echo.

REM Install AgentStack
echo Installing AgentStack...
pip install agentstack

if %ERRORLEVEL% neq 0 (
    echo Failed to install AgentStack.
    exit /b 1
)

echo Successfully installed AgentStack.
echo.

REM Check AgentStack version
echo Checking AgentStack version...
agentstack --version
echo.

REM Create AgentStack project directory
echo Creating AgentStack project directory...
mkdir agents
cd agents

REM Initialize AgentStack project
echo Initializing AgentStack project...
echo Note: You will be prompted to configure your project during initialization.
echo.
echo Press Enter to continue...
pause >nul

agentstack init know_defeat_agents --wizard

if %ERRORLEVEL% neq 0 (
    echo Failed to initialize AgentStack project.
    exit /b 1
)

echo Successfully initialized AgentStack project.
echo.

REM Copy sample configuration
echo Copying sample configuration...
cd know_defeat_agents
copy ..\..\docs\agentstack\sample_config.yaml .\agentstack.yaml

echo Sample configuration copied.
echo.

REM Create tool directories
echo Creating tool directories...
mkdir tools\risk
mkdir tools\orders
mkdir tools\performance
mkdir tools\patterns
mkdir tools\sentiment

echo Tool directories created.
echo.

REM Install dependencies
echo Installing required dependencies...
pip install pandas numpy ta asyncpg matplotlib seaborn

echo Dependencies installed.
echo.

REM Setup complete
echo ================================================================
echo AgentStack setup complete!
echo ================================================================
echo.
echo Next steps:
echo 1. Review the sample configuration in agents\know_defeat_agents\agentstack.yaml
echo 2. Generate your first trading agent with: agentstack generate agent market_analyzer
echo 3. Create your first task with: agentstack generate task analyze_market
echo 4. Implement your custom tools in the tools\ directory
echo 5. Run your agent project with: agentstack run
echo.
echo For more information, see the AgentStack documentation in:
echo docs\agentstack\README.md
echo docs\agentstack\trading_integration.md
echo.
echo Happy coding! 