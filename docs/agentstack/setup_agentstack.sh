#!/bin/bash
# AgentStack Setup Script for Know-Defeat
# This script helps set up AgentStack for use with the Know-Defeat trading system

# Text formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}           AgentStack Setup for Know-Defeat                     ${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed or not in PATH. Please install Anaconda or Miniconda.${NC}"
    exit 1
fi

# Activate the Autogen environment
echo -e "${BLUE}Activating Autogen conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate Autogen

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate Autogen environment. Please ensure it exists.${NC}"
    echo -e "${BLUE}You can create it with: conda create -n Autogen python=3.10${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully activated Autogen environment.${NC}"
echo ""

# Install AgentStack
echo -e "${BLUE}Installing AgentStack...${NC}"
pip install agentstack

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install AgentStack.${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully installed AgentStack.${NC}"
echo ""

# Check AgentStack version
echo -e "${BLUE}Checking AgentStack version...${NC}"
AGENTSTACK_VERSION=$(agentstack --version 2>&1)
echo -e "${GREEN}AgentStack version: ${AGENTSTACK_VERSION}${NC}"
echo ""

# Create AgentStack project directory
echo -e "${BLUE}Creating AgentStack project directory...${NC}"
mkdir -p agents
cd agents

# Initialize AgentStack project
echo -e "${BLUE}Initializing AgentStack project...${NC}"
echo -e "${BLUE}Note: You will be prompted to configure your project during initialization.${NC}"
echo ""
echo -e "${BLUE}Press Enter to continue...${NC}"
read

agentstack init know_defeat_agents --wizard

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to initialize AgentStack project.${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully initialized AgentStack project.${NC}"
echo ""

# Copy sample configuration
echo -e "${BLUE}Copying sample configuration...${NC}"
cd know_defeat_agents
cp ../../docs/agentstack/sample_config.yaml ./agentstack.yaml

echo -e "${GREEN}Sample configuration copied.${NC}"
echo ""

# Create tool directories
echo -e "${BLUE}Creating tool directories...${NC}"
mkdir -p tools/risk
mkdir -p tools/orders
mkdir -p tools/performance
mkdir -p tools/patterns
mkdir -p tools/sentiment

echo -e "${GREEN}Tool directories created.${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}Installing required dependencies...${NC}"
pip install pandas numpy ta asyncpg matplotlib seaborn

echo -e "${GREEN}Dependencies installed.${NC}"
echo ""

# Setup complete
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}AgentStack setup complete!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "1. Review the sample configuration in ${BLUE}agents/know_defeat_agents/agentstack.yaml${NC}"
echo -e "2. Generate your first trading agent with: ${BLUE}agentstack generate agent market_analyzer${NC}"
echo -e "3. Create your first task with: ${BLUE}agentstack generate task analyze_market${NC}"
echo -e "4. Implement your custom tools in the ${BLUE}tools/${NC} directory"
echo -e "5. Run your agent project with: ${BLUE}agentstack run${NC}"
echo ""
echo -e "For more information, see the AgentStack documentation in:"
echo -e "${BLUE}docs/agentstack/README.md${NC}"
echo -e "${BLUE}docs/agentstack/trading_integration.md${NC}"
echo ""
echo -e "${BLUE}Happy coding!${NC}" 