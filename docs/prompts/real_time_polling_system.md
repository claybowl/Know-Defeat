# Real-Time Polling System Development Prompt

## Project Context
You are tasked with developing a real-time polling system for the **Know Defeat** algorithmic trading platform. This system will continuously monitor bot performance metrics, update rankings, and dynamically adjust trading strategies based on evolving market conditions.

## Technical Requirements
Design a modular, Python-based polling system that meets the following criteria:
- Integrates seamlessly with the existing database structure (PostgreSQL with TimescaleDB extension)
- Aligns with the current bot management architecture
- Utilizes `async/await` patterns for non-blocking operations
- Supports configurable polling intervals
- Includes robust logging and error handling mechanisms
- Can be deployed with minimal changes to the existing codebase

## Core Functionality
The system must:
- Poll the performance of all active trading bots at user-defined intervals
- Calculate and update key performance metrics such as win rate, profit factor, and more
- Trigger periodic re-ranking of algorithms based on current performance data
- Dynamically enable or disable bots based on their performance rankings
- Monitor trade execution effectiveness and latency
- Track changes in market conditions and adjust strategy parameters accordingly
- Store historical polling data for in-depth performance analysis
- Provide alerting mechanisms for detecting anomalous bot behavior

## Architecture Considerations
When designing the system, ensure the following:
- Minimize excessive database load through efficient query design
- Implement caching mechanisms for frequently accessed data
- Ensure thread safety for concurrent operations
- Include graceful shutdown and recovery mechanisms
- Design for horizontal scaling to accommodate an increasing number of bots

## Database Integration
The polling system should interact with the following existing tables:
- `bot_metrics`: For storing and updating performance metrics
- `bot_rankings`: For tracking and updating bot rankings
- `sim_bot_trades`: For analyzing trade execution details
- `tick_data`: For correlating bot performance with market conditions

## Implementation Deliverables
Develop a Python module that includes:
- A `PerformancePoller` class to manage the core polling logic
- Database connection and query handling functionality
- Functions for calculating performance metrics
- Procedures for updating bot rankings
- Configuration and settings management
- Comprehensive logging and monitoring capabilities
- Detailed documentation of the system design and API

The implementation must adhere to the existing project structure and coding conventions, while prioritizing maintainability and extensibility for future enhancements.