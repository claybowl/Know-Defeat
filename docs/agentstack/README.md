# AgentStack Documentation

## Introduction

AgentStack is a valuable developer tool for quickly scaffolding agent projects. Think of it as "create-next-app" for AI Agents.

### Features of AgentStack

* Instant project setup with `agentstack init`
* Useful CLI commands for generating new agents and tasks in the development cycle
* A myriad of pre-built tools for Agents

## What is the Agent Stack?

The agent stack is the list of tools that are collectively the agent stack. This is similar to the tech stack of a web app. Whether a project is built with AgentStack or not, the concept of the agent stack remains the same.

## What is AgentStack?

AgentStack is called AgentStack because it's the easiest way to quickly scaffold your agent stack! With a couple CLI commands, you can create a near-production ready agent!

## Installation

### Using the Installer (Recommended)

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://install.agentstack.sh | sh
```

### Installing with Brew

```bash
brew tap agentops-ai/tap
brew install agentstack
```

### Installing with pipx

```bash
pipx install agentstack
```

### Installing with UV

1. Install UV with their bash install script

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment

```bash
uv venv
```

3. Install AgentStack

```bash
uv pip install agentstack
```

### Verification

Run `agentstack --version` to verify that the CLI is installed and accessible.

## Quickstart

### Initialize a New Project

```bash
agentstack init <project_name>
```

#### With the Wizard

```bash
agentstack init <project_name> --wizard
```

#### With a Template

```bash
agentstack init --template=<template_name/url>
```

### Building Your Project

AgentStack 0.3 and beyond is framework-agnostic! Choose any supported framework and start building.

#### Agents

To generate a new agent:

```bash
agentstack generate agent <agent_name>
```

#### Tasks

To generate a new task:

```bash
agentstack generate task <task_name>
```

## CLI Reference

It all starts with calling:

```bash
agentstack
```

### Shortcut Aliases

Many top-level AgentStack commands can be invoked using a single-letter prefix to save keystrokes.

### Global Flags

These flags work with all commands:

- `--debug` - Print a full traceback when an error is encountered.
- `--path=<path>` - Set the working directory of the current AgentStack project.
- `--version` - Prints the current version and exits.

### agentstack init

This initializes a new AgentStack project.

```bash
agentstack init <slug_name>
```

#### Init Creates a Virtual Environment

AgentStack creates a new directory, initializes a new virtual environment, installs dependencies, and populates the project structure.

#### Initializing with the Wizard

```bash
agentstack init --wizard
```

#### Initializing from a Template

```bash
agentstack init --template=<template_name>
```

A template_name can be one of three identifiers:
- A built-in AgentStack template
- A template file from the internet (full HTTPS URL)
- A local template file (absolute or relative path)

### agentstack run

This runs your AgentStack project.

```bash
agentstack run
```

#### Overriding Inputs

```bash
agentstack run --input-topic=Sports
```

#### Running other project commands

```bash
agentstack run --function=<function_name>
```

### Generate Commands

#### agentstack generate agent

Generate a new agent:

```bash
agentstack generate agent <agent_name>
```

Options:
- `--role` (optional) - Prompt parameter: The role of the agent
- `--goal` (optional) - Prompt parameter: The goal of the agent
- `--backstory` (optional) - Prompt parameter: The backstory of the agent
- `--llm` (optional) - Which model to use for this agent

##### Example

```bash
agentstack generate agent script_writer
```

#### agentstack generate task

Generate a new task:

```bash
agentstack generate task <task_name>
```

Options:
- `--description` (optional) - Prompt parameter: Explain the task in detail
- `--expected_output` (optional) - What is the expected output from the agent
- `--agent` (optional) - The name of the agent to assign the task to

##### Example

```bash
agentstack g t gen_script --description "Write a short film script about secret agents"
```

### Tools Commands

#### agentstack tools list

Lists all tools available in AgentStack:

```bash
agentstack tools list
```

#### agentstack tools add

Shows an interactive interface for selecting which Tool to add and which Agents to add it to:

```bash
agentstack tools add <tool_name>
```

Add a Tool to a single Agent:

```bash
agentstack tools add <tool_name> --agent=<agent_name>
```

Add a Tool to multiple Agents:

```bash
agentstack tools add <tool_name> --agents=<agent_name>,<agent_name>,<agent_name>
```

#### agentstack tools remove

Removes a tool from all Agents in the project:

```bash
agentstack tools remove <tool_name>
```

### Templates

Projects can be exported into a template to facilitate sharing configurations.

#### agentstack export

```bash
agentstack export <filename>
```

### Other Commands

#### agentstack update

Check for updates and allow the user to install the latest release of AgentStack.

#### agentstack login

Authenticate with agentstack.sh for hosted integrations.

## Resources

- [AgentStack Documentation](https://docs.agentstack.sh/)
- [AgentStack GitHub](https://github.com/agentops-ai/agentstack) 