# Know-Defeat
Know Defeat Repository

## Project Structure
- src/
  - collector/
  - weight_calculator/
  - database/
  - validations/
  - resolution/
  - config/
  - training/
  - monitoring/
- database_schema/


-----------------------

import os

def create_directories():
    # Main directories
    directories = [
        # Source directories
        'src/collector',
        'src/weight_calculator',
        'src/database',
        'src/validations',
        'src/resolution',
        'src/config',
        
        # Training directories
        'src/training/data',
        'src/training/models',
        'src/training/logger',
        'src/training/alert',
        'src/training/dashboard',
        'src/training/analysis',
        
        # Monitoring directories
        'src/monitoring/data',
        'src/monitoring/models',
        'src/monitoring/logger',
        'src/monitoring/alert',
        'src/monitoring/dashboard',
        'src/monitoring/analysis',
        
        # Database schema directories
        'database_schema/training_history',
        'database_schema/bot_metrics',
        'database_schema/system_metrics'
    ]
    
    # Create directories
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        # Create __init__.py in each directory
        with open(os.path.join(dir_path, '__init__.py'), 'a'):
            pass

    # Update README.md
    readme_content = """# Know-Defeat
Know Defeat Repository

## Project Structure
- src/
  - collector/
  - weight_calculator/
  - database/
  - validations/
  - resolution/
  - config/
  - training/
  - monitoring/
- database_schema/
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)

    print("Directories created successfully!")

if __name__ == "__main__":
    create_directories()