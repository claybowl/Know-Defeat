import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Load and analyze trade data from all_trades.csv."""
    # Load the CSV file (adjust the path if necessary)
    try:
        df = pd.read_csv('../../all_trades.csv')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Display the first few rows of the dataframe
    print("First 5 rows of the trade data:")
    print(df.head())
    
    # Display dataframe information
    print("\nDataFrame Information:")
    print(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Plot a histogram of the trade PnL
    if 'trade_pnl' in df.columns:
        plt.figure(figsize=(10,6))
        sns.histplot(df['trade_pnl'], bins=20, kde=True)
        plt.title('Trade PnL Distribution')
        plt.xlabel('Trade PnL')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("Column 'trade_pnl' not found in the dataframe.")
    
    # Group by trade_direction and show counts and mean PnL
    if 'trade_direction' in df.columns:
        trade_counts = df.groupby('trade_direction').size()
        print("\nTrade counts by direction:")
        print(trade_counts)
        
        if 'trade_pnl' in df.columns:
            avg_pnl_by_direction = df.groupby('trade_direction')['trade_pnl'].mean()
            print("\nAverage Trade PnL by direction:")
            print(avg_pnl_by_direction)
    else:
        print("Column 'trade_direction' not found in the dataframe.")


if __name__ == '__main__':
    main() 