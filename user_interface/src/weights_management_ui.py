import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import asyncio
import asyncpg
from datetime import datetime
import numpy as np

class WeightsManagementUI:
    """
    Streamlit module for managing variable weights and viewing bot rankings.
    This class provides methods to render UI components for the ranking system.
    """
    
    def __init__(self, db_config):
        """Initialize with database configuration."""
        self.db_config = db_config
    
    async def _get_db_pool(self):
        """Create and return a database connection pool."""
        return await asyncpg.create_pool(**self.db_config)
    
    async def fetch_variable_weights(self):
        """Fetch variable weights from the database."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                # Check if table exists
                table_exists = await connection.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'variable_weights'
                    );
                """)
                
                if not table_exists:
                    return None
                
                # Fetch weights
                weights = await connection.fetch("""
                    SELECT weight_id, variable_name, weight, description, last_updated
                    FROM variable_weights
                    ORDER BY variable_name;
                """)
                
                return weights
        except Exception as e:
            st.error(f"Error accessing variable weights: {e}")
            return None
        finally:
            await pool.close()
    
    async def update_variable_weight(self, variable_name, weight):
        """Update a variable weight in the database."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                await connection.execute("""
                    UPDATE variable_weights
                    SET weight = $2, last_updated = NOW()
                    WHERE variable_name = $1;
                """, variable_name, weight)
            return True
        except Exception as e:
            st.error(f"Error updating weight: {e}")
            return False
        finally:
            await pool.close()
    
    async def update_multiple_weights(self, variable_names, weights):
        """Update multiple variable weights at once."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                # Use a transaction to update all weights atomically
                async with connection.transaction():
                    for var_name, weight in zip(variable_names, weights):
                        await connection.execute("""
                            UPDATE variable_weights
                            SET weight = $2, last_updated = NOW()
                            WHERE variable_name = $1;
                        """, var_name, weight)
            return True
        except Exception as e:
            st.error(f"Error updating multiple weights: {e}")
            return False
        finally:
            await pool.close()
    
    async def initialize_default_weights(self):
        """Initialize default variable weights in the database."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                # Create table if it doesn't exist
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS variable_weights (
                        weight_id SERIAL PRIMARY KEY,
                        variable_name VARCHAR(50) NOT NULL UNIQUE,
                        weight DECIMAL(4,1) NOT NULL,
                        description TEXT,
                        last_updated TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Insert default weights
                await connection.execute("""
                    INSERT INTO variable_weights (variable_name, weight, description)
                    VALUES
                        ('one_hour_performance', 15.0, 'Performance over the last hour'),
                        ('two_hour_performance', 12.5, 'Performance over the last two hours'),
                        ('one_day_performance', 10.0, 'Performance over the last day'),
                        ('one_week_performance', 7.5, 'Performance over the last week'),
                        ('one_month_performance', 5.0, 'Performance over the last month'),
                        ('avg_win_rate', 10.0, 'Average win rate of trades'),
                        ('avg_drawdown', 8.0, 'Average drawdown (lower is better)'),
                        ('profit_per_second', 12.0, 'Average profit per second of trading'),
                        ('price_model_score', 5.0, 'Score from price prediction model'),
                        ('volume_model_score', 5.0, 'Score from volume prediction model'),
                        ('price_wall_score', 3.0, 'Score based on order book price walls'),
                        ('win_streak_2', 2.0, 'Frequency of 2-trade win streaks'),
                        ('win_streak_3', 1.5, 'Frequency of 3-trade win streaks'),
                        ('win_streak_4', 1.5, 'Frequency of 4-trade win streaks'),
                        ('win_streak_5', 1.0, 'Frequency of 5-trade win streaks'),
                        ('win_streak_6', 0.5, 'Frequency of 6-trade win streaks'),
                        ('win_streak_7', 0.5, 'Frequency of 7-trade win streaks')
                    ON CONFLICT (variable_name) DO NOTHING;
                """)
            return True
        except Exception as e:
            st.error(f"Error initializing default weights: {e}")
            return False
        finally:
            await pool.close()
    
    async def fetch_bot_rankings(self):
        """Fetch bot rankings from the database."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                # Check if table exists
                table_exists = await connection.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'bot_rankings'
                    );
                """)
                
                if not table_exists:
                    return None
                
                # Fetch rankings
                rankings = await connection.fetch("""
                    SELECT ranking_id, bot_id, rank_score, timestamp, is_active
                    FROM bot_rankings
                    ORDER BY rank_score DESC;
                """)
                return rankings
        except Exception as e:
            st.error(f"Error accessing bot rankings: {e}")
            return None
        finally:
            await pool.close()
    
    async def fetch_bots_info(self):
        """Fetch additional bot information to enhance ranking display."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                # Fetch bot metrics for active bots
                bots_info = await connection.fetch("""
                    SELECT DISTINCT ON (bot_id)
                        bot_id,
                        ticker,
                        avg_win_rate,
                        profit_per_second,
                        one_hour_performance
                    FROM bot_metrics
                    ORDER BY bot_id, timestamp DESC;
                """)
                return bots_info
        except Exception as e:
            st.error(f"Error fetching bot info: {e}")
            return None
        finally:
            await pool.close()
    
    async def update_bot_rankings(self):
        """Update all bot rankings."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                # Check if the function exists
                function_exists = await connection.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM pg_proc
                        WHERE proname = 'update_all_bot_rankings'
                    );
                """)
                
                if function_exists:
                    # Call the function
                    await connection.execute("SELECT update_all_bot_rankings();")
                else:
                    # Fallback: Use BotRanker directly
                    st.warning("Using fallback ranking method. Database function not found.")
                    from bot_ranker import BotRanker
                    ranker = BotRanker(pool)
                    await ranker.rank_bots()
                
                return True
        except Exception as e:
            st.error(f"Error updating bot rankings: {e}")
            return False
        finally:
            await pool.close()
    
    async def toggle_bot_active(self, bot_id, is_active):
        """Toggle a bot's active status."""
        pool = await self._get_db_pool()
        try:
            async with pool.acquire() as connection:
                await connection.execute("""
                    UPDATE bot_rankings
                    SET is_active = $2
                    WHERE bot_id = $1;
                """, bot_id, is_active)
                return True
        except Exception as e:
            st.error(f"Error toggling bot status: {e}")
            return False
        finally:
            await pool.close()
    
    def render_weights_management(self):
        """Render the weights management UI section."""
        st.subheader("Variable Weights Management")
        
        # Create three columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Fetch current weights
            weights = asyncio.run(self.fetch_variable_weights())
            
            if weights is not None and len(weights) > 0:
                # Convert to a more user-friendly format for editing
                weights_df = pd.DataFrame(weights)
                
                # Extract just the variable_name and weight columns
                editable_weights = pd.DataFrame({
                    'Variable': weights_df['variable_name'],
                    'Weight (%)': weights_df['weight'],
                    'Description': weights_df['description'] if 'description' in weights_df.columns else ''
                })
                
                # Create an editable dataframe
                edited_weights = st.data_editor(
                    editable_weights,
                    column_config={
                        "Variable": st.column_config.TextColumn(
                            "Variable",
                            help="Name of the metric variable",
                            disabled=True
                        ),
                        "Weight (%)": st.column_config.NumberColumn(
                            "Weight (%)",
                            help="Weight percentage for this variable (0-100)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            format="%.1f"
                        ),
                        "Description": st.column_config.TextColumn(
                            "Description",
                            help="Description of this variable",
                            disabled=True
                        )
                    },
                    hide_index=True,
                    key="weights_editor"
                )
                
                # Button to save the changes
                if st.button("Save Weight Changes", type="primary"):
                    # Check if any weights have changed
                    original_weights = editable_weights['Weight (%)'].values
                    new_weights = edited_weights['Weight (%)'].values
                    
                    if not np.array_equal(original_weights, new_weights):
                        # Update the weights in the database
                        success = asyncio.run(self.update_multiple_weights(
                            edited_weights['Variable'].tolist(),
                            edited_weights['Weight (%)'].tolist()
                        ))
                        
                        if success:
                            st.success("Weights updated successfully!")
                            # Update the bot rankings
                            asyncio.run(self.update_bot_rankings())
                        else:
                            st.error("Failed to update weights.")
                    else:
                        st.info("No changes detected in weights.")
            else:
                st.warning("No variable weights found in the database.")
                
                if st.button("Initialize Default Weights"):
                    success = asyncio.run(self.initialize_default_weights())
                    if success:
                        st.success("Default weights initialized!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to initialize weights.")
        
        with col2:
            if weights is not None and len(weights) > 0:
                weights_df = pd.DataFrame(weights)
                
                # Create a visualization of the weight distribution
                fig = px.bar(
                    weights_df, 
                    x='variable_name', 
                    y='weight',
                    labels={'variable_name': 'Variable', 'weight': 'Weight (%)'},
                    title="Weight Distribution"
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a pie chart for weight proportion
                fig2 = px.pie(
                    weights_df, 
                    names='variable_name', 
                    values='weight',
                    title="Weight Proportion"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    def render_bot_rankings(self):
        """Render the bot rankings UI section."""
        st.subheader("Bot Rankings")
        
        # Fetch current rankings
        rankings = asyncio.run(self.fetch_bot_rankings())
        
        if rankings is not None and len(rankings) > 0:
            # Convert to DataFrame
            rankings_df = pd.DataFrame(rankings)
            
            # Fetch additional bot info
            bots_info = asyncio.run(self.fetch_bots_info())
            
            if bots_info is not None and len(bots_info) > 0:
                bots_info_df = pd.DataFrame(bots_info)
                
                # Merge rankings with bot info
                merged_df = pd.merge(
                    rankings_df,
                    bots_info_df,
                    on='bot_id',
                    how='left'
                )
                
                # Format for display
                display_df = merged_df[['bot_id', 'ticker', 'rank_score', 'avg_win_rate', 
                                        'profit_per_second', 'is_active', 'timestamp']]
                display_df.columns = ['Bot ID', 'Ticker', 'Rank Score', 'Win Rate (%)', 
                                      'Profit/Second ($)', 'Active', 'Last Updated']
                
                # Format numeric columns
                display_df['Rank Score'] = display_df['Rank Score'].apply(lambda x: f"{x:.2f}")
                display_df['Win Rate (%)'] = display_df['Win Rate (%)'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                display_df['Profit/Second ($)'] = display_df['Profit/Second ($)'].apply(lambda x: f"${x:.4f}" if pd.notnull(x) else "N/A")
                
                # Add rank as the index
                display_df = display_df.sort_values('Rank Score', ascending=False)
                display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
                
                # Create columns for the table and controls
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display the rankings table
                    st.dataframe(display_df, use_container_width=True)
                
                with col2:
                    # Bot activation controls
                    st.subheader("Bot Controls")
                    
                    # Select a bot to toggle
                    bot_options = [f"Bot {row['bot_id']} ({row['ticker']})" for _, row in merged_df.iterrows()]
                    selected_bot = st.selectbox("Select Bot", bot_options)
                    
                    # Extract bot_id from selection
                    if selected_bot:
                        selected_bot_id = int(selected_bot.split(" ")[1].split(" ")[0])
                        
                        # Get current status
                        is_active = merged_df[merged_df['bot_id'] == selected_bot_id]['is_active'].iloc[0]
                        
                        # Toggle status button
                        if is_active:
                            if st.button("Deactivate Bot"):
                                success = asyncio.run(self.toggle_bot_active(selected_bot_id, False))
                                if success:
                                    st.success(f"Bot {selected_bot_id} deactivated!")
                                    st.experimental_rerun()
                        else:
                            if st.button("Activate Bot"):
                                success = asyncio.run(self.toggle_bot_active(selected_bot_id, True))
                                if success:
                                    st.success(f"Bot {selected_bot_id} activated!")
                                    st.experimental_rerun()
                
                # Create rank visualization
                st.subheader("Ranking Visualization")
                
                # Bar chart of bot rankings
                fig = px.bar(
                    merged_df.sort_values('rank_score', ascending=False),
                    x=[f"Bot {row['bot_id']} ({row['ticker']})" for _, row in merged_df.sort_values('rank_score', ascending=False).iterrows()],
                    y='rank_score',
                    color='is_active',
                    labels={'rank_score': 'Rank Score', 'is_active': 'Active Status'},
                    title="Bot Ranking Scores",
                    color_discrete_map={True: 'green', False: 'gray'}
                )
                
                fig.update_layout(
                    xaxis_title="Bot",
                    yaxis_title="Rank Score",
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Bot information not available.")
        else:
            st.warning("No bot rankings available.")
            
            if st.button("Initialize Bot Rankings"):
                success = asyncio.run(self.update_bot_rankings())
                if success:
                    st.success("Bot rankings initialized!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to initialize rankings.")
    
    def render(self):
        """Render the complete ranking system UI module."""
        st.header("Bot Ranking System")
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Variable Weights", "Bot Rankings"])
        
        with tab1:
            self.render_weights_management()
        
        with tab2:
            self.render_bot_rankings()

# Example usage in a Streamlit app:
# 
# import streamlit as st
# from weights_management_ui import WeightsManagementUI
# 
# DB_CONFIG = {
#     'user': 'clayb',
#     'password': 'musicman',
#     'database': 'tick_data',
#     'host': 'localhost'
# }
# 
# def main():
#     st.title("KnowDefeat Trading System")
#     
#     # Create tabs
#     tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Bot Control", "Rankings", "Settings"])
#     
#     with tab1:
#         st.header("Trading System Dashboard")
#         # Your existing dashboard code here...
#     
#     with tab2:
#         st.header("Bot Control")
#         # Your existing bot control code here...
#     
#     with tab3:
#         # Render the ranking system UI
#         weights_ui = WeightsManagementUI(DB_CONFIG)
#         weights_ui.render()
#     
#     with tab4:
#         st.header("System Settings")
#         # Your existing settings code here...
# 
# if __name__ == "__main__":
#     main()