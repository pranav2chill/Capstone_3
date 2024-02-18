import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ChampionTeamsEDA:
    def __init__(self, data, champions_teams):
        """
        Initialize the ChampionTeamsEDA class.

        Parameters:
        - data (pandas.DataFrame): The data containing the match information.
        - champions_teams (list): The list of champion teams to analyze.
        """
        self.data = data
        self.champions_teams = champions_teams
        self.champion_data = self.data[self.data['team'].isin(champions_teams)].copy()
        self.clean_numeric_columns()

    def clean_numeric_columns(self):
        """
        Clean the numeric columns in the champion data.
        """
        numeric_columns = ['kill', 'death', 'acs']
        for column in numeric_columns:
            self.champion_data[column] = pd.to_numeric(self.champion_data[column].astype(str).replace('\xa0', ''), errors='coerce')

    def calculate_win_numeric(self):
        """
        Calculate the win numeric column based on the win_lose column.
        """
        self.champion_data['win_numeric'] = self.champion_data['win_lose'].apply(lambda x: 1 if x == 'team win' else 0)

    def map_preferences(self):
        """
        Visualize the map preferences for champion teams.
        """
        map_counts = self.champion_data['map'].value_counts()
        
        # Visualization
        map_counts.plot(kind='bar', figsize=(10, 6), color='teal')
        plt.title('Map Preferences for Champion Teams')
        plt.ylabel('Number of Matches')
        plt.xlabel('Map')
        plt.xticks(rotation=45)
        plt.show()

    def most_picked_agents(self, top_n=10):
        """
        Visualize the most picked agents by champion teams.

        Parameters:
        - top_n (int): The number of top agents to display. Default is 10.
        """
        agent_counts = self.champion_data['agent'].value_counts().head(top_n)
        # Visualization
        agent_counts.plot(kind='bar', figsize=(10, 6), color='purple')
        plt.title('Most Picked Agents by Champion Teams')
        plt.ylabel('Number of Picks')
        plt.xlabel('Agent')
        plt.xticks(rotation=45)
        plt.show()
    
    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ChampionTeamsEDA:
    def __init__(self, data, champions_teams):
        """
        Initialize the ChampionTeamsEDA class.

        Parameters:
        - data (pandas.DataFrame): The data containing the match information.
        - champions_teams (list): The list of champion teams to analyze.
        """
        self.data = data
        self.champions_teams = champions_teams
        self.champion_data = self.data[self.data['team'].isin(champions_teams)].copy()
        self.clean_numeric_columns()

    def clean_numeric_columns(self):
        """
        Clean the numeric columns in the champion data.
        """
        numeric_columns = ['kill', 'death', 'acs']
        for column in numeric_columns:
            self.champion_data[column] = pd.to_numeric(self.champion_data[column].astype(str).replace('\xa0', ''), errors='coerce')

    def calculate_win_numeric(self):
        """
        Calculate the win numeric column based on the win_lose column.
        """
        self.champion_data['win_numeric'] = self.champion_data['win_lose'].apply(lambda x: 1 if x == 'team win' else 0)

    def map_preferences(self):
        """
        Visualize the map preferences for champion teams.
        """
        map_counts = self.champion_data['map'].value_counts()
        
        # Visualization
        map_counts.plot(kind='bar', figsize=(10, 6), color='teal')
        plt.title('Map Preferences for Champion Teams')
        plt.ylabel('Number of Matches')
        plt.xlabel('Map')
        plt.xticks(rotation=45)
        plt.show()

    def most_picked_agents(self, top_n=10):
        """
        Visualize the most picked agents by champion teams.

        Parameters:
        - top_n (int): The number of top agents to display. Default is 10.
        """
        agent_counts = self.champion_data['agent'].value_counts().head(top_n)
        # Visualization
        agent_counts.plot(kind='bar', figsize=(10, 6), color='purple')
        plt.title('Most Picked Agents by Champion Teams')
        plt.ylabel('Number of Picks')
        plt.xlabel('Agent')
        plt.xticks(rotation=45)
        plt.show()
    
    def most_picked_agents_per_map(self, top_n=10):
            """
            Visualize the most picked agents by champion teams on each map.

            Parameters:
            - top_n (int): The number of top agents to display. Default is 10.
            """
            for map_name in self.champion_data['map'].unique():
                print(f"Map: {map_name}")
                map_data = self.champion_data[self.champion_data['map'] == map_name]
                agent_counts = map_data['agent'].value_counts().head(top_n)
                
                # Visualization
                agent_counts.plot(kind='bar', figsize=(10, 6))
                plt.title(f'Most Picked Agents by Champion Teams on {map_name}')
                plt.ylabel('Number of Picks')
                plt.xlabel('Agent')
                plt.xticks(rotation=45)
                plt.show()

    def win_percentage_per_team(self):
        """
        Visualize the win percentage for champion teams.
        """
        self.calculate_win_numeric()
        win_percentage = self.champion_data.groupby('team').mean(numeric_only=True)['win_numeric']
        # Visualization
        win_percentage.plot(kind='bar', figsize=(10, 6), color='green')
        plt.title('Win Percentage for Champion Teams')
        plt.ylabel('Win Percentage')
        plt.xlabel('Team')
        plt.xticks(rotation=45)
        plt.show()
