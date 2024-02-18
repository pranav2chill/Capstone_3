
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GeneralEDA:
    def __init__(self, data):
        """
        Initialize the GeneralEDA class.

        Parameters:
        - data (pandas.DataFrame): The input data for exploratory data analysis.
        """
        self.data = data

    def most_picked_maps(self):
        """
        Plot a bar chart showing the most picked maps.
        """
        most_picked_maps = self.data['map'].value_counts()
        plt.figure(figsize=(10, 6))
        most_picked_maps.plot(kind='bar', color='skyblue')
        plt.title('Most Picked Maps')
        plt.xlabel('Map')
        plt.ylabel('Number of Picks')
        plt.xticks(rotation=45)
        plt.show()

    def most_picked_agents(self):
        """
        Plot a bar chart showing the most picked agents.
        """
        most_picked_agents = self.data['agent'].value_counts()
        plt.figure(figsize=(12, 8))
        most_picked_agents.plot(kind='bar', color='lightgreen')
        plt.title('Most Picked Agents')
        plt.xlabel('Agent')
        plt.ylabel('Number of Picks')
        plt.xticks(rotation=45)
        plt.show()

    def most_picked_agents_per_map(self, top_n=8):
        """
        Plot bar charts showing the most picked agents for each map.

        Parameters:
        - top_n (int): The number of top agents to display for each map (default: 8)
        """
        maps = self.data['map'].unique()
        for map_name in maps:
            plt.figure(figsize=(10, 6))
            map_data = self.data[self.data['map'] == map_name]['agent'].value_counts().head(top_n)
            sns.barplot(x=map_data.values, y=map_data.index, palette='coolwarm')
            plt.title(f'Most Picked Agents on {map_name} (Top {top_n})')
            plt.xlabel('Number of Picks')
            plt.ylabel('Agent')
            plt.show()

    def boxplots(self):
        """
        Plot boxplots for the 'kill', 'death', and 'acs' columns.
        """
        for col in ['kill', 'death', 'acs']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        sns.boxplot(ax=axes[0], x=self.data['kill'])
        axes[0].set_title('Boxplot of Kills')
        sns.boxplot(ax=axes[1], x=self.data['death'])
        axes[1].set_title('Boxplot of Deaths')
        sns.boxplot(ax=axes[2], x=self.data['acs'])
        axes[2].set_title('Boxplot of ACS')
        plt.tight_layout()
        plt.show()

    def correlation_matrix(self):
        """
        Plot a correlation matrix for performance metrics.
        """
        cols = ['kill', 'death', 'assist', 'acs', 'kast%', 'adr', 'hs%', 'fk', 'fd']
        for col in cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        performance_metrics = self.data[['kill', 'death', 'assist', 'acs', 'kast%', 'adr', 'hs%', 'fk', 'fd']]
        corr_matrix = performance_metrics.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title('Correlation Matrix of Player Performance Metrics')
        plt.show()


    def win_percentage_per_team(self, champions_teams=None):
        """
        Plot a bar chart showing the win percentage per team.

        Parameters:
        - champions_teams (list): List of teams to highlight as championship teams (default: None)
        """
        team_wins = self.data[self.data['win_lose'] == 'team win'].groupby('team').size()
        team_games = self.data.groupby('team').size()
        team_win_percentage = (team_wins / team_games).sort_values(ascending=False) * 100

        colors = ['darkblue' if team in champions_teams else 'coral' for team in team_win_percentage.index] if champions_teams else 'coral'
        
        plt.figure(figsize=(14, 10))
        team_win_percentage.plot(kind='bar', color=colors)
        plt.title('Win Percentage per Team' + (' (Championship Teams Highlighted)' if champions_teams else ''))
        plt.xlabel('Team')
        plt.ylabel('Win Percentage (%)')

        if champions_teams:
            tick_labels = plt.gca().get_xticklabels()
            for i, label in enumerate(tick_labels):
                if label.get_text() in champions_teams:
                    tick_labels[i].set_fontweight('bold')
                    tick_labels[i].set_color('darkblue')

        plt.xticks(rotation=45)
        plt.show()