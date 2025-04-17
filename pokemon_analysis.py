import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# set display options for better readability
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# set style for visualizations
sns.set_theme()  # use seaborn's default styling
sns.set_palette("husl")

# set the path to the pokemon-env directory
data_dir = 'pokemon-env'

# i load the datasets
poke_df = pd.read_csv(os.path.join(data_dir, 'pokemon_data_science.csv'))
type_df = pd.read_csv(os.path.join(data_dir, 'pokemon.csv'))

# standardize the name column to lowercase in both dataframes
if 'Name' in poke_df.columns:
    poke_df = poke_df.rename(columns={'Name': 'name'})
if 'Name' in type_df.columns:
    type_df = type_df.rename(columns={'Name': 'name'})

# get effectiveness columns from pokemon.csv
effectiveness_cols = [col for col in type_df.columns if 'effective' in col.lower()]
type_effectiveness = type_df[['name'] + effectiveness_cols]

# merge the dataframes on Pokemon name (using lowercase for consistency)
poke_df['name_lower'] = poke_df['name'].str.lower()
type_effectiveness['name_lower'] = type_effectiveness['name'].str.lower()
merged_df = pd.merge(poke_df, type_effectiveness, on='name_lower', how='left', suffixes=('', '_y'))

# drop the temporary lowercase name column and the duplicate name column
merged_df = merged_df.drop(['name_lower', 'name_y'], axis=1, errors='ignore')

print("Dataset shape:", merged_df.shape)
print("\nFirst few rows of the merged dataset:")
print(merged_df.head())

# calculate type statistics using different metrics
# using median and 75th percentile for better representation of type strengths
type_stats = pd.DataFrame()

# for HP: use median and 75th percentile (more representative of bulk)
type_stats['HP_Median'] = merged_df.groupby('Type_1')['HP'].median()
type_stats['HP_75th'] = merged_df.groupby('Type_1')['HP'].quantile(0.75)

# for Attack stats: use 75th percentile (represents offensive potential better)
type_stats['Attack_75th'] = merged_df.groupby('Type_1')['Attack'].quantile(0.75)
type_stats['Sp_Atk_75th'] = merged_df.groupby('Type_1')['Sp_Atk'].quantile(0.75)

# for Defense stats: use median and 75th percentile
type_stats['Defense_Median'] = merged_df.groupby('Type_1')['Defense'].median()
type_stats['Sp_Def_Median'] = merged_df.groupby('Type_1')['Sp_Def'].median()
type_stats['Defense_75th'] = merged_df.groupby('Type_1')['Defense'].quantile(0.75)
type_stats['Sp_Def_75th'] = merged_df.groupby('Type_1')['Sp_Def'].quantile(0.75)

# for Speed: use both median and max (shows both typical and potential speed)
type_stats['Speed_Median'] = merged_df.groupby('Type_1')['Speed'].median()
type_stats['Speed_Max'] = merged_df.groupby('Type_1')['Speed'].max()

# calculate total stats using 75th percentiles
type_stats['Total_75th'] = merged_df.groupby('Type_1')['Total'].quantile(0.75)

print("\nType Statistics (using various metrics):")
print(type_stats.round(2))

# calculate type effectiveness scores
# considering both offensive and defensive capabilities
effectiveness_cols = [col for col in merged_df.columns if 'effective' in col.lower()]
type_effectiveness_scores = pd.DataFrame()

for type_name in merged_df['Type_1'].unique():
    type_data = merged_df[merged_df['Type_1'] == type_name][effectiveness_cols]
    
    # calculate mean effectiveness
    mean_score = type_data[effectiveness_cols].mean().mean()
    
    # calculate vulnerability score (how many times weak to other types)
    vulnerability_score = (type_data[effectiveness_cols] > 1).mean().mean()
    
    # calculate resistance score (how many times resistant to other types)
    resistance_score = (type_data[effectiveness_cols] < 1).mean().mean()
    
    # store scores
    type_effectiveness_scores.loc[type_name, 'Mean_Effectiveness'] = mean_score
    type_effectiveness_scores.loc[type_name, 'Vulnerability_Score'] = vulnerability_score
    type_effectiveness_scores.loc[type_name, 'Resistance_Score'] = resistance_score

# calculate offensive and defensive scores using improved metrics
merged_df['Offensive_Power'] = merged_df[['Attack', 'Sp_Atk']].max(axis=1)  # best offensive stat
merged_df['Defensive_Power'] = merged_df[['Defense', 'Sp_Def']].mean(axis=1)  # average of defenses

type_metrics = pd.DataFrame()
type_metrics['Offensive_Power'] = merged_df.groupby('Type_1')['Offensive_Power'].quantile(0.75)
type_metrics['Defensive_Power'] = merged_df.groupby('Type_1')['Defensive_Power'].quantile(0.75)
type_metrics['Speed'] = merged_df.groupby('Type_1')['Speed'].quantile(0.75)
type_metrics['HP'] = merged_df.groupby('Type_1')['HP'].quantile(0.75)
type_metrics['Pokemon_Count'] = merged_df.groupby('Type_1')['name'].count()

# add effectiveness scores to type_metrics
type_metrics = pd.concat([type_metrics, type_effectiveness_scores], axis=1)

print("\nType Battle Metrics (with improved scoring):")
print(type_metrics.round(3))

# create visualizations to understand type distribution and relationships
# considering both offensive and defensive capabilities
plt.figure(figsize=(12, 8))
effectiveness_means = merged_df.groupby('Type_1')[effectiveness_cols].mean()
sns.heatmap(effectiveness_means, cmap='RdYlBu_r', center=1)
plt.title('Type Effectiveness Heatmap')
plt.tight_layout()
plt.savefig('type_effectiveness_heatmap.png')
plt.close()

# create radar chart for top 5 types based on 75th percentile total stats
top_5_types = type_stats['Total_75th'].nlargest(5)

# prepare data for radar chart
stats_cols = ['HP_75th', 'Attack_75th', 'Defense_75th', 'Sp_Atk_75th', 'Sp_Def_75th', 'Speed_Median']
angles = np.linspace(0, 2*np.pi, len(stats_cols), endpoint=False)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for type_name in top_5_types.index:
    values = type_stats.loc[type_name, stats_cols].values
    values = np.concatenate((values, [values[0]]))  # complete the circle
    angles_plot = np.concatenate((angles, [angles[0]]))  # complete the circle
    ax.plot(angles_plot, values, '-', linewidth=2, label=type_name)
    ax.fill(angles_plot, values, alpha=0.25)

ax.set_xticks(angles)
ax.set_xticklabels([col.replace('_', ' ') for col in stats_cols])
ax.set_title('Top 5 Types - Advanced Stat Comparison')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig('top_5_types_radar.png')
plt.close()

print("\nAnalysis complete! Visualizations have been saved as 'type_effectiveness_heatmap.png' and 'top_5_types_radar.png'")

# calculate and display final type rankings with improved metrics
type_rankings = type_metrics.copy()
type_rankings['Total_Power'] = type_stats['Total_75th']

# normalize all metrics to 0-1 scale
for column in type_rankings.columns:
    type_rankings[column] = (type_rankings[column] - type_rankings[column].min()) / \
                           (type_rankings[column].max() - type_rankings[column].min())

# calculate final score with new weights and metrics
weights = {
    'Offensive_Power': 0.15,
    'Defensive_Power': 0.15,
    'Speed': 0.1,
    'HP': 0.1,
    'Pokemon_Count': 0.1,
    'Total_Power': 0.1,
    'Mean_Effectiveness': 0.1,
    'Vulnerability_Score': -0.1,  # negative weight because higher vulnerability is bad
    'Resistance_Score': 0.1
}

type_rankings['Final_Score'] = sum(type_rankings[metric] * weight 
                                 for metric, weight in weights.items())

# sort by final score
type_rankings = type_rankings.sort_values('Final_Score', ascending=False)

print("\nFinal Type Rankings (normalized scores):")
print(type_rankings['Final_Score'].round(3))

# print detailed metrics for each type
print("\nDetailed Type Analysis:")
detailed_metrics = type_rankings[['Offensive_Power', 'Defensive_Power', 'Speed', 
                                'HP', 'Pokemon_Count', 'Total_Power',
                                'Mean_Effectiveness', 'Vulnerability_Score',
                                'Resistance_Score', 'Final_Score']].round(3)
print(detailed_metrics)

# i create a scatter plot of type scores
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Offensive_Power', y='Defensive_Power', hue='Type_1', data=merged_df)
plt.title('Type Effectiveness Scatter Plot')
plt.xlabel('Offensive Power')
plt.ylabel('Defensive Power')
plt.tight_layout()
plt.savefig('type_effectiveness_scatter.png')
plt.close()

# analyze mega evolution patterns and their impact on type effectiveness
# considering both offensive and defensive capabilities
mega_evolution_counts = merged_df['Mega'].value_counts()
print("\nMega Evolution Counts:")
print(mega_evolution_counts)

# prepare features for machine learning models
# focusing on key stats that influence battle outcomes
# ... existing code ...

# train and evaluate models to predict battle success
# using multiple algorithms to compare performance
# ... existing code ...

# analyze feature importance to understand key factors in battles
# ... existing code ...

# save results and visualizations for further analysis
# ... existing code ... 