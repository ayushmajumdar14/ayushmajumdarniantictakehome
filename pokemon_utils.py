import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_prepare_data():
    """Load and prepare Pokemon data for analysis."""
    # Load datasets
    data_dir = 'pokemon-env'
    pokemon_df = pd.read_csv(os.path.join(data_dir, 'pokemon_data_science.csv'))
    
    # Clean up column names
    if 'Name' in pokemon_df.columns:
        pokemon_df = pokemon_df.rename(columns={'Name': 'name'})
    
    # Create feature matrix with available columns
    base_features = ['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed']
    
    # Calculate total stats if not present
    if 'Total' not in pokemon_df.columns:
        pokemon_df['Total'] = pokemon_df[base_features].sum(axis=1)
    
    # Add additional features if they exist
    additional_features = []
    if 'Generation' in pokemon_df.columns:
        additional_features.append('Generation')
    if 'isLegendary' in pokemon_df.columns:
        additional_features.append('isLegendary')
    
    features = base_features + ['Total'] + additional_features
    
    # Ensure all required columns exist
    for feature in features:
        if feature not in pokemon_df.columns:
            raise ValueError(f"Required column '{feature}' not found in the dataset")
    
    X = pokemon_df[features].values
    
    # Check if Mega_Evolution column exists
    if 'Mega_Evolution' not in pokemon_df.columns:
        if 'hasMegaEvolution' in pokemon_df.columns:
            y = pokemon_df['hasMegaEvolution'].values
        else:
            raise ValueError("Neither 'Mega_Evolution' nor 'hasMegaEvolution' column found in the dataset")
    else:
        y = pokemon_df['Mega_Evolution'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Store pokemon_df as a global variable for other functions to use
    global _pokemon_df
    _pokemon_df = pokemon_df
    
    return X_train, X_test, y_train, y_test, features

def get_pokemon_df():
    """Get the Pokemon DataFrame."""
    global _pokemon_df
    if '_pokemon_df' not in globals():
        raise RuntimeError("Pokemon DataFrame not initialized. Call load_and_prepare_data() first.")
    return _pokemon_df

def calculate_type_scores(pokemon_df, type_chart_df, weights):
    """Calculate type scores based on given weights."""
    # Get effectiveness columns - look for both 'effective' and 'against' patterns
    effectiveness_cols = [col for col in type_chart_df.columns if 'effective' in col.lower() or 'against_' in col.lower()]
    
    if not effectiveness_cols:
        print("Warning: No effectiveness columns found. Available columns:", type_chart_df.columns.tolist())
        effectiveness_cols = [col for col in pokemon_df.columns if 'against_' in col.lower()]
        if effectiveness_cols:
            print("Found effectiveness columns in pokemon_df instead:", effectiveness_cols)
            type_chart_df = pokemon_df  # Use pokemon_df for effectiveness data if it contains the columns
    
    print(f"Using effectiveness columns: {effectiveness_cols}")
    
    # Initialize results DataFrame
    type_scores = pd.DataFrame()
    
    # Calculate offensive and defensive power using improved metrics
    pokemon_df['Offensive_Power'] = pokemon_df[['Attack', 'Sp_Atk']].max(axis=1)  # best offensive stat
    pokemon_df['Defensive_Power'] = pokemon_df[['Defense', 'Sp_Def']].mean(axis=1)  # average of defenses
    
    # Calculate type metrics using 75th percentile
    type_scores['Offensive_Power'] = pokemon_df.groupby('Type_1')['Offensive_Power'].quantile(0.75)
    type_scores['Defensive_Power'] = pokemon_df.groupby('Type_1')['Defensive_Power'].quantile(0.75)
    type_scores['Speed'] = pokemon_df.groupby('Type_1')['Speed'].quantile(0.75)
    type_scores['Pokemon_Count'] = pokemon_df.groupby('Type_1')['name'].count()
    
    # Calculate type effectiveness scores
    for type_name in pokemon_df['Type_1'].unique():
        type_data = type_chart_df[type_chart_df['Type_1'] == type_name]
        
        if not type_data.empty and effectiveness_cols:
            # Print debug info
            print(f"\nProcessing type: {type_name}")
            print(f"Type data shape: {type_data.shape}")
            print(f"Effectiveness values for {type_name}:")
            print(type_data[effectiveness_cols].iloc[0])
            
            # Calculate mean effectiveness
            effectiveness_values = type_data[effectiveness_cols].values.flatten()  # Flatten the array
            mean_score = np.nanmean(effectiveness_values)
            
            # Calculate vulnerability score (how many times weak to other types)
            vulnerability_score = np.nanmean(effectiveness_values > 1)
            
            # Calculate resistance score (how many times resistant to other types)
            resistance_score = np.nanmean(effectiveness_values < 1)
            
            print(f"Calculated scores for {type_name}:")
            print(f"  Mean effectiveness: {mean_score}")
            print(f"  Vulnerability: {vulnerability_score}")
            print(f"  Resistance: {resistance_score}")
            
            # Store scores
            type_scores.loc[type_name, 'Type_Effectiveness'] = float(mean_score)
            type_scores.loc[type_name, 'Vulnerability_Score'] = float(vulnerability_score)
            type_scores.loc[type_name, 'Resistance_Score'] = float(resistance_score)
        else:
            print(f"Warning: No effectiveness data found for type {type_name}")
            # Use neutral values if no data is found
            type_scores.loc[type_name, 'Type_Effectiveness'] = 1.0
            type_scores.loc[type_name, 'Vulnerability_Score'] = 0.0
            type_scores.loc[type_name, 'Resistance_Score'] = 0.0
    
    # Calculate total score using weights
    type_scores['Total_Score'] = (
        weights['offensive'] * type_scores['Offensive_Power'] +
        weights['defensive'] * type_scores['Defensive_Power'] +
        weights['speed'] * type_scores['Speed'] +
        weights['effectiveness'] * type_scores['Type_Effectiveness'] +
        weights['pokemon_count'] * type_scores['Pokemon_Count'] +
        weights['vulnerability'] * type_scores['Vulnerability_Score'] +
        weights['resistance'] * type_scores['Resistance_Score']
    )
    
    return type_scores

def print_model_evaluation(model_name, cv_scores, classification_report, confusion_matrix):
    """Print model evaluation metrics in a formatted way."""
    import streamlit as st
    
    st.subheader(f"{model_name} Model Evaluation")
    
    # Cross-validation scores
    st.write("Cross-validation Scores:")
    cv_df = pd.DataFrame({
        'Fold': range(1, len(cv_scores) + 1),
        'Score': cv_scores
    })
    st.dataframe(cv_df)
    
    # Classification report
    st.write("Classification Report:")
    st.code(classification_report)
    
    # Confusion matrix
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

def plot_feature_importance(importance, features, model_name):
    """Plot feature importance for a given model."""
    import streamlit as st
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['Feature'],
            y=importance_df['Importance'],
            text=importance_df['Importance'].round(3),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"{model_name} Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        showlegend=False
    )
    
    st.plotly_chart(fig)

def normalize_scores(df):
    """Normalize scores to 0-1 range for each metric."""
    scaler = MinMaxScaler()
    cols_to_normalize = df.columns
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df),
        columns=cols_to_normalize,
        index=df.index
    )
    return df_normalized 