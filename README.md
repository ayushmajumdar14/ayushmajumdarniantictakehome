# Pokemon Analysis Project

A comprehensive Pokemon analysis project that explores various aspects of Pokemon data, including:
- Type effectiveness and strategic implications
- SQL analysis of Pokemon stats
- Machine Learning models for predicting Mega Evolution
- Statistical analysis and visualizations

## Features

- **Type Effectiveness Analysis**: Interactive analysis of Pokemon type matchups with customizable weights
- **SQL Analysis**: Query-based analysis of Pokemon stats by type
- **Machine Learning Models**: Implementation of AdaBoost, Random Forest, and Logistic Regression for Mega Evolution prediction
- **Data Visualization**: Interactive charts and heatmaps for data exploration

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ayushmajumdar14/pokemontakehomerepo.git
cd pokemontakehomerepo
```

2. Create and activate a virtual environment:
```bash
python -m venv pokemon-env
source pokemon-env/bin/activate  # On Windows: pokemon-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `pokemon_utils.py`: Utility functions for data processing
- `adaboost_model.py`: AdaBoost model implementation
- `type_effectiveness_heatmap.png`: Visualization of type effectiveness

## Technologies Used

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SQLite3
- Plotly
- Seaborn

## Analysis Sections

1. **Type Effectiveness**
   - Interactive type matchup analysis
   - Customizable scoring system
   - Detailed visualization of type relationships

2. **SQL Analysis**
   - Top Pokemon by type
   - Statistical analysis of base stats
   - Type distribution patterns

3. **Machine Learning**
   - Mega Evolution prediction
   - Feature importance analysis
   - Model comparison and evaluation

## Author

Ayush Majumdar

## Acknowledgments

Special thanks to Niantic for the opportunity to work on this Pokemon-themed data science project. As a lifelong Pokemon fan, it's been a joy to combine my passion for Pokemon with data analysis.
