import sqlite3
import pandas as pd

# create a connection to a new SQLite database
conn = sqlite3.connect('pokemon.db')

df = pd.read_csv('pokemon-env/pokemon_data_science.csv')

# Write the dataframe to SQL
df.to_sql('PokemonStats', conn, if_exists='replace', index=False)

# Example queries
def run_query(query):
    print(f"\nExecuting query: {query}")
    result = pd.read_sql_query(query, conn)
    print(result)
    return result

# get all Fire type Pokemon
query1 = """
SELECT Name, Type_1, Type_2, Total
FROM PokemonStats
WHERE Type_1 = 'Fire' OR Type_2 = 'Fire'
ORDER BY Total DESC;
"""

# get average stats by Type_1
query2 = """
SELECT Type_1,
       AVG(HP) as avg_hp,
       AVG(Attack) as avg_attack,
       AVG(Defense) as avg_defense
FROM PokemonStats
GROUP BY Type_1
ORDER BY avg_hp DESC;
"""

# find the top 5 strongest Pokemon by total stats
query3 = """
SELECT Name, Type_1, Type_2, Total
FROM PokemonStats
ORDER BY Total DESC
LIMIT 5;
"""

#query four here is for finding the top 3 Pokemon in terms of total stats for each Type_1
query4 = """
WITH RankedPokemon AS (
    SELECT 
        Name,
        Type_1, -- primary type
        Total,
        ROW_NUMBER() OVER (PARTITION BY Type_1 ORDER BY Total DESC) as rank
    FROM PokemonStats
)
SELECT 
    Type_1,
    Name,
    Total
FROM RankedPokemon
WHERE rank <= 3
ORDER BY Type_1, Total DESC;
"""

if __name__ == "__main__":
    print("Running example SQL queries...")
    #run_query(query1)
    #run_query(query2)
    #run_query(query3)
    run_query(query4)

    conn.close() 