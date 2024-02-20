# Extract first 10 rows from csv and save it again.
import pandas as pd

df = pd.read_csv('articles.csv', nrows=10, sep=',', encoding='utf-8')
col_list = list(df.columns)
df.to_csv('medium_articles-10.csv', columns=col_list, index=False, encoding='utf-8')
