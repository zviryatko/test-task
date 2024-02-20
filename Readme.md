# Setup

1. Clone the repo.
2. `docker-compose up -d`
3. `docker compose run --rm app python3 index-data.py`- this will index the data into the elasticsearch
4. open http://0.0.0.0:5000

Use the option "Use KNN" to enable search with K-Nearest Neighbors algorithm with the text match search.
Articles are placed in `articles.csv` file.
