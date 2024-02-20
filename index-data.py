import sys
import helper as h
import nltk
nltk.download('punkt')

file = 'articles.csv'
if len(sys.argv) > 1:
    file = sys.argv[1]
articles = h.get_articles(file)
tokenizer, model = h.get_tokenizer_model()
processed_articles = h.chunk_articles(articles, tokenizer, model)

es = h.get_elasticsearch()
index_name = 'articles'
index_body = {
    "settings": {"index": {"number_of_shards": 1}},
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "text": {"type": "text"},
            "chunks": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 768},
                },
            },
        },
    },
}
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
es.indices.create(index=index_name, body=index_body)

for article in processed_articles:
    chunks = []
    for sentence in article.sentences:
        chunks.append({"text": sentence.text, "embedding": sentence.embedding.tolist()})
    document = {"title": article.title, "text": article.text, "chunks": chunks}
    response = es.index(index=index_name, id=article.id, body=document)
    print(response)
