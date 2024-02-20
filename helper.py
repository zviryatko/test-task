import os
import sys
import pandas as pd
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import nltk

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_tokenizer_model():
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    # model = SentenceTransformer(model_name)
    return tokenizer, model


def get_elasticsearch():
    """
    Get the Elasticsearch client.
    :return: Elasticsearch client
    """
    return Elasticsearch(hosts=[os.getenv('ELASTICSEARCH_URL')])


class CsvArticle:
    def __init__(self, id: int, title: str, text: str):
        self.id = id
        self.title = title
        self.text = text


def get_articles(file) -> list[CsvArticle]:
    """
    Read the input file and return a list of input texts.
    :param file: input file
    :return: list of input texts
    """
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        sys.exit(1)
    articles = []
    df = pd.read_csv(file)
    for i in range(len(df)):
        articles.append(CsvArticle(i, df.loc[i, 'title'], df.loc[i, 'text']))
    return articles


class Sentence:
    def __init__(self, text: str, embedding):
        self.text = text
        self.embedding = embedding


class Article:
    def __init__(self, id: int, title: str, text: str, sentences: list[Sentence]):
        self.id = id
        self.title = title
        self.text = text
        self.sentences = sentences


def chunk_articles(articles: list[CsvArticle], tokenizer, model) -> list[Article]:
    """
    Chunk the articles into smaller pieces and return a list of ArticleChunk objects.
    :param articles: list of Article objects
    :param tokenizer: tokenizer
    :param model: model
    :return: list of ArticleChunk objects
    """
    for i, article in enumerate(articles):
        input_texts = nltk.sent_tokenize(article.text)
        input_dict = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**input_dict)
        embedding = average_pool(outputs.last_hidden_state, input_dict['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        article.sentences = []
        for text, emb in zip(input_texts, embedding):
            article.sentences.append(Sentence(text, emb))
        yield article
