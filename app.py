from flask import Flask
from markupsafe import escape
from flask import request
import helper as h
import json

app = Flask(__name__)


class SearchEntry():
    def __init__(self, id, title, score, explanation = None):
        self.id = id
        self.title = title
        self.score = score
        self.explanation = explanation


@app.route("/", methods=["GET"])
def hello_world():
    search = ""
    if request is not None:
        search = request.args.get("q") or ""

    use_knn = request.args.get("knn") or False
    explain = request.args.get("explain") or False

    def checked(value):
        return "checked" if value else ""

    output = f"""
    <h1>Search API</h1>
    <p>Search API is a simple API that allows you to search for a string in a list of strings.</p>
    <form action="/">
        <label for="search">Search:</label><br>
        <input type="text" id="q" name="q" value="{escape(search)}">
        <label><input type="checkbox" name="knn" value="knn" {checked(use_knn)}> Use KNN</label>
        <label><input type="checkbox" name="explain" value="explain" {checked(explain)}> Explain</label>
        <input type="submit" value="Submit">
    </form>
    """

    if search is not None and search != "":
        search_results = get_search_results(search, use_knn, explain)
        def explanation_html(explanation):
            if explanation is None:
                return ""
            return f"<pre>{json.dumps(explanation, indent=2)}</pre>"
        output += f"""
        <h2>Search Results</h2>
        <ol>
            {"".join([f"<li>{result.title} <em>({result.score})</em>{explanation_html(result.explanation)}</li>"for result in search_results])}
        </ol>
        """
        if len(search_results) == 0:
            output += "<p>No results found.</p>"

    return output


def get_search_results(search: str, use_knn: bool, explain: bool):
    tokenizer, model = h.get_tokenizer_model()
    query_dict = tokenizer(["query: " + search], padding=True, truncation=True, return_tensors="pt")
    query_outputs = model(**query_dict)
    query_embedding = h.average_pool(query_outputs.last_hidden_state, query_dict["attention_mask"])
    # query_embedding = F.normalize(query_embedding, p=2, dim=1)

    query_body = {
        "query": {
            "multi_match": {
                "query": search,
                "fields": ["title^2", "text"],
            }
        }
    }
    if use_knn:
        query_body["knn"] = {
            "query_vector": query_embedding[0].tolist(),
            "field": "chunks.embedding",
            "k": 5,
            "num_candidates": 10,
        }

    if explain:
        query_body["explain"] = True

    es = h.get_elasticsearch()
    response = es.search(index="articles", body=query_body)
    hits = response.body.get("hits").get("hits")
    # get titles from hits
    return [SearchEntry(hit["_id"], hit["_source"]["title"], hit["_score"], hit["_explanation"] if explain else None) for hit in hits]
