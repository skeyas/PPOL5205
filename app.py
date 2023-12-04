import os

from flask import Flask, render_template, request

app = Flask(__name__, static_folder="static")


@app.route("/", methods=['GET', 'POST'])
def root():
    cluster_method = [('kmeans', 'K-Means'), ('dbscan', 'DBSCAN')]
    col_source = [("article_without_keywords", "Without keywords"),
                  ("extreme_tokens", "Extreme"),
                  ("spacy_tokenized_article", "Spacy entities"),
                  ("word_tokenized_article", "Standard")]

    selected_cluster = request.form.get('clustering_method')
    default_cluster = 'kmeans'
    if selected_cluster is not None:
        for t in cluster_method:
            if t[0] == str(selected_cluster):
                default_cluster = t[1]
    else:
        selected_cluster = default_cluster

    selected_source = request.form.get('col_source')
    default_source = 'article_without_keywords'
    if selected_source is not None:
        for t in col_source:
            if t[0] == str(selected_source):
                default_source = t[1]
    else:
        selected_source = default_source

    cluster_wordclouds = [f for f in os.listdir(
        f"static/word_cloud_by_cluster/{str(selected_cluster)}/{str(selected_source)}")]

    return render_template("index.html", msg=str(selected_cluster),
                           selected_cluster_value=str(selected_cluster),
                           cluster_method=cluster_method,
                           default_cluster=default_cluster,
                           selected_source_value=str(selected_source),
                           col_source=col_source,
                           wordclouds=cluster_wordclouds
                           )
