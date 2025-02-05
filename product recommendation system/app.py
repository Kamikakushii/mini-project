from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("products.csv")

def recommend_products(product_name, df, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['product_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()
    if product_name not in indices:
        return []
    
    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df['product_name'].iloc[product_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        product_name = request.form['product_name']
        recommendations = recommend_products(product_name, df)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
