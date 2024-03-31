from flask import Flask, jsonify

app = Flask(__name__)

# Dummy machine learning model function to generate movie recommendations
def recommend_movies(user_id):
    recommended_movies = [
        {"Jumanji": "Movie 1", "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.imdb.com%2Ftitle%2Ftt7975244%2F&psig=AOvVaw1kVuTUTF9Xh37GP09svBS_&ust=1708053864004000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCKjMw9CyrIQDFQAAAAAdAAAAABAE": "movie1.jpg"},
        {"title": "Movie 2", "image_url": "movie2.png"},
        {"title": "Movie 3", "image_url": "movie3.png"},
    ]
    return recommended_movies

@app.route('/recommendations/<int:user_id>')
def get_recommendations(user_id):
    recommended_movies = recommend_movies(user_id)
    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
