from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

documents = [
    "Cat chases boy",
    "Boy chases cat",
    "Cat eats fish",
    "Boy eats candy",
]
# "Cat and caT are friends"

vectorizer = CountVectorizer(lowercase=True)
X = vectorizer.fit_transform(documents)

print("Vocabulary:")
print(vectorizer.vocabulary_)

print("\nBoW Matrix:")
print(X.toarray())

similarity_matrix = cosine_similarity(X)

print("Matriz de similitud coseno:")
print(similarity_matrix)

distance_matrix = euclidean_distances(X)

print("Matriz de distancia euclidiana:")
print(distance_matrix)
