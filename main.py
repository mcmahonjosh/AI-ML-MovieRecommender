import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv('imdb_top_1000.csv')
top500_movies_full = movies.sort_values(by='IMDB_Rating', ascending=False).head(500)
top500_movies = top500_movies_full[['Series_Title', 'Overview']].copy()

top500_movies['Similarity_To_Preference'] = pd.NA

vectorizer = TfidfVectorizer(stop_words='english') #increase cosine similarity by removing unimportant words ("I", "my", "is", etc)
vectorizer.fit(top500_movies['Overview'])

preference = input("Enter your preferences in a movie: ")

while True:
    try:
        NumberMovies = int(input("And how many movies would you like me to recommend? "))
        break
    except ValueError:
        print("Please enter a valid number.")
        
preference_vector = vectorizer.transform([preference])

for index, row in top500_movies.iterrows():
    overview = row['Overview']
    overview_vector = vectorizer.transform([overview])
    cosine_sim = cosine_similarity(preference_vector, overview_vector)
    top500_movies.at[index, 'Similarity_To_Preference'] = cosine_sim[0][0]

df = pd.DataFrame(top500_movies)
sorted_by_cosine = df.sort_values(by='Similarity_To_Preference', ascending=False)
top_N = sorted_by_cosine.head(NumberMovies)


print('The top ' + str(NumberMovies) + ' movies I would recommend based on your preferences are: ', '\n')
print(top_N[['Series_Title', 'Similarity_To_Preference']].rename(columns={'Similarity_To_Preference' : 'Similarity Score'}).to_string(index=False))




#print(top500_movies_copy)


#top500_movies.to_csv('top_500_movies.csv', index=False)


#sentence1 = "I love programming in Python."
#sentence2 = "I do my homework everyday"
#sentence3 = "Python is my favorite programming language."


#tfidf_matrix = vectorizer.fit_transform([sentence1, sentence3])

#cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

#print("Cosine Similarity:", cosine_sim[0][0])