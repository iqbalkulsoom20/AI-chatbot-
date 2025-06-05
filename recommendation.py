import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(user_preferences, meal_data):
    # Combine meal attributes into a single text column
    meal_data['combined'] = meal_data['Cuisine'] + " " + meal_data['Dietary Tags'] + " " + meal_data['Mood Association']

    # Vectorize the combined text
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(meal_data['combined'])

    # Vectorize user preferences
    user_vector = tfidf.transform([user_preferences])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    meal_data['similarity_score'] = similarity_scores[0]

    # Sort by similarity score
    recommendations = meal_data.sort_values(by='similarity_score', ascending=False)
    return recommendations.head(5)  # Top 5 recommendations

def collaborative_filtering(user_id, user_data, meal_data):
    try:
        # Convert Past Orders from string to list of integers
        user_data['Past Orders'] = user_data['Past Orders'].apply(
            lambda x: [int(i) for i in str(x).split(',')] if pd.notna(x) else []
        )
        
        # Find the current user's orders
        current_user_orders = user_data.loc[user_data['User ID'] == user_id, 'Past Orders'].iloc[0]
        
        # Find similar users (who ordered at least one same item)
        similar_users = user_data[
            user_data['Past Orders'].apply(
                lambda x: any(item in x for item in current_user_orders)
            )
        ]
        
        # Get recommendations from similar users
        recommended_meals = []
        for orders in similar_users['Past Orders']:
            recommended_meals.extend(orders)
        
        # Remove duplicates and meals the user already ordered
        recommended_meals = list(set(recommended_meals) - set(current_user_orders))
        
        return recommended_meals[:5]  # Return top 5 recommendations
    
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return []  # Return empty list if error occurs
    
def hybrid_recommendation(user_id, user_preferences, user_data, meal_data):
    # Get content-based recommendations
    content_based = content_based_recommendation(user_preferences, meal_data)
    
    # Get collaborative filtering recommendations
    collaborative = collaborative_filtering(user_id, user_data, meal_data)
    
    # Combine and rank recommendations
    hybrid_recommendations = list(set(content_based['Meal ID'].tolist() + collaborative))
    return hybrid_recommendations[:5]  # Top 5 recommendations

# Example usage
if __name__ == "__main__":
    # Load data
    meal_data = pd.read_excel("databases/Meal Data.xlsx")
    user_data = pd.read_excel("databases/User Data.xlsx")

    # Example: Get recommendations for User ID 101
    user_id = 101
    user_preferences = "Vegetarian Italian Comfort"
    recommendations = hybrid_recommendation(user_id, user_preferences, user_data, meal_data)
    print(f"Recommended Meals: {recommendations}")
    