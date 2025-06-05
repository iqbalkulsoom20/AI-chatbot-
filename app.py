from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from mood_detection import detect_mood
from recommendation import hybrid_recommendation

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load data
try:
    meal_data = pd.read_excel("databases/Meal Data.xlsx")
    user_data = pd.read_excel("databases/User Data.xlsx")
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    meal_data = pd.DataFrame()
    user_data = pd.DataFrame()

@app.route('/')
def home():
    return """
    <h1>AI Meal Recommendation System</h1>
    <p>Backend is running. Use the /recommend endpoint:</p>
    <ul>
        <li>Method: POST</li>
        <li>Endpoint: /recommend</li>
        <li>Required JSON: {"user_id": 1, "mood": "your mood", "preferences": ["list", "of", "prefs"]}</li>
    </ul>
    """

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not all(key in data for key in ["user_id", "mood", "preferences"]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        user_id = data["user_id"]
        mood_text = data["mood"]
        preferences = data["preferences"]
        
        print(f"Received request - User: {user_id}, Mood: {mood_text}, Prefs: {preferences}")
        
        mood = detect_mood(mood_text)
        print(f"Detected mood: {mood}")
        
        recommendations = hybrid_recommendation(
            user_id=user_id,
            user_preferences=" ".join(preferences),
            user_data=user_data,
            meal_data=meal_data
        )
        
        return jsonify({
            "status": "success",
            "mood_detected": mood,
            "recommendations": recommendations
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)