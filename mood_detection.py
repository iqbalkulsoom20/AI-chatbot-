from transformers import pipeline

def detect_mood(text):
    # Load a pre-trained sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Analyze the user's input
    result = sentiment_pipeline(text)[0]
    
    # Return the detected mood (POSITIVE or NEGATIVE)
    return result['label']

# Example usage
if __name__ == "__main__":
    user_input = "I'm feeling stressed today."
    mood = detect_mood(user_input)  # This calls the function where sentiment_pipeline is defined
    print(f"Detected Mood: {mood}")