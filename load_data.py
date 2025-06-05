import pandas as pd

# Load meal data
meal_data = pd.read_excel("databases/Meal Data.xlsx")

# Load user data
user_data = pd.read_excel("databases/User Data.xlsx")

# Display the data
print("Meal Data:")
print(meal_data.head())

print("\nUser Data:")
print(user_data.head())