# ============================================================
# NAIVE BAYES EXAMPLE 2: Should I Play Tennis Today?
# ============================================================
# This is a CLASSIC beginner machine learning dataset!
# We use weather conditions to predict: Play Tennis? Yes or No
#
# Features (inputs):
#   - Outlook  : Sunny, Overcast, or Rain
#   - Humidity : High or Normal
#   - Wind     : Weak or Strong
#
# Label (output):
#   - Play Tennis: Yes or No
# ============================================================

# Step 1: Import tools
import pandas as pd                             # For creating a nice data table
from sklearn.naive_bayes import CategoricalNB   # Naive Bayes for category data
from sklearn.preprocessing import LabelEncoder  # Converts words to numbers

# ============================================================
# Step 2: Create the dataset manually
# This famous dataset has 14 weather days and the decision
# of whether tennis was played that day.
# ============================================================

# Each row = one day of weather data
data = {
    "Outlook":  ["Sunny",    "Sunny",    "Overcast", "Rain",  "Rain",
                 "Rain",     "Overcast", "Sunny",    "Sunny", "Rain",
                 "Sunny",    "Overcast", "Overcast", "Rain"],

    "Humidity": ["High",     "High",     "High",     "High",  "Normal",
                 "Normal",   "Normal",   "High",     "Normal","Normal",
                 "Normal",   "High",     "Normal",   "High"],

    "Wind":     ["Weak",     "Strong",   "Weak",     "Weak",  "Weak",
                 "Strong",   "Strong",   "Weak",     "Weak",  "Weak",
                 "Strong",   "Strong",   "Weak",     "Strong"],

    "PlayTennis": ["No",  "No",  "Yes", "Yes", "Yes",
                   "No",  "Yes", "No",  "Yes", "Yes",
                   "Yes", "Yes", "Yes", "No"]
}

# Put the data into a DataFrame (like a spreadsheet in Python)
df = pd.DataFrame(data)

print("=== STEP 2: Our Dataset (14 days of weather) ===")
print(df)
print()
print(f"Days we played tennis : {sum(df['PlayTennis'] == 'Yes')}")
print(f"Days we skipped tennis: {sum(df['PlayTennis'] == 'No')}")
print()

# ============================================================
# Step 3: Convert text categories to numbers
# Machine learning models need NUMBERS, not words!
#
# LabelEncoder converts like this:
#   "Sunny" -> 2,  "Overcast" -> 0,  "Rain" -> 1
#   "High"  -> 0,  "Normal"   -> 1
#   "Weak"  -> 1,  "Strong"   -> 0
#   "Yes"   -> 1,  "No"       -> 0
# ============================================================

# Create a separate encoder for each column
# (Each column has its own set of categories)
encoder_outlook  = LabelEncoder()
encoder_humidity = LabelEncoder()
encoder_wind     = LabelEncoder()
encoder_label    = LabelEncoder()

# fit_transform: learn the categories AND convert to numbers
df["Outlook_num"]   = encoder_outlook.fit_transform(df["Outlook"])
df["Humidity_num"]  = encoder_humidity.fit_transform(df["Humidity"])
df["Wind_num"]      = encoder_wind.fit_transform(df["Wind"])
df["Play_num"]      = encoder_label.fit_transform(df["PlayTennis"])

print("=== STEP 3: After converting words to numbers ===")
print(df[["Outlook", "Outlook_num", "Humidity", "Humidity_num",
          "Wind", "Wind_num", "PlayTennis", "Play_num"]])
print()

# ============================================================
# Step 4: Split data into inputs (X) and output (y)
# X = the features the model uses to make a decision
# y = the correct answer we want the model to predict
# ============================================================

X = df[["Outlook_num", "Humidity_num", "Wind_num"]]   # Input features
y = df["Play_num"]                                     # Output label

# ============================================================
# Step 5: Train the Naive Bayes model
# CategoricalNB is best for data with distinct categories
# ============================================================

model = CategoricalNB()   # Create an empty model
model.fit(X, y)           # Train it using our 14 days of data

print("=== STEP 5: Model trained on 14 days! ===")
print()

# ============================================================
# Step 6: Predict for new weather conditions
# ============================================================

# Let's predict for 3 new days
# We must convert words to the same numbers as before!
# We use transform() (NOT fit_transform) — same encoding as training

new_days = [
    {"Outlook": "Sunny",    "Humidity": "High",   "Wind": "Weak"},
    {"Outlook": "Rain",     "Humidity": "Normal", "Wind": "Weak"},
    {"Outlook": "Overcast", "Humidity": "High",   "Wind": "Strong"},
]

print("=== STEP 6: Predictions for new days ===")
print()

for day in new_days:
    # Convert each word to its number using the encoders we trained
    outlook_num  = encoder_outlook.transform([day["Outlook"]])[0]
    humidity_num = encoder_humidity.transform([day["Humidity"]])[0]
    wind_num     = encoder_wind.transform([day["Wind"]])[0]

    # Make a single row of input data for the model
    single_input = [[outlook_num, humidity_num, wind_num]]

    # Get prediction and probabilities
    prediction  = model.predict(single_input)[0]
    probability = model.predict_proba(single_input)[0]

    # Convert number back to "Yes" or "No"
    result = encoder_label.inverse_transform([prediction])[0]

    print(f"Weather: Outlook={day['Outlook']}, Humidity={day['Humidity']}, Wind={day['Wind']}")
    print(f"  --> Should I play tennis? {result}")
    print(f"  --> Probability YES: {probability[1]*100:.1f}%")
    print(f"  --> Probability NO : {probability[0]*100:.1f}%")
    print()


