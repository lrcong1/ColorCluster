import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

# Load color descriptions from a CSV file
df = pd.read_csv("/Users/zhongjin/Downloads/BZAN557/color.csv")
color_descriptions = df["x"].tolist()

# Define common base colors
common_base_colors = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
    "grey",  # Ensure "grey" is used as the standard
    "brown",
    "beige",
    "gold",
]


# Function to identify base color in description
def identify_base_color(description, base_colors):
    # Standardize "grey" and "gray" to ensure uniformity
    standardized_description = description.lower().replace("gray", "grey")
    for color in base_colors:
        if color in standardized_description:
            return color
    return "other"


# Preprocess descriptions to identify base colors
preprocessed_colors = [
    identify_base_color(desc, common_base_colors) for desc in color_descriptions
]

# Group descriptions by identified base color or 'other'
grouped_descriptions = {}
for desc, base_color in zip(color_descriptions, preprocessed_colors):
    if base_color in grouped_descriptions:
        grouped_descriptions[base_color].append(desc)
    else:
        grouped_descriptions[base_color] = [desc]

# For descriptions grouped as 'other', apply clustering based on text features
other_descriptions = grouped_descriptions.pop("other", [])
if other_descriptions:
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(other_descriptions)
    X_normalized = normalize(X)
    kmeans = KMeans(n_clusters=8, random_state=42)  # Adjust clusters as needed
    kmeans.fit(X_normalized)
    clusters = kmeans.labels_
    # Update grouped_descriptions with new clusters for 'other' descriptions
    for desc, cluster in zip(other_descriptions, clusters):
        cluster_key = f"Cluster {cluster + 1}"
        if cluster_key in grouped_descriptions:
            grouped_descriptions[cluster_key].append(desc)
        else:
            grouped_descriptions[cluster_key] = [desc]

# Convert the grouped descriptions to a DataFrame for easy viewing
grouped_colors_df = pd.DataFrame(
    [(key, ", ".join(values)) for key, values in grouped_descriptions.items()],
    columns=["Group", "Descriptions"],
)

# Display the DataFrame
print(grouped_colors_df)

# If you want to save this DataFrame to a CSV file
grouped_colors_df.to_csv("grouped_colors.csv", index=False)

from collections import Counter
import re


# Function to extract the most common words from descriptions
def extract_common_words(descriptions):
    # Combine all descriptions into a single string
    combined_descriptions = " ".join(descriptions)
    # Use regular expression to find all words, convert to lowercase
    words = re.findall(r"\b[a-z]+\b", combined_descriptions.lower())
    # Filter out common stop words (customize this list as needed)
    stop_words = set(["and", "the", "with", "in", "of", "a", "on", "for", "as", "is"])
    filtered_words = [word for word in words if word not in stop_words]
    # Count and return the most common word
    most_common_words = Counter(filtered_words).most_common(
        3
    )  # Adjust number as needed
    return most_common_words


# Iterate over each cluster to analyze descriptions
for index, row in grouped_colors_df.iterrows():
    # Assuming 'Descriptions' column contains the color descriptions as a string
    descriptions = row["Descriptions"].split(", ")  # Split descriptions into a list
    common_words = extract_common_words(descriptions)
    # Print the most common words for each cluster
    print(f"Group {row['Group']}: Most common words/themes: {common_words}")
    # Here you can decide on a new name based on the common words/themes

# Create a new DataFrame with expanded descriptions for each cluster
expanded_cluster_data = []

for index, row in grouped_colors_df.iterrows():
    cluster_name = row["Group"]
    descriptions = row["Descriptions"].split(", ")
    for description in descriptions:
        expanded_cluster_data.append(
            {"cluster_name": cluster_name, "description": description}
        )

# Convert the list of dictionaries to a DataFrame
expanded_cluster_df = pd.DataFrame(expanded_cluster_data)

# Display the new DataFrame
print(expanded_cluster_df)

expanded_cluster_df.to_csv(
    "/Users/zhongjin/Downloads/BZAN557/color_cluster.csv", index=False
)
