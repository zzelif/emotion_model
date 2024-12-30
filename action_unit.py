import pandas as pd
import os
#
# #Aggregate the individual csv
# csv_directory = "action_units/angry"
#
# all_data = []
#
# for file in os.listdir(csv_directory):
#     if file.endswith('.csv'):
#         file_path = os.path.join(csv_directory, file)
#         csv_data = pd.read_csv(file_path)
#
#         csv_data['filename'] = file
#
#         all_data.append(csv_data)
#
# combined_data = pd.concat(all_data, ignore_index=True)
#
# combined_data.to_csv("angry_aggregated_au_data.csv", index=False)
# print(f"Combined {len(all_data)} files into one dataset.")


# Paths for filtered CSVs
csv_paths = {
    'angry': "action_units/aggregate report/angry_aggregated_au_data.csv",
    'happy': "action_units/aggregate report/happy_aggregated_au_data.csv",
    'neutral': "action_units/aggregate report/neutral_aggregated_au_data.csv",
    'sad': "action_units/aggregate report/sad_aggregated_au_data.csv",
    'surprised': "action_units/aggregate report/surprised_aggregated_au_data.csv"
}

dfs = []

# Load and label each CSV
for emotion, path in csv_paths.items():
    df = pd.read_csv(path)
    df['emotion'] = emotion  # Add emotion label
    dfs.append(df)

# Combine all CSVs into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.columns = combined_df.columns.str.strip()

# Save combined data for future use
combined_df.to_csv("combined_au_data.csv", index=False)
print(f"Combined {len(combined_df)} rows into one dataset.")

# Debugging: Check the column names in combined_df
print("Columns in combined_df:", combined_df.columns)

# Define selected AUs
intensity_suffix = "_r"  # Change this to "_c" if you want confidence scores
selected_aus = [f"AU04{intensity_suffix}", f"AU09{intensity_suffix}", f"AU15{intensity_suffix}",
                f"AU17{intensity_suffix}", f"AU06{intensity_suffix}", f"AU07{intensity_suffix}",
                f"AU10{intensity_suffix}", f"AU12{intensity_suffix}", f"AU14{intensity_suffix}",
                f"AU20{intensity_suffix}", f"AU01{intensity_suffix}", f"AU02{intensity_suffix}",
                f"AU05{intensity_suffix}", f"AU25{intensity_suffix}", f"AU26{intensity_suffix}"]

# Check for missing columns
missing_columns = [col for col in selected_aus if col not in combined_df.columns]
if missing_columns:
    print(f"Warning: The following columns are missing from combined_df: {missing_columns}")
else:
    print("All selected AUs are present in combined_df.")

# Filter columns that exist in the DataFrame
selected_columns = ['filename'] + [col for col in selected_aus if col in combined_df.columns] + ['emotion']
filtered_df = combined_df[selected_columns]

# Save filtered data
filtered_df.to_csv("filtered_au_data.csv", index=False)
print(f"Filtered data saved with shape {filtered_df.shape}")

# Match filenames with image files
image_dir = "Dataset/Train"  # Base directory for images
image_files = []
for emotion in ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']:
    emotion_dir = os.path.join(image_dir, emotion)
    if os.path.exists(emotion_dir):
        image_files.extend([f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))])

# Filter rows based on matching filenames
filtered_df = filtered_df[filtered_df['filename'].isin(image_files)]

# Save the final dataset
output_path = "action_units/aggregate report/final_au_image_data.csv"
filtered_df.to_csv(output_path, index=False)
print(f"Final filtered data saved to {output_path} with shape {filtered_df.shape}")
