import json

# JSON data that needs filtering
f = open('/root/fashion_compatibility_mcn/data/valid_no_dup_with_category_3more_name.json','r')
existing_json_data = json.load(f)

# Function to filter out outfits containing the word "dress" in any part
def filter_out_dress(data):
    filtered_data = {}
    for outfit_number, outfit_parts in data.items():
        # Check both upper and lower parts for the word "dress" (case-insensitive)
        if not any("dress" in part["name"].lower() for part in outfit_parts.values()):
            filtered_data[outfit_number] = outfit_parts
    return filtered_data

# Filter the existing JSON data
filtered_json_data = filter_out_dress(existing_json_data)

# Create a new JSON file and write the filtered data to it
output_file_path = "valid_no_dup_with_category_3more_name.json"

with open(output_file_path, 'w') as json_file:
    json.dump(filtered_json_data, json_file, indent=4)

print(f"Filtered JSON data has been saved to {output_file_path}")