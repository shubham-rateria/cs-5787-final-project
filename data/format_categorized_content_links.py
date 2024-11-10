import json
import csv

# Load the JSON data
with open("categorized_content_links.json", "r") as file:
    data = json.load(file)

# Open a CSV file to write the data
with open("categorized_content_links.csv", "w", newline='', encoding='utf-8') as csvfile:
    # Define the column headers
    fieldnames = ["s1", "s2", "label", "url", "path1", "path2", "path3", "path4", "path5"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the headers to the CSV
    writer.writeheader()
    
    # Iterate through the JSON data and write rows to CSV
    for category, items in data.items():
        for entry in items:
            url = entry["url"]
            path = entry["path"]
            
            # Assume 10 rows per URL for placeholder sentence pairs
            for _ in range(10):
                # Prepare the row dictionary
                row = {
                    "s1": "",  # Placeholder for s1
                    "s2": "",  # Placeholder for s2
                    "label": "",  # Placeholder for label
                    "url": url,
                }
                
                # Add paths to the row, allowing up to 5 parts for flexibility
                for i, part in enumerate(path):
                    row[f"path{i+1}"] = part
                
                # Write the row to the CSV file
                writer.writerow(row)

print("CSV file 'categorized_content_links.csv' created successfully.")
