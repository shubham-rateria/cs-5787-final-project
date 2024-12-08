import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils import remove_duplicate_strings
# Custom Dataset to load data from one column of a CSV file
class CSVDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (str): Path to the CSV file.
            transform (callable, optional): Optional transform to apply to the data.
        """
        # Load the CSV file
        self.data = pd.read_csv(file_path)
        
        # Remove rows where the Evidence column has "Agent stopped due to iteration limit or time limit."
        self.data = self.data[self.data['Evidence'] != "Agent stopped due to iteration limit or time limit."]
        self.evidences = remove_duplicate_strings(self.data['Evidence'].to_list())
        
        # Reshape data into a single column with labels (0, 1, 2) for each category
        self.texts = []
        self.labels = []

        # Iterate through the rows and create a dataset of text and its corresponding label
        for index, row in self.data.iterrows():
            # Add the text and its associated label for each of the three columns
            self.texts.append(row['Supports'])
            self.labels.append(0)  # label 0 for 'supports'

            self.texts.append(row['Contradicts'])
            self.labels.append(1)  # label 1 for 'contradicts'
            
            self.texts.append(row['Ambiguous'])
            self.labels.append(2)  # label 2 for 'ambiguous'
        
        self.transform = transform
        
    def get_evidences(self):
        return self.evidences

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get the text and the label for a specific index
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.transform:
            text = self.transform(text)
        
        return text, label
    

def split_csv_file(file_path, train_file, val_file, test_file, test_size=0.2, val_size=0.15):
    # Read the CSV file
    data = pd.read_csv(file_path)
    data = data[data['Evidence'] != "Agent stopped due to iteration limit or time limit."]
    
    # Split the data into train and test sets
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    
    # Split the train set into train and validation sets
    train, val = train_test_split(train, test_size=val_size/(1-test_size), random_state=42)
    
    # Save the split data into separate CSV files
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)


if __name__ == "__main__":
    # dataset = CSVDataset(file_path="data/csv/generated_claim_triplets_with_topics.csv")
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # for batch in data_loader:
    #     print(batch)
    #     break
    split_csv_file("data/csv/generated_claim_triplets_with_topics.csv", "data/csv/train.csv", "data/csv/val.csv", "data/csv/test.csv")

