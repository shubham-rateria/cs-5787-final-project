import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get the text and the label for a specific index
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.transform:
            text = self.transform(text)
        
        return text, label
    
if __name__ == "__main__":
    dataset = CSVDataset(file_path="data/csv/generated_claim_triplets_with_topics.csv")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch in data_loader:
        print(batch)
        break

