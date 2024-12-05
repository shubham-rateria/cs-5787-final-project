import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset to load data from one column of a CSV file
class CSVDataset(Dataset):
    def __init__(self, file_path, column_name, transform=None):
        """
        Args:
            file_path (str): Path to the CSV file.
            column_name (str): Name of the column to extract.
            transform (callable, optional): Optional transform to apply to the data.
        """
        # Load the CSV file
        self.data = pd.read_csv(file_path)
        self.column_name = column_name
        self.transform = transform

    def __len__(self):
        # Return the total number of rows in the column
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the value from the specified column
        value = self.data.iloc[idx][self.column_name]

        # Convert to tensor
        value = torch.tensor(value, dtype=torch.float32)

        # Apply transformation if provided
        if self.transform:
            value = self.transform(value)

        return value