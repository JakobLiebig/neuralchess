
import torch.utils.data as tdata

class DataframeDataset(tdata.Dataset):
    def __init__(self, feature_df, label_df, feature_transform, label_transform):
        self.feature_df = feature_df
        self.feature_transform = feature_transform
        
        self.label_df = label_df
        self.label_transform = label_transform

    def __getitem__(self, idx):
        features = self.feature_df.iloc[idx]
        features_t = self.feature_transform(features)

        label = self.label_df.iloc[idx]
        label_t = self.label_transform(label)

        return features_t, label_t

    def __len__(self):
        return len(self.feature_df)

