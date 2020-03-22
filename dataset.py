from torch.utils.data import Dataset
import numpy as np

class WrapperDataset(Dataset):
    """
    A Dataset object wrapper for the initialized from numpy arrays.
    """

    def __init__(self, X, y):
        # Shape has to be (Batch size, data
        self.X = X.astype(np.float32)
        # self.X = np.expand_dims(X, axis=-1)

        # To return LongTensor
        self.y = np.where(y == 'FAKE', 0, 1).astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve images by their index.
        :param idx: index of the image in the dataset.
        :return: (X[idx], y[idx]): the image and its label
        """
        return self.X[idx], self.y[idx]

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y