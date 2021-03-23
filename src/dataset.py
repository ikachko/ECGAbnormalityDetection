class ECGDataset(data.Dataset):
    def __init__(self, csv_path, sig_limit=None):
        """

        :param csv_path:
        :param sig_limit:
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path + " was not found !")

        self.record_df = pd.read_csv(csv_path)
        
        if sig_limit is not None:
            self.record_df = self.record_df[self.record_df.Length == sig_limit]

        self.prefix = prefix
    
    def __len__(self) -> int:
        return self.record_df.shape[0]
    
    def __getitem__(self, index: int) -> (np.ndarray, str):
        """
        Get a file by passing an index
        """
        
        # Get a signal
        record_path = self.prefix + self.record_df.PathToData.iloc[index]
        signal = sio.loadmat(record_path)['val']

        # Get a disease
        dx = self.record_df.Disease.iloc[index]

        return signal, dx