from torch import Dataset

class CenterNetDataset(Dataset):
    def __init__(self, data_dir, split='train', input_size=512, max_obj=50):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.output_size = input_size // 4
        self.max_obj = max_obj

        