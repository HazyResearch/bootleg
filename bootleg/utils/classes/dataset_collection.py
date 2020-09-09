

class DatasetCollection:
    def __init__(self, args, file, dataset, data_loader, data_slices, data_sample):
        self.args = args
        self.file = file
        self.dataset = dataset
        self.data_loader = data_loader
        self.data_slices = data_slices
        self.data_sample = data_sample

    def __str__(self):
        return f"DatasetCollection of {self.file}"