import logging, torch, torchvision
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from polyvore_dataset import CategoryDataset, collate_fn

# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Save pytorch model with best performance
class BestSaver(object):
    def __init__(self, comment=None):
        # Get current executing script name
        import __main__, os
        exe_fname=os.path.basename(__main__.__file__)
        save_path = "model_{}".format(exe_fname.split(".")[0])
        
        if comment is not None and str(comment):
            save_path = save_path + "_" + str(comment)

        save_path = save_path + ".pth"

        self.save_path = save_path
        self.best = float('-inf')

    def save(self, metric, data):
        if metric > self.best:
            self.best = metric
            torch.save(data, self.save_path)
            logging.info("Saved best model to {}".format(self.save_path))

# Configure logging for training log
def config_logging(comment=None):
    # Get current executing script name
    import __main__, os
    exe_fname=os.path.basename(__main__.__file__)
    log_fname = "log_{}".format(exe_fname.split(".")[0])

    if comment is not None and str(comment):
        log_fname = log_fname + "_" + str(comment)

    log_fname = log_fname + ".log"
    log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_fname), logging.StreamHandler()]
    )

def prepare_dataloaders(root_dir="/root/fashion_CSCE670/data/images/", data_dir="/root/fashion_CSCE670/data/", batch_size=16, img_size=224, use_mean_img=True, neg_samples=True, num_workers=1, collate_fn=collate_fn):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_dataset = CategoryDataset(
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file="train_no_dup_with_category_3more_name.json",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_dataset = CategoryDataset(
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file="valid_no_dup_with_category_3more_name.json",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    test_dataset = CategoryDataset(
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file="test_no_dup_with_category_3more_name.json",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader