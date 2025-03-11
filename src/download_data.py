import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medmnist import PathMNIST
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log",) 
    ]
)

logger = logging.getLogger(__name__)

def download_data(batch_size, data_root='data/', tl=True):
    """
    Download and prepare PathMNIST data with appropriate transformations.

    Args:
    batch_size (int): Batch size for the data loaders.
    data_root (str): Path to store/download datasets.
    tl (bool): Whether to use transfer learning transforms (True) or basic transforms (False).

    Returns:
    train_loader (DataLoader): DataLoader for training data with augmentations.
    val_loader (DataLoader): DataLoader for validation data.
    test_loader (DataLoader): DataLoader for test data.
    class_names (dict): Dictionary mapping class indices to class names.
    """
    try:
        logger.info('Downloading PathMNIST data...')

        if tl == True:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5]),
            ])


        train_dataset = PathMNIST(split='train', root=data_root, transform=train_transform, download=True)
        val_dataset = PathMNIST(split='val', root=data_root, transform=transform, download=True)
        test_dataset = PathMNIST(split='test', root=data_root, transform=transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        class_names = train_dataset.info['label']

        logger.info(f'PathMNIST data successfully downloaded with batch size {batch_size}.')
        logger.info(f'PathMNIST Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')

        return train_loader, val_loader, test_loader, class_names
    
    except Exception as e:
        logger.critical(f'Failed to initialize data loaders: {str(e)}', exc_info=True)
        raise


