from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import STL10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class STL10DataModule(STL10):
    def __init__(self, root: str = None,
                 split: str = "train", 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        if root==None:
            root = "./data"
        if transform==None:
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, label = super().__getitem__(index)
        return (data, label)

    def __len__(self) -> int:
        return super().__len__()

    def get_dataloader(self, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)