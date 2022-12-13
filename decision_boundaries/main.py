import argparse
from trainer.trainer import Trainer
from model.resnet18model import ResnetClassifier
from datamodules.cifar10 import CIFAR10DataModule
from datamodules.mnist import MNISTDataModule
from datamodules.fmnist import FashionMNISTDataModule
from datamodules.stl10 import STL10DataModule
from utils import plot_data
from pathlib import Path


parser= argparse.ArgumentParser()

parser.add_argument("--dataset",type=str,required=True,
                    help="Name of the supported dataset - cifar10, mnist, fashionmnist, stl10")
parser.add_argument("--attack_name",type=str,required=True,
                    help="Name of the supported attack - PGD, Carlini_Wagner")
parser.add_argument("--abs_stepsize",type=float, required=True,
                    help="abs_stepsize < epsilon e.g 1/256,....,8/256")
parser.add_argument("--epsilon", type=float, required=True,
                    help="epsilon value e.g 1/256,2/256,....,16/25")



def main():
    args=parser.parse_args()
    dataset_name=args.dataset
    attack_name=args.attack_name
    abs_stepsize=args.abs_stepsize
    epsilons=args.epsilon
    # dataset_name="cifar10"
    # attack_name="PGD"
    # abs_stepsize=2/256
    # epsilons=16/256
    saved_models_path = "./saved_models"
    results_path = "./results"
    Path(saved_models_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    print("Downloading required data")
    if dataset_name.lower()=="cifar10":
        trainset = CIFAR10DataModule(root="./data", train=True, download=True)
        testset = CIFAR10DataModule(root="./data", train=False, download=True)    
        batch_size=1000
       
        
    elif dataset_name.lower()=="mnist":
        trainset= MNISTDataModule(root="./data", train=True, download=True)
        testset=MNISTDataModule(root="./data", train=False, download=True)
        batch_size=1000
        
        
    elif dataset_name.lower()=="fashionmnist":
        trainset= FashionMNISTDataModule(root="./data", train=True, download=True)
        testset=FashionMNISTDataModule(root="./data", train=False, download=True)
        batch_size=1000
       
        
    elif dataset_name.lower()=="stl10":
        trainset=STL10DataModule(root="./data", split="train", download=True)
        testset=STL10DataModule(root="./data", split="test", download=True)
        batch_size=500
        
    model=ResnetClassifier.resnet18_from_dataset(dataset_name)
    
    nameofclasses=testset.classes
    train_dataloader = trainset.get_dataloader(batch_size=64,shuffle=True, num_workers=2)
    test_dataloader = testset.get_dataloader(batch_size=64, shuffle=False, num_workers=2)
    sample_dataloader = testset.get_dataloader(batch_size=batch_size, shuffle=False, num_workers=2)
    
    trainer = Trainer(model=model,
                     dataset=dataset_name,
                     train_dataloader=train_dataloader, 
                     test_dataloader=test_dataloader,
                     epochs=30)
    

    model = trainer.train()
    print("Generating visualizations")
    plot_data(model,dataset_name,sample_dataloader,nameofclasses,attack_name,epsilons,abs_stepsize)
    print("Done!")

if __name__ == "__main__":
    main()



