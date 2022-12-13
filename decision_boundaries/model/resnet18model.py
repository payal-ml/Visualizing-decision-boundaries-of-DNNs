from torchvision.models import resnet18
import torch
class ResnetClassifier:
        def __init__(self,input_channel=3, Output_channel=64,kernel_size=(7,7),stride=(2,2),padding=(3, 3),final_out=10):
            self.input_channel = input_channel
            self.output_channel = Output_channel
            self.kernel_size = kernel_size
            self.stride=stride
            self.padding=padding
            self.final_out=final_out

        def resnet18model(self):
            model = resnet18(pretrained=True) 
            model.conv1=torch.nn.Conv2d(self.input_channel, self.output_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
            model.fc = torch.nn.Linear(512,self.final_out,bias=True)
            if torch.cuda.is_available():
                model.cuda()
            return model

        @classmethod
        def resnet18_from_dataset(cls, dataset_name):
            if str.lower(dataset_name) not in ["cifar10","mnist","fashionmnist","stl10"]:
                raise ValueError("only supported for dataset CIFAR10, MNIST, FashionMNIST and STL10")
            if str.lower(dataset_name) == "cifar10":
                classifier = cls(kernel_size=(3,3), padding=(1,1), stride=(1,1))           
            elif str.lower(dataset_name) == "mnist":
                classifier = cls(input_channel=1)
            elif str.lower(dataset_name) == "fashionmnist":
                classifier = cls(input_channel=1)
            elif str.lower(dataset_name) == "stl10":
                classifier = cls(kernel_size=(3,3),stride=(1,1),padding=(1, 1))
            return classifier.resnet18model()