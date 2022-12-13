import torch
import requests

class Trainer:
        def __init__(self,model,dataset,train_dataloader, test_dataloader,epochs=10):
            self.model=model
            self.dataset=dataset
            self.train_dataloader = train_dataloader
            self.test_dataloader = test_dataloader            
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
            self.loss_fn=torch.nn.CrossEntropyLoss().cuda()
            self.epochs = epochs
        
        # Training and testing loop
        def train_loop(self,dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                X=X.cuda()
                y=y.cuda()
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        def val_loop(self,dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            test_loss, correct = 0, 0
            model.eval()
            with torch.no_grad():
                for X, y in dataloader:
                    X=X.cuda()
                    #X=amplitude_only(X)
                    y=y.cuda()
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        def train_epochs(self):
            print(f"Trained model doesnot exists at the specified path.\nTraining from scratch...")
            for t in range(self.epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                self.train_loop(self.train_dataloader, self.model, self.loss_fn, self.optimizer)
                self.val_loop(self.test_dataloader, self.model, self.loss_fn)

        def train(self, use_model_checkpoint=True):
            if self.dataset.lower()=="cifar10":
                model_path = "saved_models/resnet18_cif_10.pt"
                if use_model_checkpoint == False:
                    self.train_epochs()
                    torch.save(self.model.state_dict(), model_path)
                else:
                    try:
                        print(f"Loading the trained model-checkpoint of {self.dataset}")
                        self.model.load_state_dict(torch.load(model_path))
                    except FileNotFoundError:
                        print("Trained model-checkpoint not found, downloading the pretrained checkpoint")
                        ckpt_url = "https://seafile.rlp.net/f/d866d7baeba4456b86de/?dl=1"
                        r = requests.get(ckpt_url)
                        with open(model_path,"wb") as f:
                            f.write(r.content)
                        print("Checkpoint is downloaded")
                        self.model.load_state_dict(torch.load(model_path))
                    print("Validating the model")
                    self.val_loop(self.test_dataloader, self.model, self.loss_fn)
                    
            elif self.dataset.lower()=="mnist":
                model_path = "saved_models/resnet18_mnist.pt"
                if use_model_checkpoint == False:
                    self.train_epochs()
                    torch.save(self.model.state_dict(), model_path)
                else:
                    try:
                        print(f"Loading the trained model-checkpoint of {self.dataset}")
                        self.model.load_state_dict(torch.load(model_path))
                    except FileNotFoundError:
                        print("Trained model-checkpoint not found, downloading the pretrained checkpoint")
                        ckpt_url = "https://seafile.rlp.net/f/37d38461df6a4d29be48/?dl=1"
                        r = requests.get(ckpt_url)
                        with open(model_path,"wb") as f:
                            f.write(r.content)
                        print("Checkpoint is downloaded")
                        self.model.load_state_dict(torch.load(model_path))
                    print("Validating the model")
                    self.val_loop(self.test_dataloader, self.model, self.loss_fn)
            elif self.dataset.lower()=="fashionmnist":
                model_path = "saved_models/resnet18_fashionmnist.pt"
                if use_model_checkpoint == False:
                    self.train_epochs()
                    torch.save(self.model.state_dict(), model_path)
                else:
                    try:
                        print(f"Loading the trained model-checkpoint of {self.dataset}")
                        self.model.load_state_dict(torch.load(model_path))
                    except FileNotFoundError:
                        print("Trained model-checkpoint not found, downloading the pretrained checkpoint")
                        ckpt_url = "https://seafile.rlp.net/f/4b78fb1abf5741f799c3/?dl=1"
                        r = requests.get(ckpt_url)
                        with open(model_path,"wb") as f:
                            f.write(r.content)
                        print("Checkpoint is downloaded")
                        self.model.load_state_dict(torch.load(model_path))
                    print("Validating the model")
                    self.val_loop(self.test_dataloader, self.model, self.loss_fn)
            elif self.dataset.lower()=="stl10" :
                model_path = "saved_models/resnet18_stl10.pt"
                if use_model_checkpoint == False:
                    self.train_epochs()
                    torch.save(self.model.state_dict(), model_path)
                else:
                    try:
                        print(f"Loading the trained model-checkpoint of {self.dataset}")
                        self.model.load_state_dict(torch.load(model_path))
                    except FileNotFoundError:
                        print("Trained model-checkpoint not found, downloading the pretrained checkpoint")
                        ckpt_url = "https://seafile.rlp.net/f/94f4627764414af0ba8b/?dl=1"
                        r = requests.get(ckpt_url)
                        with open(model_path,"wb") as f:
                            f.write(r.content)
                        print("Checkpoint is downloaded")
                        self.model.load_state_dict(torch.load(model_path))
                    print("Validating the model")
                    self.val_loop(self.test_dataloader, self.model, self.loss_fn)
            return self.model



