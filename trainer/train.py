from config import torch, optim, DEVICE, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, K_FOLDS, KFold, DataLoader, Subset, load_dataset
from model import SimplifiedVGG

class Trainer:
    def __init__(self):
        self.device = DEVICE
        self.num_classes = NUM_CLASSES
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.k_folds = K_FOLDS
        self.train_loader, _ = load_dataset.load_dataset(batch_size=self.batch_size, path="./data")
        self.kf = KFold(n_splits=self.k_folds, shuffle=True)

    def train(self):
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(range(len(self.train_loader.dataset)))):
            print(f"Fold {fold+1}/{self.k_folds}")
            train_subset = Subset(self.train_loader.dataset, train_idx)
            val_subset = Subset(self.train_loader.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
            
            model = SimplifiedVGG().to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            
            for epoch in range(self.num_epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device).long()
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct_train += (predicted == labels).sum().item()
                    total_train += labels.size(0)
                
                scheduler.step()

            torch.save(model.state_dict(), "trained_model.pth")
            print("Model saved as 'trained_model.pth'.")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
