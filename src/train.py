import torch
import torch.nn as nn
import torch.optim as optim
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log") 
    ]
)

logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, learning_rate, num_epochs, device, tl=True):
    """
    Train a PyTorch model on PathMNIST data.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters() if tl else model.parameters(), lr=learning_rate)

    best_val_acc = 0
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).squeeze()
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0) 
                correct += predicted.eq(targets).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device).squeeze()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'saved_model/resnet50_pathmnist.pth' if tl else 'saved_model/model_pathmnist.pth')
                logger.info(f'New best model saved with val accuracy: {best_val_acc:.2f}%')                
        
        return model

    except Exception as e:
        logger.critical(f"An error occurred during training: {str(e)}", exc_info=True)

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    try:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total

        logger.info(f'Accuracy: {accuracy:.2f}%')

        return accuracy

    except Exception as e:
        logger.critical(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        return 0.0
