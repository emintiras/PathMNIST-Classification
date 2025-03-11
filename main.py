import torch
import logging
import os
from src.download_data import download_data
from src.model import PathMNISTModel, resnet_model
from src.train import train, test
from src.grad_cam import gradcam

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("logs.log")
    ]
)

logger = logging.getLogger(__name__)


def main(load_pretrained=False, tl=True):
    """
    Main function to run the PathMNIST classification pipeline.
    """
    try:

        if not os.path.exists('saved_model'):
            os.makedirs('saved_model')
            logger.info("Created 'saved_model' directory for storing model weights")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {device}")
        train_loader, val_loader, test_loader, class_names = download_data(batch_size=128, tl=tl)
        logger.info("Data loaders initialized.")
        
        if load_pretrained and os.path.exists('saved_model'):
            try:
                model = resnet_model(num_classes=9).to(device) if tl else PathMNISTModel(num_classes=9).to(device)
                model_path = 'saved_model/resnet50_pathmnist.pth' if tl else 'saved_model/model_pathmnist.pth'
                
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                    logger.info(f"Successfully loaded pretrained model from {model_path}")
                else:
                    logger.warning(f"No pretrained model found at {model_path}. Training a new model instead.")
                    model = train(model=model, train_loader=train_loader, val_loader=val_loader, 
                                 learning_rate=1e-3, num_epochs=10, device=device, tl=tl)
            except Exception as e:
                logger.error(f"Failed to load pretrained models: {str(e)}", exc_info=True)
                return
        else:
            logger.info('Training a model from scratch.')

            if tl == True:
                model = resnet_model(num_classes=9)
                model = train(model=model, train_loader=train_loader, val_loader=val_loader, 
                             learning_rate=1e-3, num_epochs=10, device=device, tl=tl)
            else:
                model = PathMNISTModel()
                model = train(model=model, train_loader=train_loader, val_loader=val_loader, 
                             learning_rate=1e-3, num_epochs=10, device=device, tl=tl)
            
            logger.info("The model training completed.")

        test_accuracy = test(model=model, test_loader=test_loader, device=device)
        logger.info(f"Final test accuracy: {test_accuracy:.2f}%")
        
        gradcam(model=model, test_loader=test_loader, class_names=class_names, device=device, num_images=3)
        logger.info("GradCAM visualizations generated")
        
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}", exc_info=True)
        

if __name__ == "__main__":
    load_pretrained, tl = True, True
    main(load_pretrained=load_pretrained, tl=tl)