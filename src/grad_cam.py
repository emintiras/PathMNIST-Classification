import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def gradcam(model, test_loader, class_names, device, num_images=3):
    """
    Generate and visualize Gradient-weighted Class Activation Maps (Grad-CAM) for model predictions.

    Args:
    model (torch.nn.Module): Trained PyTorch model (expected to be a ResNet-based architecture).
    test_loader (DataLoader): DataLoader containing test images.
    class_names (dict): Dictionary mapping class indices to class names.
    device (torch.device): Device to run the model on ('cuda' or 'cpu').
    num_images (int, optional): Number of images to visualize. Defaults to 3.    
    """
    target_layers = [model.layer4[-1]] 
    grad_cam = GradCAM(model=model, target_layers=target_layers)

    test_images, test_labels = next(iter(test_loader))
    images_to_explain = test_images[:num_images].to(device)
    labels_to_explain = test_labels[:num_images].squeeze().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images_to_explain)
        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(16, 10))
    for i in range(num_images):
        input_tensor = images_to_explain[i].unsqueeze(0)
        input_tensor.requires_grad=True

        target_category = predicted[i].item()
        targets = [ClassifierOutputTarget(target_category)]
        
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        rgb_img = images_to_explain[i].cpu().numpy().transpose(1, 2, 0)
        rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        rgb_img = np.clip(rgb_img, 0, 1)
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        confidence = nn.Softmax(dim=0)(outputs[i])[predicted[i]].item()

        plt.subplot(2, num_images, i+1)
        plt.imshow(rgb_img)
        plt.title(f"Original - True: {class_names[str(labels_to_explain[i].item())]}")
        plt.axis('off')
        
        plt.subplot(2, num_images, num_images+i+1)
        plt.imshow(visualization)
        plt.title(f"GradCAM - Pred: {class_names[str(predicted[i].item())]} (Conf: {confidence:.2f})")
        plt.axis('off')
        

    plt.tight_layout()
    plt.show()




