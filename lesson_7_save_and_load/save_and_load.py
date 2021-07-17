import torch
import torch.onnx as onnx
import torchvision.models as models

# Load pre-trained and save
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# Create new and load the saved model into it
model = models.vgg16()  # pretraint=False
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Saving model and class structure together
torch.save(model, 'model_with_class_structure.pth')
# Load
model = torch.load('model_with_class_structure.pth')


# Export model to ONNX
input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')
