
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import copy

# Load and preprocess images
def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # convert to tensor

    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Convert tensor to image
def im_convert(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Load VGG19 Model
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def get_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)

    # Normalization for pre-trained models
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1), 
                                  nn.InstanceNorm2d(3))

    # Model layers for content and style losses
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, content_losses, style_losses

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, 
                       style_weight=1000000, content_weight=1):
    model, content_losses, style_losses = get_model_and_losses(cnn, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss : {style_score.item():4f}, Content Loss : {content_score.item():4f}")

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size and file paths
imsize = 512 if torch.cuda.is_available() else 128  # use small size for CPU

content_img_path = "path_to_room_image.jpg"
style_img_path = "path_to_style_image.jpg"

content_img = image_loader(content_img_path, imsize, device)
style_img = image_loader(style_img_path, imsize, device)

# Input image to optimize (content image initialized)
input_img = content_img.clone()

# Load VGG19 model pre-trained on ImageNet
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Run style transfer
output = run_style_transfer(cnn, content_img, style_img, input_img)

# Save the output
output_image = im_convert(output)
output_image.save("stylized_room_output.jpg")
