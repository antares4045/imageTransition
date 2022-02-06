import torch
import os
from PIL import Image





class ImageRemembererNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons=500, hidden_layers=4):

        super(ImageRemembererNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.fcIn = torch.nn.Linear(2, n_hidden_neurons)
        self.activIn = torch.nn.Tanh()

        for i in range(hidden_layers):
            setattr(self, f"fc{i}", torch.nn.Linear(n_hidden_neurons, n_hidden_neurons))
            setattr(self, f"activ{i}", torch.nn.Tanh())

        self.fcOut = torch.nn.Linear(n_hidden_neurons, 4)
        self.activOut = torch.nn.Sigmoid()


        self.loss = lambda output, target :torch.mean((output-target)**2)

        self.optimizer = torch.optim.Adam(self.parameters(), 
                             lr=1.0e-3)
        
    def forward(self, x):
        x = self.fcIn(x)
        x = self.activIn(x)
        for i in range(self.hidden_layers):
            x = getattr(self,f"fc{i}")(x)
            x = getattr(self,f"activ{i}")(x)

        x = self.fcOut(x)
        x = self.activOut(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        return x
    
    def learnBatch(self, x, y):
        self.optimizer.zero_grad()
        preds = self.forward(x) 
        loss_value = self.loss(preds, y)
        loss_value.backward()
        self.optimizer.step()
        return loss_value

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)
    
    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

def genPilImage(h, w, colours):
    img = Image.new(mode="RGBA", size=(h,w))
    pixels = img.load()
    for x in range(h):
        for y in range(w):
            colour = tuple(map(lambda sigVal: int((sigVal + 1) * 255 // 2), colours[x * w + y]))
            pixels[x, y] = colour
    return img


CURRENT_DIRECTORY = os.path.dirname(__file__)

def folderMustExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(imageName, models, previews, device, net):
    
    

    image = Image.open(imageName)
    X = []
    Y = []
    for x in range(image.height): # в PIL x - высота
        for y in range(image.width):
            X.append((x, y))
            Y.append(list(map(lambda value: (value * 2) / 255 - 1, image.getpixel((x, y)))))

    X = torch.FloatTensor(X).to(device)
    Y = torch.FloatTensor(Y).to(device)

    for epoch in range(175001):
        loss = net.learnBatch(X, Y)
        if epoch % 1000 == 0:
            print("epoch", epoch, "loss", loss)
            name = epoch // 1000
            net.save(os.path.join(models, f"{name}.model"))
            preds = net.inference(X)
            genPilImage(image.height, image.width, preds.cpu().detach().numpy()).save(os.path.join(previews,f"{name}.png"))


if __name__ == "__main__":
    imageName = os.path.join(CURRENT_DIRECTORY, "add.png")
    models = os.path.join(CURRENT_DIRECTORY, "models-first")
    previews = os.path.join(CURRENT_DIRECTORY, "repaired-first")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    folderMustExist(models)
    folderMustExist(previews)
    
    net = ImageRemembererNet(500, 6)
    net.to(device)

    main(imageName, models, previews, device, net)