import os
import torch

from firstLearn import CURRENT_DIRECTORY, folderMustExist, ImageRemembererNet, main

if __name__ == "__main__":
    imageName = os.path.join(CURRENT_DIRECTORY, "ball.png")
    models = os.path.join(CURRENT_DIRECTORY, "models-second")
    previews = os.path.join(CURRENT_DIRECTORY, "repaired-second")

    firstModel = os.path.join(CURRENT_DIRECTORY, "models-first", "175.model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    folderMustExist(models)
    folderMustExist(previews)
    
    net = ImageRemembererNet(500, 6)
    net.to(device)
    net.load(firstModel)

    main(imageName, models, previews, device, net)