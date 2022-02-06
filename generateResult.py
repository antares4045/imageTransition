import torch
import os
from firstLearn import CURRENT_DIRECTORY, folderMustExist, ImageRemembererNet,genPilImage

def init(device, firstModelPath, secondModelPath):
    net = ImageRemembererNet(500, 6)
    net.to(device)

    netOld = ImageRemembererNet(500, 6)
    netOld.to(device)

    netNew = ImageRemembererNet(500, 6)
    netNew.to(device)

    netOld.load(firstModelPath)
    old_state_dict = netOld.state_dict().copy()

    netNew.load(secondModelPath)
    new_state_dict = netNew.state_dict().copy()

    return net, old_state_dict, new_state_dict


def main(device, firstModelPath, secondModelPath, output_path, steps_count, height, width):
    net, old_state_dict, new_state_dict = init(device, firstModelPath, secondModelPath)
    X = torch.FloatTensor([(x,y) for x in range(height) for y in range(width)]).to(device)
    
    folderMustExist(output_path)

    keys = old_state_dict.keys()
    current_state_dict = old_state_dict.copy()
    step_state_dict = new_state_dict.copy()
    for key in map(str, keys):
        step_state_dict[key] = (step_state_dict[key] - old_state_dict[key]) / steps_count

    images = []

    net.load_state_dict(current_state_dict)
    preds = net.inference(X)
    image = genPilImage(height, width, preds.cpu().detach().numpy())
    image.save(os.path.join(output_path,f"{0}.png"))
    images.append(image)

    for i in range(steps_count):
        for key in map(str, keys):
            current_state_dict[key] += step_state_dict[key]
        net.load_state_dict(current_state_dict)
        preds = net.inference(X)
        image = genPilImage(height, width, preds.cpu().detach().numpy())
        image.save(os.path.join(output_path,f"{i+1}.png"))
        images.append(image)

    images[0].save(
        os.path.join(CURRENT_DIRECTORY, f"out.gif"),
        save_all=True,
        append_images=images[1:] + images[-1:0:-1],
        optimize=True,
        duration=1,
        loop=0
        )
    os.system(f"ffmpeg -f image2 -r 10 -i {os.path.join(output_path, f'%01d.png')} -vcodec mpeg4 -y {os.path.join(CURRENT_DIRECTORY, f'out.mp4')}")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    firstModelPath = os.path.join(CURRENT_DIRECTORY, "models-first", "175.model")
    secondModelPath = os.path.join(CURRENT_DIRECTORY, "models-second", "8.model")

    output_path = os.path.join(CURRENT_DIRECTORY, "steps")
    steps_count = 100

    height = 100
    width = 100

    main(device, firstModelPath, secondModelPath, output_path, steps_count, height, width)