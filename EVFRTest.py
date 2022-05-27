import torch
import torchvision
import PIL.Image as Image
import numpy as np
from Models.EVFRNet import EVFRNet

def get_image(img_path):
    with open(img_path, 'rb') as img_file:
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = img.resize((560, 420))
    return img

def decode_segmap_C4(image, nc=4):
    label_colors = np.array([(0, 0, 0), (0, 0, 142), (150, 120, 90), (255, 0, 0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def predict(img_list:list, device='cpu', RGB=False) -> list:
    num_img = len(img_list)
    trans = torchvision.transforms.ToTensor()

    img_tensor_list = [trans(img_list[i]).unsqueeze(0) for i in range(num_img)]
    img_tensor_batch = torch.cat(img_tensor_list, dim=0)

    img_tensor_batch = img_tensor_batch.to(device)

    ckpt_base = 'ckpt/'
    model = EVFRNet(2, 'LargerASPP', 'DAEMA')
    model.to(device)

    backbone_path = ckpt_base + 'backbone_ckpt_epoch_10.pth'
    decoder1_path = ckpt_base + 'decoder1_ckpt_epoch_10.pth'
    decoder2_path = ckpt_base + 'decoder2_ckpt_epoch_10.pth'

    backbone_ckpt = torch.load(backbone_path, map_location=device)
    decoder1_ckpt = torch.load(decoder1_path, map_location=device)
    decoder2_ckpt = torch.load(decoder2_path, map_location=device)

    model.backbone.load_state_dict(backbone_ckpt['net'])
    model.decoder1.load_state_dict(decoder1_ckpt['net'])
    model.decoder2.load_state_dict(decoder2_ckpt['net'])

    # param_bb, param_d1, param_d2 = 0, 0, 0
    # for param in model.backbone.parameters():
    #     param_bb += param.view(-1).size()[0]
    # for param in model.decoder1.parameters():
    #     param_d1 += param.view(-1).size()[0]
    # for param in model.decoder2.parameters():
    #     param_d2 += param.view(-1).size()[0]
    #
    # print(param_bb, param_d1, param_d2)

    model.eval()

    out = model(img_tensor_batch)['res_all']

    if RGB:
        pred = [decode_segmap_C4(torch.argmax(out[k], dim=0).detach().cpu().numpy()) for k in range(num_img)]
    else:
        pred = [torch.argmax(out[k], dim=0).detach().cpu().numpy() for k in range(num_img)]

    return pred

'''
Params:
    42500160 43787266 6499844
GPU:
    1084 ms per image
CPU (server):
    4068 ms per image
'''

def try_samples():
    import time
    num_img = 5
    img1 = get_image('sample_image/A.png')
    img2 = get_image('sample_image/B.png')
    img3 = get_image('sample_image/C.png')
    img4 = get_image('sample_image/D.png')
    img5 = get_image('sample_image/E.png')
    img_list = [img1, img2, img3, img4, img5]

    start_time = time.time()
    pred = predict(img_list, device='cuda:1')
    end_time = time.time()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.axis('off')
    for i in range(num_img):
        plt.subplot(num_img, 2, 2 * i + 1)
        pred_img = decode_segmap_C4(pred[i])
        plt.imshow(pred_img)
        plt.subplot(num_img, 2, 2 * i + 2)
        plt.imshow(img_list[i])
    plt.show()

    elapsed = end_time - start_time
    print('{:.2f} s elapsed, {:.0f} ms per image'.format(elapsed, elapsed * 1000 / num_img))

def try_one():
    import time
    num_img = 5
    img1 = get_image('zero_image/14.jpg')
    img_list = [img1]

    start_time = time.time()
    pred = predict(img_list, device='cuda:1')
    end_time = time.time()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.axis('off')
    pred_img = decode_segmap_C4(pred[0])
    plt.subplot(121)
    plt.imshow(pred_img)
    plt.subplot(122)
    plt.imshow(img_list[0])
    plt.show()

    elapsed = end_time - start_time
    print('{:.2f} s elapsed, {:.0f} ms per image'.format(elapsed, elapsed * 1000 / num_img))


if __name__ == '__main__':
    try_one()


