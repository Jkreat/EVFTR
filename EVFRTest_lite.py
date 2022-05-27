import torch
import PIL.Image as Image
import numpy as np
from Models.EVFRNet_lite import EVFRNet_lite

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

def to_tensor(pic):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

def predict(img_list:list, device='cpu', RGB=False) -> list:
    num_img = len(img_list)

    img_tensor_list = [to_tensor(img_list[i]).unsqueeze(0) for i in range(num_img)]
    img_tensor_batch = torch.cat(img_tensor_list, dim=0)

    img_tensor_batch = img_tensor_batch.to(device)

    ckpt_base = 'ckpt_lite/'
    model = EVFRNet_lite(2, 'ASPP', 'DAEMA')
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

    model.eval()

    out = model(img_tensor_batch)['res_all']
    # pred = [torch.argmax(out[k], dim=0).detach().cpu().numpy() for k in range(num_img)]

    if RGB:
        pred = [decode_segmap_C4(torch.argmax(out[k], dim=0).detach().cpu().numpy()) for k in range(num_img)]
    else:
        pred = [torch.argmax(out[k], dim=0).detach().cpu().numpy() for k in range(num_img)]

    return pred

'''
Params:
    23508032 16125954 6499844
GPU:
    747 ms per image
CPU (server):
    2369 ms per image
'''

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
    # import time
    # num_img = 5
    # img1 = get_image('sample_image/A.png')
    # img2 = get_image('sample_image/B.png')
    # img3 = get_image('sample_image/C.png')
    # img4 = get_image('sample_image/D.png')
    # img5 = get_image('sample_image/E.png')
    # img_list = [img1, img2, img3, img4, img5]
    #
    # device = 'cuda:0'
    # start_time = time.time()
    # pred = predict(img_list, device=device)
    # end_time = time.time()
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.axis('off')
    # for i in range(num_img):
    #     plt.subplot(num_img,2,2*i+1)
    #     pred_img = decode_segmap_C4(pred[i])
    #     plt.imshow(pred_img)
    #     plt.subplot(num_img,2,2*i+2)
    #     plt.imshow(img_list[i])
    # plt.show()
    #
    # elapsed = end_time-start_time
    # print('{:.2f} s elapsed, {:.0f} ms per image'.format(elapsed, elapsed*1000/num_img))
    try_one()

