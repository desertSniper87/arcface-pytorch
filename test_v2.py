import torch
from torch import nn
from torch.nn import DataParallel
from torchvision.transforms import transforms

from config import Config
from models import resnet_face18, resnet34, resnet50

from PIL import Image

# normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

transform_ = transforms.Compose([
    # transforms.Resize(img_size),
    transforms.ToTensor(),
    # *normalize,
])

def get_predicted_class(img_path):
    img_path = f'{opt.data_root}/personai_icartoonface_rectest/icartoonface_rectest/{img_path}'
    x = model(transform_(Image.open(img_path).convert('RGB')).to("mps").unsqueeze(0))
    probabilities = nn.functional.softmax(x, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item()


correct_num, total_num = 0, 0
if __name__ == '__main__':

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
        

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
        model.load_state_dict(torch.load(opt.test_model_path, map_location='cuda'))
    elif torch.backends.mps.is_available():
        model.to(torch.device("mps"))
        model.load_state_dict(torch.load(opt.test_model_path, map_location='mps'))

    imgpaths = []
    imgpath_classids = []

    with open(f"{opt.data_root}/icartoonface_rectest_info.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_info = line.strip().split()
            if len(line_info) == 6:
                imgpaths.append(line_info[0])
                imgpath_classids.append(line_info[-1])
            if len(line_info) == 2:
                imgpath1, imgpath2 = line_info[0], line_info[1]
                idx1, idx2 = imgpaths.index(imgpath1), imgpaths.index(imgpath2)
                total_num += 1
                if get_predicted_class(imgpath1) == get_predicted_class(imgpath2) \
                        and imgpath_classids[idx1] == imgpath_classids[idx2]:
                    print(f'ok match', end=' ')
                    correct_num += 1
                elif imgpath_classids[idx1] == -1 or imgpath_classids[idx2] == -1:
                    print(f'ok no match', end=' ')
                    correct_num += 1
                elif get_predicted_class(imgpath1) != get_predicted_class(imgpath2) \
                        and imgpath_classids[idx1] != imgpath_classids[idx2]:
                    print(f'ok no match', end=' ')
                    correct_num += 1
                else:
                    print(f'not ok', end=' ')
                print(
                    f'\t{idx1}\t{idx2}\t{imgpath1}\t{imgpath_classids[idx1]}\t{get_predicted_class(imgpath1)}\t{imgpath2}\t{imgpath_classids[idx2]}\t{get_predicted_class(imgpath2)}',
                    100.0 * correct_num / total_num)

    print(100.0 * correct_num / total_num)