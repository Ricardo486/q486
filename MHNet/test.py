import time
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
sys.path.append('../models')
import numpy as np
import os, argparse
import cv2
from models.MHNet import Net
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./COD_datasets/TestDataset/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

import pickle
from modules.module import DCTFeatureExtractor

with open('./freq_mean_std.pkl', 'rb') as f:
    freq_stats = pickle.load(f)
    freq_mean = torch.tensor(freq_stats['mean'])
    freq_std = torch.tensor(freq_stats['std'])
dct_module = DCTFeatureExtractor(freq_mean, freq_std)

model = Net(dct_module)

modelname = 'MHNet'

model.load_state_dict(torch.load('./weight.pth'))
model.cuda()
model.eval()

test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
for dataset in test_datasets:
    save_path = './' + modelname + '/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        start_time = time.perf_counter()
        res, _, _, _, _, _, _ = model(image)
        end_time = time.perf_counter()
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
    print(dataset + '..Test Done!')

for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    mask_root = './COD_datasets/TestDataset/{}/GT'.format(_data_name)
    pred_root = './' + modelname + '/' + _data_name + '/'
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "meanEm": em["curve"].mean(),
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
    }
    print(results)
    file = open("evalresults_" + modelname + ".txt", "a")
    file.write(_data_name + ' ' + str(results) + '\n' + '\n')
