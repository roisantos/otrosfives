import os
import argparse
import torch
import time
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import networks as models
from datasets.fives import FIVES
from tester import Tester
from utils import losses
from utils.helpers import get_instance

def main(CFG, check_path):
    weight_path = os.path.join(CFG.save_dir, CFG['dataset']['type'], CFG['loss']['type'], check_path)
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']

    # TODO: Fix dataset for test set.
    test_dataset = FIVES(CFG=CFG, mode="test")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)

    test = Tester(model, loss, CFG, checkpoint, test_loader, device)
    test.test()

if __name__ == '__main__':
    start_time = time.time()  # Start timer

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", default="config/fives_unet.yaml", help="Configuration file to load")
    parser.add_argument("-le", "--weight_path", default="checkpoint-epoch200.pth", type=str,
                        help='The path of weight.pt')
    
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))

    main(CFG=CFG, check_path=args.weight_path)

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
