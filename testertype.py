import time
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
import ttach as tta

from trainer import Trainer
from utils.helpers import double_threshold_iteration
from utils.metrics import get_metrics, count_connect_component

# Mapping from final letter to image type name.
TYPE_MAP = {
    'N': 'Normal',
    'A': 'AMD',
    'D': 'DR',
    'G': 'Glaucoma'
}

class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, device):
        super(Trainer, self).__init__()
        self.device = device
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader

        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if device == 'cuda':
            cudnn.benchmark = True

        # This dictionary will hold lists of metric values per image type.
        self.per_type = {}  # e.g. { "Normal": { "AUC": [...], "F1": [...], ... }, ... }

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, total=len(self.test_loader))
        tic = time.time()

        with torch.no_grad():
            for i, batch_data in enumerate(tbar):
                # Expecting (img, gt, filename) from the FIVES dataset.
                if len(batch_data) == 3:
                    img, gt, filename = batch_data
                else:
                    img, gt = batch_data
                    filename = f"sample_{i}.png"

                self.data_time.update(time.time() - tic)
                img = img.to(self.device)
                gt = gt.to(self.device)
                
                pre = self.model(img)
                if isinstance(pre, tuple):
                    logits_aux, logits = pre
                    loss_val = self.loss(logits_aux, gt) + self.loss(logits, gt)
                    pre = logits
                else:
                    loss_val = self.loss(pre, gt)

                self.total_loss.update(loss_val.item())
                self.batch_time.update(time.time() - tic)

                # Double threshold iteration if enabled.
                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    metrics_dict = get_metrics(pre, gt, predict_b=pre_DTI)
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    metrics_dict = get_metrics(pre, gt, self.CFG.threshold)
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre, gt, threshold=self.CFG.threshold))
                
                # Update the overall metrics.
                self._metrics_update(*metrics_dict.values())

                # --- New: Accumulate per-image-type metrics ---
                base_name = os.path.splitext(os.path.basename(filename))[0]
                image_type_letter = base_name[-1]
                image_type = TYPE_MAP.get(image_type_letter, "Unknown")

                if image_type not in self.per_type:
                    self.per_type[image_type] = { key: [] for key in metrics_dict.keys() }
                for key, value in metrics_dict.items():
                    self.per_type[image_type][key].append(value)

                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} MCC {:.4f} | B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(),
                        self.batch_time.average, self.data_time.average
                    )
                )
                tic = time.time()

        logger.info("###### TEST EVALUATION ######")
        logger.info(f"test time:  {self.batch_time.average}")
        logger.info(f"     loss:  {self.total_loss.average}")
        if self.CFG.CCC:
            logger.info(f"     CCC:  {self.CCC.average}")
        for k, v in self._metrics_ave().items():
            logger.info(f"{str(k):5s}: {v}")

        # --- Report per-image-type averages ---
        logger.info("###### PER-TYPE AVERAGES ######")
        for typ, metrics_lists in self.per_type.items():
            averages = { k: (sum(vals)/len(vals) if vals else 0.0) for k, vals in metrics_lists.items() }
            logger.info(f"Type: {typ}")
            for metric, avg_val in averages.items():
                logger.info(f"   {metric:5s}: {avg_val:.4f}")
