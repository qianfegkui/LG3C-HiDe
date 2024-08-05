import copy
import time

import numpy as np
import torch
from tqdm import tqdm

import himallgg

log = himallgg.utils.get_logger()
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn import metrics


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None
        self.num_classes = 6
        self.class_counts = torch.zeros(self.num_classes).to(self.args.device)

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_tes_f1 = ckpt["best_tes_f1"]
        self.test_f1_when_best_dev = ckpt['test_f1_when_best_dev']
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1,best_tes_f1, best_epoch, best_state = self.best_dev_f1, self.best_tes_f1,self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            
            dev_f1 = self.evaluate()
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            test_f1 = self.evaluate(test=True)
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            if best_tes_f1 is None or test_f1 > best_tes_f1:
                best_tes_f1 = test_f1
                test_f1_when_best_dev = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))
            log.info("** Best in epoch {} **:".format(best_epoch))
            log.info("** Best best_dev_f1 {} **:".format(test_f1_when_best_dev))
            log.info("** Best f1 {} **:".format(best_tes_f1))

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1 = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1 = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))

        return best_dev_f1, best_epoch, best_state, test_f1_when_best_dev, best_tes_f1

    def update_class_counts(self, class_counts, data, num_classes, device):
        """
        Update the class counts based on the label tensor in the data.

        Args:
            class_counts (torch.Tensor): Tensor storing the current counts of each class.
            data (dict): A dictionary containing the data, including 'label_tensor'.
            num_classes (int): The number of classes.
            device (str): The device to run the operation on (e.g., 'cuda:0').

        Returns:
            torch.Tensor: Updated class counts tensor.
        """
        labels = data['label_tensor'].to(device)
        labels = labels.reshape(-1)  # Flatten the label tensor if necessary
        class_counts += torch.bincount(labels, minlength=num_classes).float().to(device)
        return class_counts


    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            # if(epoch==1):
            #     self.class_counts = self.update_class_counts(self.class_counts, data, self.num_classes, self.args.device)
            # print( self.class_counts)
            for k, v in data.items():
                if k == 'sentence':
                    continue
                else:
                    data[k] = v.to(self.args.device)
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
 
        
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if k == 'sentence':
                        continue
                    else:
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            
            print(metrics.classification_report(golds, preds, digits=4))
            f1 = metrics.f1_score(golds, preds, average="weighted")
            print(confusion_matrix(golds,preds))

        return f1

