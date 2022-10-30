import tqdm
import torch
import logging
import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().cpu().numpy(),
                          torch.round(prediction).detach().cpu().numpy())
    recall = metrics.recall_score(ground_truth.detach().cpu().numpy(),
                                  torch.round(prediction).detach().cpu().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().cpu().numpy(),
        torch.round(prediction).detach().cpu().numpy())
    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(),torch.round(prediction).detach().cpu().numpy())
    logger.info('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision)+ 'acc: ' + str(acc))
    print('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision) + ' acc: ' + str(acc))
    return auc, acc


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device

    def forward(self, pred, data):
        """
                :param pred: raw pred [B, L, H]
                :param data[0]: [B, L, 2]
                :return: ce loss
                """
        me = data[0]
        loss = 0
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        for student in range(me.shape[0]):
            index = torch.tensor([i for i in range(self.max_step - 1)], dtype=torch.long, device=self.device)
            p = pred[student, index, me[student, 1:, 0]-1]
            q = me[student, 1:, 0]
            a = me[student, 1:, 2]
            a = a.float()
            for i in range(len(p) - 1, -1, -1):
                if q[i] > 0:
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training:', mininterval=2):
        pred, closs = model(batch)
        loss, _, _ = loss_func(pred, batch)
        optimizer.zero_grad()
        loss = loss + closs
        loss.backward()
        optimizer.step()
    return model


def test_epoch(model, testLoader, loss_func, device):
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    with torch.no_grad():
        for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
            pred, _ = model(batch)
            loss, p, a = loss_func(pred, batch)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
    return performance(ground_truth, prediction)
