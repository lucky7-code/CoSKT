"""
Usage:
    run.py  [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 16891]
    --skills=<int>                      num of skill [default: 101]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 16]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 20]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimension of hidden state [default: 80]
    --models=<string>                    models type [default: CoSKT]
    --dataset=<string>                  dataset [default: 2009]
"""

import random
import logging
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from docopt import docopt
from dataLoader.dataloader import getDataLoader
from evaluation import eval
from models.CoSKT import CoSKT



def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global model
    args = docopt(__doc__)
    dataset = args['--dataset']
    length = int(args['--length'])
    questions = int(args['--questions'])
    skills = int(args['--skills'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = int(args['--cuda'])
    hidden = int(args['--hidden'])
    model_type = args['--models']

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_{dataset}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda', cuda)
    else:
        device = torch.device('cpu')

    train_loader, test_loader = getDataLoader(bs, device, dataset)
    model = CoSKT(bs, questions, hidden, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, length, device)
    best_auc = 0
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        print('epoch: ' + str(epoch + 1))
        model = eval.train_epoch(model, train_loader, optimizer,
                                             loss_func)
        logger.info(f'epoch {epoch + 1}')
        epoch_auc, acc = eval.test_epoch(model, test_loader, loss_func, device)
        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_acc = acc
            torch.save({'state_dict': model.state_dict()}, 'checkpoint/' + model_type + dataset+'CL.pth.tar')
            best_epoch = epoch + 1
            logger.info(f'best_auc:{best_auc}')
        print('best_epoch: %d  best_acc: %f  best_auc: %f ' % (best_epoch, best_acc, best_auc))


if __name__ == '__main__':
    main()
