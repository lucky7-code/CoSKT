import pickle
import numpy as np
import torch.utils.data as Data
from dataLoader.DKTDataSet import DKTDataSet


def getDataLoader(batch_size, device, dataset):
    train_data = pickle.load(open('../dataSet/'+dataset+'/raw/assist'+dataset+'train_data.txt', 'rb'))
    train_similar_records = pickle.load(open('../dataSet/'+dataset+'/similar_records/train_all_similar_records.txt', 'rb'))
    d_train = DKTDataSet(np.asarray(train_data), np.asarray(train_data), train_similar_records, device)
    test_data = pickle.load(open('../dataSet/'+dataset+'/raw/assist'+dataset+'test_data.txt', 'rb'))
    test_similar_records = pickle.load(open('../dataSet/'+dataset+'/similar_records/test_all_similar_records.txt', 'rb'))
    d_test = DKTDataSet(np.asarray(train_data), np.asarray(test_data), test_similar_records, device)
    trainLoader = Data.DataLoader(d_train, batch_size=batch_size, shuffle=True, drop_last=True)
    testLoader = Data.DataLoader(d_test, batch_size=batch_size, shuffle=False, drop_last=True)
    return trainLoader, testLoader

