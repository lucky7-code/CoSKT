import numpy as np
import itertools
import tqdm


class DataReader:
    def __init__(self, train_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self, file_path):
        Data_ques = np.array([])
        Data_ans = np.array([])
        with open(file_path, 'r') as file:
            for _, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[file] * 4)):
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                length = len(ques)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)+1
                mod = 0 if length % self.maxstep == 0 else (self.maxstep - length % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                Data_ques = np.append(Data_ques, ques).astype(np.int)
                Data_ans = np.append(Data_ans, ans).astype(np.int)
        return Data_ques.reshape([-1, self.maxstep]), Data_ans.reshape([-1, self.maxstep])

    def getTrainData(self):
        print('loading train data...')
        return self.getData(self.train_path)

    def getTestData(self):
        print('loading test data...')
        return self.getData(self.test_path)
