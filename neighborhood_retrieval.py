import heapq
import itertools
import numpy as np
import pickle
import time
from tqdm import tqdm


def fun(root_path):
    dataset = 'assist2009'
    phrase = 'train'
    maxstep = 50
    # num_question = 16891
    file_path = 'dataSet/2009/' + dataset + '_' + phrase + '.csv'
    data = []
    max_q = 0
    max_c = 0
    with open(file_path, 'r') as file:

        for length, ques, concepts, ans in tqdm(itertools.zip_longest(*[file] * 4)):

            concepts = np.array(concepts.strip().strip(',').split(',')).astype(np.int64)
            ques = np.array(ques.strip().strip(',').split(',')).astype(np.int64)
            if ques.max()>max_q:max_q = ques.max()
            if concepts.max() > max_c: max_c = concepts.max()
            length = len(ques)
            ans = np.array(ans.strip().strip(',').split(',')).astype(np.int64)
            slices = length // maxstep + (1 if length % maxstep > 0 else 0)
            for i in range(slices):
                temp = [[0] * 3  for _ in range(maxstep)]
                if length > 0:
                    if length >= maxstep:
                        steps = maxstep
                    else:
                        steps = length
                for j in range(steps):
                        temp[j] = [ques[i * maxstep + j], concepts[i * maxstep + j], ans[i * maxstep + j]]
                length = length - maxstep
                data.append(temp)
        print('start write dataSet..........' + 'number of student: ' + str(len(data)))
        start = time.time()
        pickle.dump(data, open('dataSet/2009/raw/' + dataset + phrase + '_data.txt', 'wb'))
        end = time.time()
        print('end write dataSet..........\n' + 'useTime:' + str(end - start))
        print(max_c)
        print(max_q)

def getSimilar_record(raw_data):
    phrase = 'train'
    num_questions = 10795
    num_skills = 1676
    raw_data = np.asarray(raw_data)
    ques = raw_data[:, :, 0]
    skills = raw_data[:, :, 1]
    ques_similar_record = []
    skills_similar_record = []
    for que_id in tqdm(range(num_questions)):
        q_index = np.argwhere(ques == que_id + 1)
        q = [tuple(t) for t in q_index]
        ques_similar_record.append(q)
    for s_id in tqdm(range(num_skills)):
        s_index = np.argwhere(skills == s_id + 1)
        s = [tuple(t) for t in s_index]
        skills_similar_record.append(s)
    print('start write neighborhoods_data..........')
    start = time.time()
    pickle.dump(ques_similar_record, open('dataSet/ednet/similar_records/' + phrase + '_ques_neighborhoods.txt', 'wb'))
    pickle.dump(skills_similar_record, open('dataSet/ednet/similar_records/' + phrase + '_skills_neighborhoods.txt', 'wb'))
    end = time.time()
    print('end write neighborhoods_data..........' + str(end - start))


def getTop_k_sub_seq(train_raw_Data, test_raw_Data, ques_similar_record, skills_similar_record, phrase):
    train_data = np.asarray(train_raw_Data)
    test_data = np.asarray(test_raw_Data)
    if phrase == 'test':
        raw_data = test_data
    else:
        raw_data = train_data
    all_similar_records = []
    for idx, one_Student in tqdm(enumerate(raw_data), total=len(raw_data)):
        one_Student_similar_records = []
        for idy, step in enumerate(one_Student):
            q = step[0]
            if q == 0:
                continue
            s = step[1]
            Q_me_SubSqueeze = one_Student[:idy, 0]
            S_me_SubSqueeze = one_Student[:idy, 1]
            Q_others = ques_similar_record[q - 1]  # [(1, 2),(4, 5),(7, 8)]
            S_others = skills_similar_record[s - 1]
            if phrase == 'train':
                Q_others.remove((idx, idy))
                S_others.remove((idx, idy))
            all_others = list(set(Q_others)).union(set(S_others))
            Q_others_SubSqueeze = [train_data[recoder[0], :recoder[1], 0] for recoder in all_others]
            S_others_SubSqueeze = [train_data[recoder[0], :recoder[1], 1] for recoder in all_others]
            Q_intersection = [len(np.intersect1d(Q_others_SubSqueeze[i],Q_me_SubSqueeze)) for i in range(len(Q_others_SubSqueeze))]
            S_intersection = [len(np.intersect1d(S_others_SubSqueeze[i],S_me_SubSqueeze)) for i in range(len(S_others_SubSqueeze))]
            Q_union = [len(np.union1d(Q_others_SubSqueeze[i], Q_me_SubSqueeze)) for i in
                              range(len(Q_others_SubSqueeze))]
            S_union = [len(np.union1d(Q_others_SubSqueeze[i], Q_me_SubSqueeze)) for i in
                              range(len(S_others_SubSqueeze))]
            Q_union = np.where(Q_union ==0 , Q_union, 1)
            S_union = np.where(S_union == 0, S_union, 1)
            Q_score = [a / b for a, b in zip(Q_intersection , Q_union)]
            S_score = [a / b for a, b in zip(S_intersection , S_union)]
            score = np.asarray(Q_score) + np.asarray(S_score)
            top_k_index = heapq.nlargest(20, range(len(score)), score.__getitem__)
            one_Student_similar_records.append([all_others[i] for i in top_k_index])
        all_similar_records.append(one_Student_similar_records)
    print('start write all_similar_records_data..........')
    start = time.time()
    pickle.dump(all_similar_records, open('dataSet/2017/similar_records/'+phrase+'_all_similar_records.txt', 'wb'))
    end = time.time()
    print('end write all_similar_records_data..........' + str(end - start))


if __name__ == '__main__':
    # fun(None)
    dataSet = '2017'
    train_raw_data = pickle.load(open('dataSet/'+dataSet+'/raw/assist'+dataSet+'train_data.txt', 'rb'))
    test_raw_data = pickle.load(open('dataSet/'+dataSet+'/raw/assist'+dataSet+'test_data.txt', 'rb'))
    ques_nighs = pickle.load(open('dataSet/'+dataSet+'/similar_records/train_ques_neighborhoods.txt', 'rb'))
    skills_nighs = pickle.load(open('dataSet/'+dataSet+'/similar_records/train_skills_neighborhoods.txt', 'rb'))
    getTop_k_sub_seq(train_raw_data, test_raw_data, ques_nighs, skills_nighs, 'test')
    # getSimilar_record(train_raw_data)
    # neigh = pickle.load(open(dataSet.getRootPath() + '/dataset/assist2017/raw/train_neighborhoods.txt', 'rb'))
