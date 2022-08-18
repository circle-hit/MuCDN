import pickle
from torch.utils.data.dataset import Dataset
from tqdm import tqdm, trange
import numpy
import torch
import json
import random

def get_speaker_graph(dataset, totalIds, speakers, position_upper=5):
    inter_speaker_graph, intra_speaker_graph = {}, {}
    inter_position_index, intra_position_index = {}, {}
    for ids in tqdm(totalIds):
        if dataset == 'meld':
            cur_speaker_list = [numpy.argmax(i) for i in speakers[ids]]
        else:
            cur_speaker_list = speakers[ids]
        cur_inter_speaker_graph = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_speaker_graph = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_position_index = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_position_index = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        
        cur_intra_speaker_graph[0][0] = 1
        for i in range(1, len(cur_speaker_list)):
            inter_cnt, intra_cnt = 1, 1
            total_cnt = 1
            speaker_now = cur_speaker_list[i]
            cur_intra_speaker_graph[i][i] = 1
            j = i-1
            while(j >= 0):
                if cur_speaker_list[j] != speaker_now:
                    cur_inter_speaker_graph[i][j] = 1
                    if total_cnt <= position_upper:  
                        cur_inter_position_index[i][j] = inter_cnt
                else:
                    cur_intra_speaker_graph[i][j] = 1
                    if total_cnt <= position_upper:
                        cur_intra_position_index[i][j] = intra_cnt
                total_cnt += 1
                j -= 1
        inter_speaker_graph[ids] = cur_inter_speaker_graph
        intra_speaker_graph[ids] = cur_intra_speaker_graph
        inter_position_index[ids] = cur_inter_position_index
        intra_position_index[ids] = cur_intra_position_index
    return inter_speaker_graph, intra_speaker_graph

def get_discourse_graph(dataset, totalIds, parsed_dialog, speakers, sentences):
    discourse_graph = {}
    dis_for_distance = {}
    discourse_speaker_mask = {}
    for i in trange(len(totalIds)):
        id = totalIds[i]
        if dataset == 'meld':
            cur_speaker_list = [numpy.argmax(i) for i in speakers[id]]
        else:
            cur_speaker_list = speakers[id]
        cur_graph = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_dis_distance = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_dis_speaker = numpy.zeros((len(cur_speaker_list)))
        if len(cur_speaker_list) == 1:
            cur_graph[0][0] = 1
            cur_dis_speaker[0] = 1
        else:
            cur_relation_lists = parsed_dialog[i]["relations"]
            for j in range(len(cur_relation_lists)):
                cur_graph[cur_relation_lists[j]["x"]][cur_relation_lists[j]["y"]] = 1
                cur_dis_distance[cur_relation_lists[j]["y"]][cur_relation_lists[j]["x"]] = 1
                if cur_speaker_list[cur_relation_lists[j]["x"]] == cur_speaker_list[cur_relation_lists[j]["y"]]:
                    cur_dis_speaker[cur_relation_lists[j]["y"]] = 1
                else:
                    cur_dis_speaker[cur_relation_lists[j]["y"]] = 0
        discourse_graph[id] = cur_graph
        discourse_speaker_mask[id] = cur_dis_speaker
        dis_for_distance[id] = trans_for_distance(cur_dis_distance)
    return discourse_graph, discourse_speaker_mask, dis_for_distance

def count_multi(totalIds, speakers, dataset):
    cnt = 0
    multiIds = []
    for i in trange(len(totalIds)):
        id = totalIds[i]
        speaker_set = set()
        if dataset == 'meld':
            cur_speaker_list = [numpy.argmax(i) for i in speakers[id]]
        else:
            cur_speaker_list = speakers[id]
        for item in cur_speaker_list:
            speaker_set.add(item)
        if len(speaker_set) > 2:
            cnt += 1
            multiIds.append(id)
    return cnt, multiIds

def SRD(data_matrix, start_node):
    '''''
    Dijkstra algorithm
    '''
    vex_num = len(data_matrix)
    flag_list = ['False'] * vex_num
    prev=[0] * vex_num
    dist=['0'] * vex_num
    for i in range(vex_num):
        flag_list[i] = False
        prev[i] = 0
        dist[i] = data_matrix[start_node][i]

    flag_list[start_node] = False
    dist[start_node] = 0
    k = 0
    for i in range(1, vex_num):
        min_value = 99999
        for j in range(vex_num):
            if flag_list[j] == False and dist[j] != float('inf'):
                min_value = dist[j]
                k = j
        flag_list[k] = True
        for j in range(vex_num):
            if data_matrix[k][j] == float('inf'):
                temp = float('inf')
            else:
                temp = min_value + data_matrix[k][j]
            if flag_list[j] == False and temp != float('inf') and temp < dist[j]:
                dist[j] = temp
                prev[j] = k
    return dist

def trans_for_distance(matrix):
    trans_matrix = [[float('inf') for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j]:
                trans_matrix[i][j] = 1
    return trans_matrix

def get_relative_distance(dis_graph, totalIds, speakers):
    intra_relative_distance = {}
    inter_relative_distance = {}
    for i in trange(len(totalIds)):
        id = totalIds[i]
        if dataset == 'meld':
            cur_speaker_list = [numpy.argmax(i) for i in speakers[id]]
        else:
            cur_speaker_list = speakers[id]
        cur_dis_graph = dis_graph[id]
        cur_intra_relative_distance = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_relative_distance = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        
        for i in range(len(cur_speaker_list)):
            relative_dis = SRD(cur_dis_graph, i)
            for j in range(len(relative_dis)):
                if cur_speaker_list[i] == cur_speaker_list[j]:
                    if relative_dis[j] != float('inf'):
                        cur_intra_relative_distance[i][j] = relative_dis[j]
                    else:
                        cur_intra_relative_distance[i][j] = 0
                else:
                    if relative_dis[j] != float('inf'):
                        cur_inter_relative_distance[i][j] = relative_dis[j]
                    else:
                        cur_inter_relative_distance[i][j] = 0
        intra_relative_distance[id] = cur_intra_relative_distance
        inter_relative_distance[id] = cur_inter_relative_distance
    return intra_relative_distance, inter_relative_distance

def get_input_data(dataset='meld'):
    if dataset == 'meld':
        speakers, emotion_labels, sentiment_labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')
        with open('./erc_data/meld/train_parsed.json') as f:
            parsed_train = json.load(f)
        with open('./erc_data/meld/valid_parsed.json') as f:
            parsed_valid = json.load(f)
        with open('./erc_data/meld/test_parsed.json') as f:
            parsed_test = json.load(f)
    else:
        speakers, emotion_labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')
        with open('./erc_data/emorynlp/train_parsed.json') as f:
            parsed_train = json.load(f)
        with open('./erc_data/emorynlp/valid_parsed.json') as f:
            parsed_valid = json.load(f)
        with open('./erc_data/emorynlp/test_parsed.json') as f:
            parsed_test = json.load(f)
    
    if dataset == 'emorynlp':
        totalIds = trainIds.tolist() + validIds.tolist() + testIds.tolist()
    else:
        totalIds = trainIds + validIds + testIds
    
    all_len = speakers.values()
    max_len = max(len(i) for i in all_len)
    print(max_len)
    multi_cnt, multiIds = count_multi(testIds, speakers, dataset)
    print("Ratio of Multi-Party Conversations for " + dataset + " is:", len(multiIds))
    print(multi_cnt / len(testIds))
    total_parsed = parsed_train + parsed_valid + parsed_test
    print("Generating Discourse Graph ...")
    discourse_graph, discourse_speaker_mask, dis_for_distance = get_discourse_graph(dataset, totalIds, total_parsed, speakers, sentences)
    print("Generating Relative Distance Graph ...")
    intra_relative_distance, inter_relative_distance = get_relative_distance(dis_for_distance, totalIds, speakers)
    print("Generating Speaker Graph ...")
    inter_speaker_graph, intra_speaker_graph = get_speaker_graph(dataset, totalIds, speakers)
    
    if dataset == 'meld':
        pickle.dump([speakers, emotion_labels, sentiment_labels, roberta1, roberta2, roberta3, roberta4, discourse_graph, discourse_speaker_mask, \
        inter_speaker_graph, intra_speaker_graph, intra_relative_distance, inter_relative_distance, \
        sentences, trainIds, testIds, validIds, multiIds], open('meld/meld_features_roberta_discourse_new.pkl', 'wb')) 
    else:
        pickle.dump([speakers, emotion_labels, roberta1, roberta2, roberta3, roberta4, discourse_graph, discourse_speaker_mask, \
           inter_speaker_graph, intra_speaker_graph, intra_relative_distance, inter_relative_distance, \
            sentences, trainIds, testIds, validIds, multiIds], open('emorynlp/emorynlp_features_roberta_discourse.pkl', 'wb'))

if __name__ == '__main__':
    for dataset in ['meld', 'emorynlp']:
        get_input_data(dataset)




