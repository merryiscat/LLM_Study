import pandas as pd
import os, sys
sys.path.append('/workspace')
from ml.embedding import EmbeddingModel
from tqdm import tqdm
import numpy as np

embedding_model = EmbeddingModel.create('intfloat/multilingual-e5-large')

class IntentClassifier:
    def __init__(self, train_dataset_csv_path):
        self.train_dataset_csv_path = train_dataset_csv_path
        self.embedding_result = self.train_knn(self.train_dataset_csv_path)
    
    # make train dataset
    def train_knn(self, train_dataset_csv_path):
        try:
            intent_df = pd.read_csv(train_dataset_csv_path)
        except:
            intent_df = pd.read_csv(train_dataset_csv_path, encoding='latin1')
        embedding_result = []
        for idx, row in tqdm(intent_df.iterrows()):
            intent = row['question']
            cls = row['class']
            vector = embedding_model.embedding_one(intent)[0].tolist()
            embedding_result.append((vector, cls))
        return embedding_result
    
    # euclidian dist
    def euc_distance(self, vector_1, vector_2):
        distance = np.linalg.norm(np.array(vector_1) - np.array(vector_2))
        return distance
    
    # calc_all_vector
    def all_vector_distance(self, query_vector, intents):
        distance_list = []
        for intent in tqdm(intents):
            distance = self.euc_distance(query_vector, intent[0])
            distance_list.append((distance, intent[1]))
        return distance_list
    
    # knn
    def knn(self, query_vector, intents, k):
        distance_list = self.all_vector_distance(query_vector, intents)
        sorted_list = sorted(distance_list, key=lambda x:x[0])
        knn_list = sorted_list[:k]
        knn_dict = dict()
        for comp in knn_list:
            if comp[1] in knn_dict:
                knn_dict[comp[1]] += 1
            else:
                knn_dict[comp[1]] = 1
        sorted_knn = sorted(knn_dict.items(), key=lambda x:x[1], reverse=True)
        return sorted_knn[0][0], sorted_knn
    
    # intent classifier
    def intent_classifier(self, query_text, k=10):
        intents = self.embedding_result
        query_embedding = embedding_model.embedding_one(query_text)[0].tolist()
        cls, sorted_knn = self.knn(query_embedding, intents, k)
        return cls, sorted_knn

# cls, sorted_knn = intent_classifier("태풍 피해는?")
# print(f'모델이 분류한 의도 : {cls}')
# print(f'상세 의도 분류 내용 : {sorted_knn}')
# print(f'\n------------------------------\n참고 - 의도 :::\n 0 : 문서 검색, 존재 여부 \n 1 : 문서 개수 반환 \n 2 : 문서 내용 질의')