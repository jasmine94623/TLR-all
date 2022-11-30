
# def Doc2Vec_Similarity(data):
#     train_data= (data["is_duplicate"] == 1)

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import nltk
from scipy import spatial
# measure similarity
# https://stackoverflow.com/questions/53503049/measure-similarity-between-two-documents-using-doc2vec/53506326#53506326


def Doc2Vec_Similarity(data):
    nltk.download('stopwords')
    nltk_stopwords = nltk.corpus.stopwords.words('english')

    # data = pd.read_csv("Doc2VecData.csv")
    theme_dict = {}
    question_dict = {}
    is_exist = set()
    theme_cnt = -1

    for idx, row in data.iterrows():
        id, qid1, qid2, q1, q2, is_duplicate = row
        if qid1 not in theme_dict:
            theme_cnt += 1
            theme_dict[qid1] = theme_cnt
        if is_duplicate == 1:
            theme_dict[qid2] = theme_dict[qid1]

    tagged_data = []
    for idx, row in data.iterrows():
        id, qid1, qid2, q1, q2, is_duplicate = row
        if id == 1:
            break
        word_list = []
        q_list = q2.lower().split()
        for word in q_list:
            if word in nltk_stopwords:
                continue
            word_list.append(word)

        tagged_data.append(TaggedDocument(
            words=word_list, tags=str(theme_dict[qid2])))

    d2v_model = Doc2Vec(vector_size=30, min_count=2, epochs=10)
    d2v_model.build_vocab(tagged_data)
    d2v_model.train(
        tagged_data, total_examples=d2v_model.corpus_count, epochs=10)

    des_sim = {}
    for idx, row in data.iterrows():
        id, qid1, qid2, q1, q2, is_duplicate = row
        vec1 = []
        vec2 = []
        q1_list = q1.lower().split()
        q2_list = q2.lower().split()
        for word in q1_list:
            if word in nltk_stopwords:
                continue
            vec1.append(word)

        for word in q2_list:
            if word in nltk_stopwords:
                continue
            vec2.append(word)
        vec1 = d2v_model.infer_vector(vec1)
        vec2 = d2v_model.infer_vector(vec2)
        cos_distance = spatial.distance.cosine(vec1, vec2)
        des_sim['POI_'+str(qid1), 'POI_'+str(qid2)] = cos_distance

    # print(des_sim)
    return des_sim
