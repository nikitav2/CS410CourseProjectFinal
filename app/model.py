import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import pickle
import spacy




# We will run this file to create the model and load it
resumeDataSet = pd.read_csv('app/data/UpdatedResumeDataSet.csv' ,encoding='utf-8')
jobDescriptionDataSet = pd.read_csv('app/data/JD_data.csv', encoding='utf-8')
RFModel = None
RFtdif = None
RFle = None
MBModel = None
MBtdif = None
MBle = None
KneighModel = None
Kneightdif = None
Kneighle = None
nlp = spacy.load("en_core_web_lg")


def clean_resume(resume_text: str) -> str:
    """
    Cleans non alphanumeric chars and punctuation from resume tesxt

    Args:
        resume_text (str): stringified version of resume
    
    Return:
        resume_text (str): cleaned version of resume

    """
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#S+', '', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]•^_`{|}~"""), ' ', resume_text)  # remove punctuations
    # resume_text = resume_text.replace('\n', ' ')
    # resume_text = resume_text.replace('\t', ' ')
    # resume_text = re.sub(' +', ' ', resume_text)
    return resume_text

def RandomForestClassifierJobDescription(resumeDataSet):
    global RFModel
    global RFtdif
    global RFle
    resumeDataSet['cleaned_resume'] = resumeDataSet.description.apply(lambda x: clean_resume(x))
    resumeDataSet['cleaned_job_title'] = resumeDataSet.job.apply(lambda x: clean_resume(x))
    print(resumeDataSet.head())
    print((resumeDataSet['cleaned_job_title']))
    var_mod = ['cleaned_job_title']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
    var_mod = None
    requiredText = resumeDataSet['cleaned_resume'].values
    requiredTarget = resumeDataSet['job'].values
    print(requiredTarget)
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    print(WordFeatures)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("this is x test", X_test)
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("n Classification report for classifier %s:n%sn" % (clf, metrics.classification_report(y_test, prediction)))
    filename = 'finalized_model_RF.sav'
    filename1 = 'tfidfs_RF.pkl'
    RFModel = filename
    RFtdif = filename1
    RFle = le
    print(type(word_vectorizer))
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(word_vectorizer, open(filename1, 'wb'))

def MultinomialClassifierResume(resumeDataSet):
    global MBModel
    global MBtdif
    global MBle
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: clean_resume(x))
    # print(resumeDataSet.head())
    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
    requiredText = resumeDataSet['cleaned_resume'].values
    print(type(requiredText[0]))
    requiredTarget = resumeDataSet['Category'].values
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
    clf = OneVsRestClassifier(MultinomialNB())
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print(type(X_test))
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("n Classification report for classifier %s:n%sn" % (clf, metrics.classification_report(y_test, prediction)))
    print(le.classes_)
    filename = 'finalized_model_multinomial.sav'
    filename1 = 'tfidf_multinomial.pkl'
    MBModel = filename
    MBtdif = filename1
    MBle = le
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(word_vectorizer, open(filename1, 'wb'))


def KNeighborsClassifierResume(resumeDataSet):
    print("Initializing KNN classifier...")
    global KneighModel
    global Kneightdif
    global Kneighle
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: clean_resume(x))
    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
    requiredText = resumeDataSet['cleaned_resume'].values
    print(type(requiredText[0]))
    requiredTarget = resumeDataSet['Category'].values
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print(type(X_test))
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("n Classification report for classifier %s:n%sn" % (clf, metrics.classification_report(y_test, prediction)))
    print(le.classes_)
    filename = 'finalized_model_kneighbors.sav'
    filename1 = 'tfidf_kneighbors.pkl'
    KneighModel = filename
    Kneightdif = filename1
    Kneighle = le
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(word_vectorizer, open(filename1, 'wb'))



def predict_job_title_knn(resume_str: str):
    global KneighModel
    global Kneightdif
    global Kneighle
    if KneighModel == None or Kneightdif == None or Kneighle == None:
        # load classifiers and store in file
        KNeighborsClassifierResume(resumeDataSet)
    
    #classify resume string
    loaded_knnmodel = pickle.load(open(KneighModel, 'rb'))
    loaded_knntdif = pickle.load(open(Kneightdif, 'rb'))
    unseen_df1 = pd.DataFrame({'text':[resume_str]})
    X_unseen1 = loaded_knntdif.transform(unseen_df1['text']).toarray()


    # Predict on `X_unseen`
    y_pred_unseen = loaded_knnmodel.predict(X_unseen1)
    job_prediction = Kneighle.classes_[y_pred_unseen][0]
    print("Prediction " + job_prediction)
    return job_prediction

def predict_job_title_mb(resume_str: str):
    global MBModel
    global MBtdif
    global MBle
    if MBModel == None or MBtdif == None or MBle == None:
        # load classifiers and store in file
        MultinomialClassifierResume(resumeDataSet)
    
    #classify resume string
    loaded_mbmodel = pickle.load(open(MBModel, 'rb'))
    loaded_mbtdif = pickle.load(open(MBtdif, 'rb'))
    unseen_df1 = pd.DataFrame({'text':[resume_str]})
    X_unseen1 = loaded_mbtdif.transform(unseen_df1['text']).toarray()


    # Predict on `X_unseen`
    y_pred_unseen = loaded_mbmodel.predict(X_unseen1)
    job_prediction = Kneighle.classes_[y_pred_unseen][0]
    print("Prediction " + job_prediction)
    return job_prediction



def reccomend_jobs(job_prediction: str) -> list[dict]:
    """
    Returns a list of dictionarys containing specific job title (python webdev), job description

    Args:
        job_prediction (str): job prediciton of user's resume
    Return:

    """
    
    job_lst = []
    similar = find_unique_jobs(job_prediction, k=3)
    
    for sim, job in similar:
        reccomendations =  jobDescriptionDataSet.loc[jobDescriptionDataSet['job'] == job]
        reccomendations = reccomendations.to_dict('records')
        job_lst.extend(reccomendations)
    return job_lst


    

def find_unique_jobs(job_prediction: str, k: int) -> list[tuple]:
    """
    Returns a list of the k most similar job descriptions to job_prediction
    """
    global nlp
    unique_jobs = jobDescriptionDataSet['job'].unique()

    # sort unique_jobs by similarity with predicted job
    word_similarities = []

    predict_vec = nlp(job_prediction)
    for job in unique_jobs:
        job_vec = nlp(job)
        similarity = predict_vec.similarity(job_vec)
        word_similarities.append((similarity, job))
    
    word_similarities.sort(reverse=True)
    return word_similarities[:k]

def classify_job(job_str: str) -> str:
    
    global RFModel
    global RFtdif
    global RFle
    if RFModel == None or RFtdif == None or RFle == None:
        # load classifiers and store in file
        RandomForestClassifierJobDescription(jobDescriptionDataSet)
    
    #classify resume string
    loaded_RFmodel = pickle.load(open(RFModel, 'rb'))
    loaded_RFtdif = pickle.load(open(RFtdif, 'rb'))
    unseen_df1 = pd.DataFrame({'text':[job_str]})
    X_unseen1 = loaded_RFtdif.transform(unseen_df1['text']).toarray()


    # Predict on `X_unseen`
    y_pred_unseen = loaded_RFmodel.predict(X_unseen1)
    job_prediction = y_pred_unseen
    print("Prediction " + job_prediction)
    return job_prediction