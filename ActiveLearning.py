from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import numpy as np

from IPython.display import clear_output

import os
import time
import numpy as np
import pandas as pd

from string import punctuation
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report, \
accuracy_score, precision_score, recall_score, fbeta_score, make_scorer, log_loss, classification_report

import sys
 
from text_preprocessing.text_preprocessing import RemoveWordsTransform, CleanTransform, LemmatizeTransform, StandardizeTransform


import matplotlib.pyplot as plt
from scipy import stats
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from sklearn import linear_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import BaggingClassifier
from scipy.sparse import csc_matrix, vstack
from scipy.stats import entropy
from collections import Counter
import copy
from multiprocessing import Pool
from sklearn.metrics import average_precision_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csc_matrix, vstack
from scipy.stats import entropy
from collections import Counter
from active_learning import ActiveLearner
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.linear_model import RidgeCV
from sklearn import neighbors
from IPython.display import clear_output



import logging
import math

print(__doc__)

from scipy.spatial import distance
import scipy 
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_validate


#from sklearn.metrics import pairwise_distance
from sklearn.metrics.pairwise import pairwise_distances


import copy

from sklearn.model_selection import StratifiedKFold

class AL(object):
    


    def __init__(self, 
                 strategy,strategy_parameters,
                 X,y,
                 Unlabeled_pool,Unlabeled_views,
                 Labeled_pool,Labeled_views,
                 index_unlabeled_pool,index_labeled_pool,
                 y_labeled,y_unlabeled,
                 index_base_classifier,
                 pipelines_grid,
                 batch_size,
                 n_iter,
                 n_label,
                 param_grid,
                 metric_grid,
                 GridSearchCV):
        
        self.X = X
        self.y = y
        self.Unlabeled_pool = Unlabeled_pool
        self.Unlabeled_views = Unlabeled_views
        
        self.Labeled_pool = Labeled_pool
        self.Labeled_views = Labeled_views
                
        self.y_labeled = y_labeled
        self.y_unlabeled = y_unlabeled
        
        self.strategy = strategy
        
        self.strategy_parameters = strategy_parameters
                
        self.index_base_classifier = index_base_classifier
        
        self.index_unlabeled_pool = index_unlabeled_pool
        
        self.index_labeled_pool = index_labeled_pool
        
        self.pipelines_grid = pipelines_grid
        
        self.models = []
        
        self.classifiers = []
        
        self.trained_models_view_1 = []
        self.trained_models_view_2 = []

        self.batch_size = batch_size

        self.n_iter = n_iter
        
        self.n_label = n_label
        
        self.param_grid = param_grid
        
        self.metric_grid = metric_grid
        
        self.GridSearchCV = GridSearchCV 
        
        self.HC = None
        
        self.HC_pool = []
        
        stopwords_gensim = []
        stopwords_nltk = stopwords.words('english')
        for i in STOPWORDS:
            stopwords_gensim.append(i)
        self.nltk_gensim_stop_words = np.union1d(stopwords_nltk,stopwords_gensim)
        self.nltk_gensim_stop_words = self.nltk_gensim_stop_words.tolist()
        
    
    def initialization(self,strategy='exploration'):
        """
        Select the intial samples in the ctive learning process
        Parameters
         ----------
         n_cv : String
             The strategy to select the initial samples.
        """
        
        if(strategy=='random'):
            selected_indices = np.random.choice(len(self.X),self.batch_size, replace=False)
            return selected_indices
            
        elif(strategy=='exploration'):
            X= self.Compute_tf_idf(self.X)
            selected_indices = self.fft(X,distance.cdist(X, X, 'hamming'),self.batch_size)
            return selected_indices
            
    
    
    def make_query(self,strategy=None,strategy_args=None):
        
   
        if(strategy_args==None):
            strategy_parameters = self.strategy_parameters
        else:
            strategy_parameters = strategy_args
        
        if(strategy==None):
            
            strategy = self.strategy
            
        selected_index = []
        
        if(strategy!='HierarchicalClustering' and len(self.index_labeled_pool)==0):
            selected_index = self.initialization()
            return selected_index
        print(strategy)
            
        if(strategy=='CoTesting'):
            
            disagreement = strategy_parameters['disagreement']
            
            selectQuery = strategy_parameters['selectQuery']
        
            selected_index = self.Co_Testing(disagreement, selectQuery)
            
        
        elif(strategy=='HierarchicalClustering'):
            
            if(self.HC == None):
                
                self.HC_pool = self.Compute_tf_idf(self.X)
                
                self.HC = HierarchicalClusterAL(self.HC_pool,self.X, self.y,self.index_labeled_pool, self.index_unlabeled_pool, 12345)
        
            selected_index = self.HierarchicalClustering()
        
        
        elif(strategy=='QuerybyCommittee'):
            
            disagreement = strategy_parameters['disagreement']
                    
            selected_index = self.QueryByCommittee(disagreement)
        
        
        elif(strategy=='ExpectedModelChange'):
            
                    
            selected_index = self.Expected_Model_Change()
            
        
        elif(strategy=='WeightedExpectedModelChange'):
            
                    
            selected_index = self.Weighted_Expected_Model_Change()
            
            
        elif(strategy=='ExpectedErrorReduction'):
            
            loss = strategy_parameters['loss']
                    
            selected_index = self.Expected_Error_Reduction(loss)
            
            
        elif(strategy=='UncertaintySampling'):
            
            heuristique = strategy_parameters['heuristique']
                    
            selected_index = self.uncertainty_sampling(heuristique)
            
        
        elif(strategy=='UncertaintyClustering'):
                    
            selected_index = self.Uncertainty_Clustering()
            
        
        elif(strategy=='DiversityClustering'):
            
            dist = strategy_parameters['dist']
            
            beta = strategy_parameters['beta']
                    
            selected_index = self.Diversity_Clustering(dist,beta)
         
        
        elif(strategy=='QuerybyDiversity'):
            
            dist = strategy_parameters['dist']
            
            beta = strategy_parameters['beta']
                    
            selected_index = self.Query_by_Diversity(dist,beta)
            
        elif(strategy=='QuerybyRandom'):
            
            print('hhhhhhhhhhhh')
            
            selected_index = self.QuerybyRandom()
            
        """elif(strategy=='QuerybyBagging'):
            
            n_bags = strategy_parameters['n_bags']
            
            method = strategy_parameters['method']
                    
            selected_index = self.query_by_bagging(n_bags,method)"""
        
        
     

        return selected_index
    
    
    def Compute_tf_idf(self,X) :
        vectorizer = TfidfVectorizer(max_df=0.5,#0.7 700
                                     min_df=3,
                                     norm='l2',                                
                                     stop_words=self.nltk_gensim_stop_words,
                                     use_idf=True,
                                     ngram_range=(1,3),
                                     max_features=500
                                    )
        y = vectorizer.fit_transform(X)
        normalizer = Normalizer(copy=False)
        scaler = StandardScaler(with_mean=False)
        lsa = make_pipeline(normalizer,scaler)
        y = lsa.fit_transform(y)
        return np.array(y.toarray())
    
    
    def Compute_tf_idf_fit(self,X) :
        vectorizer = TfidfVectorizer(max_df=0.5,#0.5 0.6
                                     min_df=2,
                                     norm='l2',                                
                                     stop_words=self.nltk_gensim_stop_words,
                                     use_idf=True,
                                     ngram_range=(1,3),
                                     max_features=500
                                    )
        y = vectorizer.fit(X)
        return y


    def Compute_tf_idf_transform(self,vectorizer,X) :
        y = vectorizer.transform(X)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(normalizer)
        y = lsa.fit_transform(y)
        return np.array(y.toarray())
    

    def Compute_tf_idf_fit_view2(self,X) :
        vectorizer = TfidfVectorizer(max_df=0.5,#0.5 0.6
                                     min_df=2,
                                     norm='l2',
                                     stop_words=self.nltk_gensim_stop_words,
                                     use_idf=True,
                                     ngram_range=(1,3),
                                    max_features=600#500
                                    )
        y = vectorizer.fit(X)
        return y

    def Compute_tf_idf_fit_view3(self,X) :
        vectorizer = TfidfVectorizer(max_df=0.5,#0.5 0.7# 0.4
                                     min_df=2,
                                     norm='l2',
                                     stop_words=self.nltk_gensim_stop_words,
                                     use_idf=True,
                                     ngram_range=(1,3),
                                      max_features=600
                                    )
        y = vectorizer.fit(X)
        return y


    def Compute_tf_idf_fit_view4(self,X) :
        vectorizer = TfidfVectorizer(max_df=0.5,#0.5 0.5
                                     min_df=3, 
                                     norm='l2',
                                     stop_words=self.nltk_gensim_stop_words,
                                     use_idf=True,
                                     ngram_range=(1,3),
                                     #max_features=500
                                    )
        y = vectorizer.fit(X)
        return y

    
    def QuerybyRandom(self):
        return np.random.choice(len(self.Unlabeled_pool),self.batch_size, replace=False)
  
    def HierarchicalClustering(self):
        
        """This method call the hierarchical clustering algorithm. 
        the source code is from the google repo of active learning"""
        
        kwargs = dict()
        kwargs["labeled"]: dict(zip(self.index_labeled, self.y[self.index_labeled]))
        kwargs["already_selected"] = self.HC.index_train.astype(int)
        selected_indexes = self.HC.select_batch_(self.batch_size, self.HC.index_train,self.HC.index_train,
                                                 self.HC.y[self.HC.index_train.astype(int)])
        self.HC.update_index(selected_indexes)
        
        selected_indices = np.where(np.isin(self.index_unlabeled_pool,selected_indexes,))
        return selected_indices
    
    def Query_by_Diversity(self,dist,beta) :
   
        Diversity = np.array([])
        y = self.Compute_tf_idf(self.Unlabeled_pool) 
        X = self.Compute_tf_idf(self.Labeled_pool)

        similarity =  distance.cdist(y, X, dist) #cosine_similarity(y,X)
        
        for i in range(len(similarity)) :
            Diversity = np.append(Diversity,        
                            np.max([item for item in similarity[i]]))        

        selected_indices =np.argsort(Diversity)[::-1]
        return selected_indices[:self.batch_size]

    
    def Uncertainty_Clustering(self):
        # Probably okay to always use MiniBatchKMeans
        # Should standardize data before clustering
        # Can cluster on standardized data but train on raw features if desired
        try:
          test_distances = self.classifiers[0].decision_function(self.Unlabeled_pool)
          train_distances = self.classifiers[0].decision_function(self.Labeled_pool)

        except:
          test_distances = self.classifiers[0].predict_proba(self.Unlabeled_pool)
          train_distances = self.classifiers[0].predict_proba(self.Labeled_pool)

        if len(test_distances.shape) < 2:
          test_min_margin = np.abs(test_distances)
          train_min_margin = np.abs(train_distances)

        else:
          test_sort_distances = np.sort(test_distances, 1)[:, -2:]
          train_sort_distances = np.sort(train_distances, 1)[:, -2:]

          test_min_margin = test_sort_distances[:, 1] - test_sort_distances[:, 0]
          test_min_margin = 1 - np.amax(test_distances, axis=1)  
          train_min_margin = train_sort_distances[:, 1] - train_sort_distances[:, 0]
          train_min_margin = 1 - np.amax(train_distances, axis=1)

        rank_ind = np.argsort(-test_min_margin)
        #rank_ind = [i for i in rank_ind if i not in already_selected]

        test_distances = np.abs(test_distances)
        train_distances = np.abs(train_distances)


        min_margin_by_class = np.min(-train_min_margin,axis=0)

        unlabeled_in_margin = np.array([i for i in range(len(self.Unlabeled_pool)) if
                                test_distances[i]<min_margin_by_class ])
    
        if (len(unlabeled_in_margin) < self.batch_size):
          print("Not enough points within margin of classifier, using simple uncertainty sampling")
          return rank_ind[:self.batch_size]

        clustering_model = MiniBatchKMeans(n_clusters=self.batch_size,init='k-means++')
        X= self.Compute_tf_idf(self.Unlabeled_pool)
        dist_to_centroid = clustering_model.fit_transform(X[unlabeled_in_margin])
        medoids = np.argmin(dist_to_centroid,axis=0)
        medoids = list(set(medoids))
        selected_indices = unlabeled_in_margin[medoids]
        selected_indices = sorted(selected_indices,key=lambda x: -test_min_margin[x])
        remaining = [i for i in rank_ind if i not in selected_indices]

        return selected_indices
    
    
    def Diversity_Clustering(self,dist,beta):
        
        y = self.Compute_tf_idf(self.Unlabeled_pool) 
        X = self.Compute_tf_idf(self.Labeled_pool)

        similarity =  distance.cdist(y,X, dist) 
        Diversity = np.array([])
        for i in range(len(similarity)) :
            Diversity = np.append(Diversity,        
                            np.max([item for item in similarity[i]]))        


        index_Diversity = np.argsort(Diversity)[::-1]
        print(int(len(Diversity)*beta))
        sorted_index_Diversity = index_Diversity[:(int(len(Diversity)*beta))+1]
        filtered_Diversity = Diversity[sorted_index_Diversity]

        clustering_model = MiniBatchKMeans(n_clusters=self.batch_size,init='k-means++')
        X= self.Compute_tf_idf(self.Unlabeled_pool)
        dist_to_centroid = clustering_model.fit_transform(X[sorted_index_Diversity])
        medoids = np.argmin(dist_to_centroid,axis=0)
        medoids = list(set(medoids))
        selected_indices = sorted_index_Diversity[medoids]
        #selected_indices = sorted(selected_indices,key=lambda x: Diversity[x],reverse=True)
        
        return selected_indices
    
    
    def uncertainty_sampling(self, heuristique):
        from scipy.stats import entropy
        probabilities = self.classifiers[0].predict_proba(self.Unlabeled_pool)
        
        score = []
        if heuristique == 'least_confident':
            score = 1 - np.amax(probabilities, axis=1)

        elif heuristique == 'max_margin':
            margin = np.partition(-probabilities, 1, axis=1)
            score = -np.abs(margin[:,0] - margin[:, 1])

        elif heuristique == 'entropy':
            score =  np.apply_along_axis(entropy, 1, probabilities)
        
      
        scores = np.array([])
        for s in score:
            scores = np.append(scores,s*-1)
        selected_indices = np.argsort(scores)[:self.batch_size]
        
        return selected_indices
    
    def query_by_bagging(n_bags=3, method="entropy"):
        """
        :param base_model: Model that will be  **fitted every iteration**
        :param n_bags: Number of bags on which train n_bags models
        :param method: 'entropy' or 'KL'
        :return:
        """
        eps = 0.0000001
        vectorizer = self.Compute_tf_idf_fit(self.Labeled_pool)
        Labeled_pool = self.Compute_tf_idf_transform(vectorizer,self.Labeled_pool)
        Unlabeled_pool = self.Compute_tf_idf_transform(vectorizer,self.Unlabeled_pool)
        clfs = BaggingClassifier(self.models[0], n_estimators=n_bags,bootstrap=False)
        clfs = clfs.fit(Labeled_pool, self.y_labeled)
        self.classifiers = copy.deepcopy(np.array(clfs))
        pc = clfs.predict_proba(Unlabeled_pool)
        selected_indices = []
        if method == 'entropy':
            pc += eps
            fitness = np.sum(pc * np.log(pc), axis=1)
            selected_indices =  np.argsort(fitness)[:batch_size]
        elif method == 'KL':
            p = np.array([clf.predict_proba(Unlabeled_pool)for clf in clfs.estimators_])
            fitness = np.mean(np.sum(p * np.log(p / pc), axis=2), axis=0)
            selected_indices = np.argsort(fitness)[-batch_size:]

        return selected_indices #, fitness/np.max(fitness)
    
    def Expected_Error_Reduction(self,loss):

        classes = np.unique(self.y_unlabeled)
        n_classes = len(classes)

        probabilities = self.classifiers[0].predict_proba(self.Unlabeled_pool)

        scores = []
        for i, x in enumerate(self.Unlabeled_pool):
            score = []
            #X=np.vstack((X_train_final, [x]))
            X=np.append(self.Labeled_pool, [x])
            for yi in range(n_classes):
                m = copy.deepcopy(self.classifiers[0])
                m.fit(X, np.append(self.y_labeled,yi ))
                p = m.predict_proba(self.Unlabeled_pool)

                if loss == '01':  # 0/1 loss
                    score.append(probabilities[i, yi] * np.sum(1-np.max(p, axis=1)))
                elif loss == 'log': # log loss
                    score.append(probabilities[i, yi] * -np.sum(p * np.log(p)))
            scores.append(np.sum(score))

        selected_indices = np.argsort(scores)[:self.batch_size]

        return selected_indices
            
    def Weighted_Expected_Model_Change(self):

        probabilities = self.classifiers[0].predict_proba(self.Unlabeled_pool)
        predictions = self.classifiers[0].predict(self.Unlabeled_pool)

        scores = []
        for i, x in enumerate(self.Unlabeled_pool):
            #X=np.vstack((self.Labeled_pool, [x]))
            X=np.append(self.Labeled_pool, [x])
            m = copy.deepcopy(self.classifiers[0])
            m = m.fit(X, np.append(self.y_labeled,predictions[i]))
            new_probabilities = m.predict_proba(self.Unlabeled_pool)
            new_predictions = m.predict(self.Unlabeled_pool)

            label_change = np.abs(new_predictions - predictions)
            prediction_change = np.abs(np.apply_along_axis(np.max, 1, new_probabilities) - np.apply_along_axis(np.max, 1,probabilities))

            score=0
            sum_change = []
            sum_change = np.array([a*b for a,b in zip(label_change,prediction_change)])

            score = np.sum(sum_change)          
            scores.append(score)

        selected_indices = np.argsort(scores)[::-1][:self.batch_size]

        return selected_indices
    
    
    def Expected_Model_Change(self):

        predictions = self.classifiers[0].predict(self.Unlabeled_pool)

        scores = []
        for i, x in enumerate(self.Unlabeled_pool):
            #X=np.vstack((self.Labeled_pool, [x]))
            X=np.append(self.Labeled_pool, [x])
            m = copy.deepcopy(self.classifiers[0])
            m = m.fit(X, np.append(self.y_labeled,predictions[i]))
            new_predictions = m.predict(self.Unlabeled_pool)

            label_change = np.abs(new_predictions - predictions)
            score=0

            score = np.sum(label_change)          
            scores.append(score)

        selected_indices = np.argsort(scores)[::-1][:self.batch_size]

        return selected_indices
        
    def Co_Testing(self, disagreement, strategy):
    
        ask_idx = []
        
        # Let the trained students provide their vote for unlabeled data
        votes = np.zeros((len(self.Unlabeled_pool), 
                              len(self.classifiers[0:])))
        
        # Let the trained students provide their confidence of the vote for unlabeled data    
        proba = np.zeros((len(self.Unlabeled_pool), 
                              len(self.classifiers[0:])))

        for i, classifier in enumerate(self.classifiers):
                votes[:, i] = classifier.predict(self.Unlabeled_views[i])
                proba[:, i] = (np.abs(classifier.decision_function(self.Unlabeled_views[i])))

        contention_points = []   
        
        if disagreement == 'vote':
            vote_entropy = self.vote_disagreement(votes,len(self.classifiers))
            ask_idx =  np.argsort(vote_entropy)[::-1]
            max_disagreement = np.max(vote_entropy)
            temp = np.where(vote_entropy==max_disagreement)
            for v in temp :
                contention_points = np.append(contention_points,v).astype(int)
        elif disagreement == 'kl_divergence':
            proba = np.array(proba).transpose(1, 0, 2).astype(float)
            avg_kl = self.kl_divergence_disagreement(proba)
            ask_idx = np.argsort(avg_kl)[::-1]
            max_disagreement = np.max(proba)
            temp = np.where(vote_entropy==max_disagreement)
            for v in temp :
                contention_points = np.append(contention_points,v).astype(int)
        
        if len(contention_points) < self.batch_size:
            print("Not enough contention points, using simple max disagreement")
            return ask_idx[:self.batch_size]
            
        #♦
            
        ask_idx = self.QueryStrategy(vote_entropy,proba,strategy,contention_points,self.batch_size)
        return ask_idx[:self.batch_size]
    
    def QueryByCommittee(self, disagreement):
    
        ask_idx = []
        
        # Let the trained students provide their vote for unlabeled data
        votes = np.zeros((len(self.Unlabeled_pool), 
                              len(self.classifiers[0:])))
        
        # Let the trained students provide their confidence of the vote for unlabeled data    
        proba = np.zeros((len(self.Unlabeled_pool), 
                              len(self.classifiers[0:])))

        for i, classifier in enumerate(self.classifiers):
                votes[:, i] = classifier.predict(self.Unlabeled_pool)
                proba[:, i] = (np.abs(classifier.decision_function(self.Unlabeled_pool)))

        contention_points = []   
        
        if disagreement == 'vote':
            vote_entropy = self.vote_disagreement(votes,len(self.classifiers))
            ask_idx =  np.argsort(vote_entropy)[::-1]
        elif disagreement == 'kl_divergence':
            proba = np.array(proba).transpose(1, 0, 2).astype(float)
            avg_kl = self.kl_divergence_disagreement(proba)
            ask_idx = np.argsort(avg_kl)[::-1]
        
        return ask_idx[:self.batch_size]
                 
    def MultiView_train(self,n_cv,view):
        """
        Train each classifier among his coresspondent view
        """ 
        self.tune_parameters(n_cv)
        classifiers = []
        models_copy = copy.deepcopy(self.models)
        Multi_views = view#self.Labeled_views
        for i, model in enumerate(models_copy):
            classifiers.append(model.fit(Multi_views[i], self.y_labeled).best_estimator_)  
        self.classifiers = copy.deepcopy(classifiers)
 
    def SingleView_train(self,n_cv,view):
        """
        Train each classifier among his coresspondent view
        """ 
        self.tune_parameters(n_cv)
        classifiers = []
        models_copy = copy.deepcopy(self.models)
        Single_view = view#self.Labeled_pool
        for i, model in enumerate(models_copy):
            classifiers.append(model.fit(Single_view, self.y_labeled).best_estimator_)  
        self.classifiers = copy.deepcopy(classifiers)
        
    
    def train(self,n_cv):
        
        if(self.strategy=='CoTesting'):
            self.MultiView_train(n_cv,self.Labeled_views)
        elif(self.strategy!='QuerybyBagging') :
            self.SingleView_train(n_cv,self.Labeled_pool)
        
        """(len(self.classifiers)==0):
            self.tune_parameters(n_cv)
            clfs = BaggingClassifier(self.models[0], n_estimators=5,bootstrap=False)
            print(self.Labeled_pool.shape)
            vectorizer = self.Compute_tf_idf_fit(np.array(self.Labeled_pool))
            Labeled_pool = np.array(Compute_tf_idf_transform(vectorizer,self.Labeled_pool))
            Unlabeled_pool = self.Compute_tf_idf_transform(vectorizer,self.Unlabeled_pool)
            print(Labeled_pool.shape)
            clfs = clfs.fit(Labeled_pool, self.y_labeled)
            self.classifiers = copy.deepcopy(np.array(clfs))"""
        
    def tune_parameters(self,n_cv):
        """
        Use the Grid search to tune the hyperparameters of the
        model and eventually the pre-processing parameters
        Parameters
         ----------
         n_cv : integer
             The number of folds.
        """
        from sklearn.model_selection import StratifiedKFold
        self.models = []
        for i  in range(len(self.pipelines_grid)):
            self.models.append(self.GridSearchCV(self.pipelines_grid[i],self.param_grid[i], scoring=self.metric_grid[i], 
                                n_jobs=1,
                                cv=  StratifiedKFold(n_splits=n_cv),
                                refit=True)
            )      
            
    def predict(self,X):
        """
        Make the actual predictions
        """
        
        return  self.classifiers[self.index_base_classifier].predict(X)
    
    def Co_Testing_predict(self,Unlabeled_views,strategy):
        """
        Implement the CreateOutputHypothesis in the CoTestin paradigm
        Parameters
        ----------
        strategy : String
             The chosen strategy to perform the actual predictions.
        """
        
        votes = np.zeros((len(Unlabeled_views[0]), 
                              len(self.classifiers)))
        
        for i, classifier in enumerate(self.classifiers):
                votes[:, i] = classifier.predict(Unlabeled_views[i]).astype(int)
                
        votes = votes.astype(int)
        preds = np.array([])
        
        if(strategy=='majority'):
            
            
            preds = np.apply_along_axis(np.argmax,0,np.apply_along_axis(np.bincount, 0, votes).astype(int))
            
        elif(strategy=='logical_and'):
            
            preds = np.apply_along_axis(np.all, 1, votes).astype(int)
            
        elif(strategy=='logical_or'):
        
            preds = np.apply_along_axis(np.any, 1, votes).astype(int)

        return preds
        
        

    def update(self,selected_index):
        """
        Update the unlabeled and labeled pool
        Parameters
        ----------
        selected_index : array of integer, shape==(n)
             The index of the selected samples by the active learner.
        """
        # add index of selected samples to index_labeled
        self.index_labeled_pool = np.append(self.index_labeled_pool,self.index_unlabeled_pool[selected_index])
        # delete the index of selected samples from index_unlabeled
        self.index_unlabeled_pool = np.delete(self.index_unlabeled_pool,selected_index)
        # add the selected samples to the pool of selected samples
        #self.Labeled_pool =  np.vstack((self.Labeled_pool,self.Unlabeled_pool[selected_index]))
        
        
        """if the labels are not prvided, please uncomment the below code line and just comment the next one"""
        
        #self.Labeled_pool =  np.append(self.Labeled_pool,self.get_label(selected_index))
        
        """if above code line is uncomment, please comment the below code line"""
        
        self.Labeled_pool =  np.append(self.Labeled_pool,self.Unlabeled_pool[selected_index])
        
    
        # delete the selected samples from the pool of unlabeled data 
        #self.Unlabeled_pool =  delete_from_csr(self.Unlabeled_pool, selected_index, [])
        self.Unlabeled_pool =  np.delete(self.Unlabeled_pool, selected_index)
        
        # update the the view of the labeled and Unlabeled pool
        for i in range(len(self.Unlabeled_views)) :
            if (len(self.Unlabeled_views[i].shape) > 1) :
                self.Labeled_views[i] = np.vstack((self.Labeled_views[i],self.Unlabeled_views[i][selected_index]))
                self.Unlabeled_views[i]= self.delete_from_csr(self.Unlabeled_views[i], selected_index, [])
            else : 
                self.Labeled_views[i] = np.append(self.Labeled_views[i],self.Unlabeled_views[i][selected_index])
                self.Unlabeled_views[i]= np.delete(self.Unlabeled_views[i],selected_index)
        
        # add the label of selected samples  self.get_label(selected_index)
        self.y_labeled = np.append(self.y_labeled,self.y_unlabeled[selected_index]).astype(np.int32)

        self.y_unlabeled = np.delete(self.y_unlabeled,selected_index).astype(np.int32)
    
    
    def conservative(self,a):
        """
        Difference between Max and Min element of a 1-D array
        Parameters
        ----------
        a : array of float, shape==(n_students)
            The predictions made by each of the students.
        Returns
        -------
        value : integer
             The max margin value of the predictions.
        """
        return (np.max(a) - np.min(a))


    def aggressive(self,a):
        """ 
        Min element of a 1-D array
        a : array of float, shape==(n_students)
            The predictions made by each of the students.
        Returns
        -------
        value : integer
            The min value of the predictions.
        """
        return (np.min(a))

    def get_label(self,selected_index):
        """
        Asks the Oracle to provide the labels of the selected samples.
        Parameters
        ----------
        selected_index : list of int, shape==(n_samples)
            The selected samples for labeling.
        Returns
        -------
        labels : list of integer, shape=(n_samples)
            The labels of the corresponding samples.
        """
        labels = []
        for ind in selected_index :
            print()
            print()
            print(self.Unlabeled_pool[ind])
            print(self.y_unlabeled[ind])
            print()
            print()
            l = input('Please, provide the label of the following abstract')
            labels = np.append(labels,int(l))
            sys.stdout.flush()
            os.system('clear')
            os.system('cls')
            clear_output()
        return labels.astype(int)
    
    def QueryStrategy(self,vote_entropy,proba,mode,contention_points_index,n):
        """
        Selects the best fit of samples according to a certain 
        strategy among hte contention points.
        Parameters
        ----------
        vote_entropy : array-like of integer, shape==(n_samples, n_students)
            The predictions that each student gives to each sample.
        proba : array-like of float, shape=(n_samples, n_students, n_class)
        mode : string, the taken values are 'conservative', 'aggressive' and 'exploration' ,
            The query strategy to filter the contention points
        contention_points_index : array-like of integer, 
            The index of the contention points
        Returns
        -------
        selected_indices : list of integer, shape=(n)
            The top N query points.
        """
    
        max_disaagrement = np.max(vote_entropy)
        selected_indices =[]
        if(mode=='conservative') :
            # choose as query the contention point Q on which the least conﬁdent 
            # of the hypotheses h1,h2,...,hk makes the most conﬁdent prediction
            conservative_value = np.apply_along_axis(self.conservative, 1, proba)
            conservative_index = np.argsort(conservative_value)

            for v in conservative_index :
                if v in contention_points_index :
                    selected_indices = np.append(selected_indices,v).astype(int)
            return selected_indices

        if(mode=='aggressive') :
            # choose the contention point on which the conﬁdence of the predictions 
            # made by h1,h2,...,hk is as close as possible 
            # (ideally, they would be equally conﬁdent in predicting diﬀerent labels)
            aggressive_value = np.apply_along_axis(self.aggressive, 1, proba)
            aggressive_index = np.argsort(aggressive_value)[::-1]

            for v in aggressive_index :
                if v in contention_points_index :
                    selected_indices = np.append(selected_indices,v).astype(int)
            return selected_indices
        
        if(mode=='exploration') :
            # choose the exploration strategy in order to cover the whole feature space
            # of the occuring contention points, the exploration is done with the KFF algorithm
            X= self.Compute_tf_idf(self.X)
            X_fft = X[contention_points_index]
            farthest_samples_selected = self.fft(X_fft,distance.cdist(X_fft, X_fft, 'hamming'),self.batch_size)
            selected_indices = contention_points_index[farthest_samples_selected]
            return selected_indices
        
        if(mode=='random'):
            index_list = np.random.choice(contention_points_index, len(contention_points_index), replace=False)
            selected_indices = index_list[:n]
            return selected_indices
        
        
    def vote_disagreement(self,votes,n_students):
        """
        Return the disagreement measurement of the given number of votes.
        It uses the vote vote to measure the disagreement.
        Parameters
        ----------
        votes : list of int, shape==(n_samples, n_students)
            The predictions that each student gives to each sample.
        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The vote entropy of the given votes.
        """
        disagreement = []
        for candidate in votes:
            disagreement.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                disagreement[-1] -= lab_count[lab] / n_students * \
                    math.log(float(lab_count[lab]) / n_students)

        return disagreement

    def kl_divergence_disagreement(self, proba):
        """
        Calculate the Kullback-Leibler (KL) divergence disaagreement measure.
        Parameters
        ----------
        proba : array-like, shape=(n_samples, n_students, n_class)
        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The kl_divergence of the given probability.
        """
       
        n_students = np.shape(proba)[1]
        consensus = np.mean(proba, axis=1) # shape=(n_samples, n_class)
        # average probability of each class across all students
        consensus = np.tile(consensus, (n_students, 1, 1)).transpose(1, 0, 2)
        kl = np.sum(proba * np.log(proba / consensus), axis=2)
        disagreement = np.mean(kl, axis=1) 
        return disagreement
    

    def delete_from_csr(self,mat, row_indices=[], col_indices=[]):
            """Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) form the CSR sparse matrix ``mat``.
            WARNING: Indices of altered axes are reset in the returned matrix"""
            #if not isinstance(mat, csr_matrix):
                #raise ValueError("works only for CSR format -- use .tocsr() first")

            rows = []
            cols = []
            if len(row_indices)>0:
                rows = list(row_indices)
            if len(col_indices)>0:
                cols = list(col_indices)

            if len(rows) > 0 and len(cols) > 0:
                row_mask = np.ones(mat.shape[0], dtype=bool)
                row_mask[rows] = False
                col_mask = np.ones(mat.shape[1], dtype=bool)
                col_mask[cols] = False
                return mat[row_mask][:,col_mask]
            elif len(rows) > 0:
                mask = np.ones(mat.shape[0], dtype=bool)
                mask[rows] = False
                return mat[mask]
            elif len(cols) > 0:
                mask = np.ones(mat.shape[1], dtype=bool)
                mask[cols] = False
                return mat[:,mask]
            else:
                return mat
            
    
    def fft(self,X,D,k):
            """
            X: input vectors (n_samples by dimensionality)
            D: distance matrix (n_samples by n_samples)
            k: number of centroids
            out: indices of centroids
            """
            n=X.shape[0]
            visited=[]
            i=np.int32(np.random.uniform(n))
            i=0
            visited.append(i)
            while len(visited)<k:
                dist=np.mean([D[i] for i in visited],0)
                for i in np.argsort(dist)[::-1]:
                    if i not in visited:
                        visited.append(i)
                        break
            return np.array(visited)
    

	
	
	
	

	
	
"""Node and Tree class to support hierarchical clustering AL method.
Assumed to be binary tree.
Node class is used to represent each node in a hierarchical clustering.
Each node has certain properties that are used in the AL method.
Tree class is used to traverse a hierarchical clustering.
"""


class Node(object):
  """Node class for hierarchical clustering.
  Initialized with name and left right children.
  """

  def __init__(self, name, left=None, right=None):
    self.name = name
    self.left = left
    self.right = right
    self.is_leaf = left is None and right is None
    self.parent = None
    # Fields for hierarchical clustering AL
    self.score = 1.0
    self.split = False
    self.best_label = None
    self.weight = None

  def set_parent(self, parent):
    self.parent = parent


class Tree(object):
  """Tree object for traversing a binary tree.
  Most methods apply to trees in general with the exception of get_pruning
  which is specific to the hierarchical clustering AL method.
  """

  def __init__(self, root, node_dict):
    """Initializes tree and creates all nodes in node_dict.
    Args:
      root: id of the root node
      node_dict: dictionary with node_id as keys and entries indicating
        left and right child of node respectively.
    """
    self.node_dict = node_dict
    self.root = self.make_tree(root)
    self.nodes = {}
    self.leaves_mapping = {}
    self.fill_parents()
    self.n_leaves = None

  def print_tree(self, node, max_depth):
    """Helper function to print out tree for debugging."""
    node_list = [node]
    output = ""
    level = 0
    while level < max_depth and len(node_list):
      children = set()
      for n in node_list:
        node = self.get_node(n)
        output += ("\t"*level+"node %d: score %.2f, weight %.2f" %
                   (node.name, node.score, node.weight)+"\n")
        if node.left:
          children.add(node.left.name)
        if node.right:
          children.add(node.right.name)
      level += 1
      node_list = children
    return print(output)

  def make_tree(self, node_id):
    if node_id is not None:
      return Node(node_id,
                  self.make_tree(self.node_dict[node_id][0]),
                  self.make_tree(self.node_dict[node_id][1]))

  def fill_parents(self):
    # Setting parent and storing nodes in dict for fast access
    def rec(pointer, parent):
      if pointer is not None:
        self.nodes[pointer.name] = pointer
        pointer.set_parent(parent)
        rec(pointer.left, pointer)
        rec(pointer.right, pointer)
    rec(self.root, None)

  def get_node(self, node_id):
    return self.nodes[node_id]

  def get_ancestor(self, node):
    ancestors = []
    if isinstance(node, int):
      node = self.get_node(node)
    while node.name != self.root.name:
      node = node.parent
      ancestors.append(node.name)
    return ancestors

  def fill_weights(self):
    for v in self.node_dict:
      node = self.get_node(v)
      node.weight = len(self.leaves_mapping[v]) / (1.0 * self.n_leaves)

  def create_child_leaves_mapping(self, leaves):
    """DP for creating child leaves mapping.
    
    Storing in dict to save recompute.
    """
    self.n_leaves = len(leaves)
    for v in leaves:
      self.leaves_mapping[v] = [v]
    node_list = set([self.get_node(v).parent for v in leaves])
    while node_list:
      to_fill = copy.copy(node_list)
      for v in node_list:
        if (v.left.name in self.leaves_mapping
            and v.right.name in self.leaves_mapping):
          to_fill.remove(v)
          self.leaves_mapping[v.name] = (self.leaves_mapping[v.left.name] +
                                         self.leaves_mapping[v.right.name])
          if v.parent is not None:
            to_fill.add(v.parent)
      node_list = to_fill
    self.fill_weights()

  def get_child_leaves(self, node):
    return self.leaves_mapping[node]

  def get_pruning(self, node):
    if node.split:
      return self.get_pruning(node.left) + self.get_pruning(node.right)
    else:
      return [node.name]




class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None



# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hierarchical cluster AL method.
Implements algorithm described in Dasgupta, S and Hsu, D,
"Hierarchical Sampling for Active Learning, 2008
"""

class HierarchicalClusterAL(SamplingMethod):
  """Implements hierarchical cluster AL based method.
  All methods are internal.  select_batch_ is called via abstract classes
  outward facing method select_batch.
  Default affininity is euclidean and default linkage is ward which links
  cluster based on variance reduction.  Hence, good results depend on
  having normalized and standardized data.
  """

  def __init__(self, X, X_train,y, index_train, index_test,seed, beta=2, affinity='euclidean', linkage='ward',
               clustering=None, max_features=None):
    """Initializes AL method and fits hierarchical cluster to data.
    Args:
      X: data
      y: labels for determinining number of clusters as an input to
        AgglomerativeClustering
      seed: random seed used for sampling datapoints for batch
      beta: width of error used to decide admissble labels, higher value of beta
        corresponds to wider confidence and less stringent definition of
        admissibility
        See scikit Aggloerative clustering method for more info
      affinity: distance metric used for hierarchical clustering
      linkage: linkage method used to determine when to join clusters
      clustering: can provide an AgglomerativeClustering that is already fit
      max_features: limit number of features used to construct hierarchical
        cluster.  If specified, PCA is used to perform feature reduction and
        the hierarchical clustering is performed using transformed features.
    """
    from sklearn.model_selection import GridSearchCV
    self.name = 'hierarchical'
    self.seed = seed
    np.random.seed(seed)
    # Variables for the hierarchical cluster
    self.already_clustered = False
    if clustering is not None:
        self.model = clustering
        self.already_clustered = True
    self.n_leaves = None
    self.n_components = None
    self.children_list = None
    self.node_dict = None
    self.root = None  # Node name, all node instances access through self.tree
    self.tree = None
    # Variables for the AL algorithm
    self.initialized = False
    self.beta = beta
    self.labels = {}
    self.pruning = []
    self.admissible = {}
    self.selected_nodes = None
    # Data variables
    self.classes = None
    self.X = X
    self.X_t = X_train
    self.index_test = np.array([])
    self.index_train = np.array([])
    self.X_test = np.array([])
    self.X_train = np.array([])
    self.model = None

    classes = list(set(y))
    self.n_classes = len(classes)
    if max_features is not None:
      transformer = PCA(n_components=max_features)
      transformer.fit(X)
      self.transformed_X = transformer.fit_transform(X)
      #connectivity = kneighbors_graph(self.transformed_X,max_features)
      self.model = AgglomerativeClustering(
          affinity=affinity, linkage=linkage, n_clusters=len(classes))
      self.fit_cluster(self.transformed_X)
    else:
      self.model = AgglomerativeClustering(
          affinity=affinity, linkage=linkage, n_clusters=len(classes))
      self.fit_cluster(self.X)
    self.y = y

    self.y_labels = {}# np.zeros(shape=len(y)) 
    # Fit cluster and update cluster variables

    self.create_tree()
    
    stopwords_gensim = []
    stopwords_nltk = stopwords.words('english')
    for i in STOPWORDS:
           stopwords_gensim.append(i)
    self.nltk_gensim_stop_words = np.union1d(stopwords_nltk,stopwords_gensim)
    self.nltk_gensim_stop_words = self.nltk_gensim_stop_words.tolist()
        
    print('Finished creating hierarchical cluster')
    

    self.index_train, self.index_test =  index_train, index_test 
    
    #train_test_split(index, test_size=1 - p_train, stratify=self.y[index])
    

 
  def update_index(self,selected_index):
        self.index_train = np.append(self.index_train,selected_index)
        self.index_test = np.delete(self.index_test,np.where(np.isin(self.index_test,selected_index)))
        
  def fit_cluster(self, X):
    if not self.already_clustered:
      self.model.fit(X)
      self.already_clustered = True
    self.n_leaves = self.model.n_leaves_
    self.n_components = self.model.n_components_
    self.children_list = self.model.children_

  def create_tree(self):
    node_dict = {}
    for i in range(self.n_leaves):
      node_dict[i] = [None, None]
    for i in range(len(self.children_list)):
      node_dict[self.n_leaves + i] = self.children_list[i]
    self.node_dict = node_dict
    # The sklearn hierarchical clustering algo numbers leaves which correspond
    # to actual datapoints 0 to n_points - 1 and all internal nodes have
    # ids greater than n_points - 1 with the root having the highest node id
    self.root = max(self.node_dict.keys())
    self.tree = Tree(self.root, self.node_dict)
    self.tree.create_child_leaves_mapping(range(self.n_leaves))
    for v in node_dict:
      self.admissible[v] = set()

  def get_child_leaves(self, node):
    return self.tree.get_child_leaves(node)

  def get_node_leaf_counts(self, node_list):
    node_counts = []
    for v in node_list:
      node_counts.append(len(self.get_child_leaves(v)))
    return np.array(node_counts)

  def get_class_counts(self, y):
    """Gets the count of all classes in a sample.
    Args:
      y: sample vector for which to perform the count
    Returns:
      count of classes for the sample vector y, the class order for count will
      be the same as that of self.classes
    """
    unique, counts = np.unique(y, return_counts=True)
    complete_counts = []
    for c in self.classes:
      if c not in unique:
        complete_counts.append(0)
      else:
        index = np.where(unique == c)[0][0]
        complete_counts.append(counts[index])
    return np.array(complete_counts)

  def observe_labels(self, labeled):
        
    #for i in labeled:
     #     self.y_labels[i] = labeled[i]
    #self.classes = np.array(
     #   sorted(list(set([self.y_labels[k] for k in self.y_labels]))))
    #self.n_classes = len(self.classes)
    
    for i in range(len(labeled)) :  ############
        self.y_labels[labeled[i]] = self.y[labeled[i]]
        
    self.classes = np.array(
        sorted(list(set([self.y_labels[k] for k in self.y_labels]))))
    
    print('hhhhhhhhhhhhhh',self.classes)
    print('hhhhhhhhhhhhhhm',self.y_labels)
    self.n_classes = len(self.classes)

  def initialize_algo(self):
    self.pruning = [self.root]
    self.labels[self.root] = np.random.choice(self.classes)
    node = self.tree.get_node(self.root)
    node.best_label = self.labels[self.root]
    self.selected_nodes = [self.root]

  def get_node_class_probabilities(self, node, y=None):
    children = self.get_child_leaves(node)
    if y is None:
      y_dict = self.y_labels
    else:
      y_dict = dict(zip(range(len(y)), y))
    labels = [y_dict[c] for c in children if c in y_dict]
    # If no labels have been observed, simply return uniform distribution
    if len(labels) == 0:
      return 0, np.ones(self.n_classes)/self.n_classes
    return len(labels), self.get_class_counts(labels) / (len(labels) * 1.0)

  def get_node_upper_lower_bounds(self, node):
    n_v, p_v = self.get_node_class_probabilities(node)
    # If no observations, return worst possible upper lower bounds
    if n_v == 0:
      return np.zeros(len(p_v)), np.ones(len(p_v))
    delta = 1. / n_v + np.sqrt(p_v * (1 - p_v) / (1. * n_v))
    return (np.maximum(p_v - delta, np.zeros(len(p_v))),
            np.minimum(p_v + delta, np.ones(len(p_v))))

  def get_node_admissibility(self, node):
    p_lb, p_up = self.get_node_upper_lower_bounds(node)
    all_other_min = np.vectorize(
        lambda i:min([1 - p_up[c] for c in range(len(self.classes)) if c != i]))
    lowest_alternative_error = self.beta * all_other_min(
        np.arange(len(self.classes)))
    return 1 - p_lb < lowest_alternative_error

  def get_adjusted_error(self, node):
    _, prob = self.get_node_class_probabilities(node)
    error = 1 - prob
    admissible = self.get_node_admissibility(node)
    not_admissible = np.where(admissible != True)[0]
    error[not_admissible] = 1.0
    return error

  def get_class_probability_pruning(self, method='lower'):
    prob_pruning = []
    for v in self.pruning:
      label = self.labels[v]
      label_ind = np.where(self.classes == label)[0][0]
      if method == 'empirical':
        _, v_prob = self.get_node_class_probabilities(v)
      else:
        lower, upper = self.get_node_upper_lower_bounds(v)
        if method == 'lower':
          v_prob = lower
        elif method == 'upper':
          v_prob = upper
        else:
          raise NotImplementedError
      prob = v_prob[label_ind]
      prob_pruning.append(prob)
    return np.array(prob_pruning)

  def get_pruning_impurity(self, y):
    impurity = []
    for v in self.pruning:
      _, prob = self.get_node_class_probabilities(v, y)
      impurity.append(1-max(prob))
    impurity = np.array(impurity)
    weights = self.get_node_leaf_counts(self.pruning)
    weights = weights / sum(weights)
    return sum(impurity*weights)

  def update_scores(self):
    node_list = set(range(self.n_leaves))
    # Loop through generations from bottom to top
    while len(node_list) > 0:
      parents = set()
      for v in node_list:
        node = self.tree.get_node(v)
        # Update admissible labels for node
        admissible = self.get_node_admissibility(v)
        admissable_indices = np.where(admissible)[0]
        for l in self.classes[admissable_indices]:
          self.admissible[v].add(l)
        # Calculate score
        v_error = self.get_adjusted_error(v)
        best_label_ind = np.argmin(v_error)
        if admissible[best_label_ind]:
          node.best_label = self.classes[best_label_ind]
        score = v_error[best_label_ind]
        node.split = False

        # Determine if node should be split
        if v >= self.n_leaves:  # v is not a leaf
          if len(admissable_indices) > 0:  # There exists an admissible label
            # Make sure label set for node so that we can flow to children
            # if necessary
            assert node.best_label is not None
            # Only split if all ancestors are admissible nodes
            # This is part  of definition of admissible pruning
            admissible_ancestors = [len(self.admissible[a]) > 0 for a in
                                    self.tree.get_ancestor(node)]
            if all(admissible_ancestors):
              left = self.node_dict[v][0]
              left_node = self.tree.get_node(left)
              right = self.node_dict[v][1]
              right_node = self.tree.get_node(right)
              node_counts = self.get_node_leaf_counts([v, left, right])
              split_score = (node_counts[1] / node_counts[0] *
                             left_node.score + node_counts[2] /
                             node_counts[0] * right_node.score)
              if split_score < score:
                score = split_score
                node.split = True
        node.score = score
        if node.parent:
          parents.add(node.parent.name)
        node_list = parents

  def update_pruning_labels(self):
    for v in self.selected_nodes:
      node = self.tree.get_node(v)
      pruning = self.tree.get_pruning(node)
      self.pruning.remove(v)
      self.pruning.extend(pruning)
    # Check that pruning covers all leave nodes
    node_counts = self.get_node_leaf_counts(self.pruning)
    assert sum(node_counts) == self.n_leaves
    # Fill in labels
    for v in self.pruning:
      node = self.tree.get_node(v)
      if node.best_label  is None:
        node.best_label = node.parent.best_label
      self.labels[v] = node.best_label

  def get_fake_labels(self):
    fake_y = np.zeros(self.X.shape[0])
    for p in self.pruning:
      indices = self.get_child_leaves(p)
      fake_y[indices] = self.labels[p]
    return fake_y

  def train_using_fake_labels(self, model, X_test, y_test):
    classes_labeled = set([self.labels[p] for p in self.pruning])
    if len(classes_labeled) == self.n_classes:
      fake_y = self.get_fake_labels()
      model.fit(self.X, fake_y)
      test_acc = model.score(X_test, y_test)
      return test_acc
    return 0

  def select_batch_(self, N, already_selected, labeled, y, **kwargs):
    # Observe labels for previously recommended batches
    self.observe_labels(labeled)

    if not self.initialized:
      self.initialize_algo()
      self.initialized = True
      print('Initialized algo')

    print('Updating scores and pruning for labels from last batch')
    self.update_scores()
    self.update_pruning_labels()
    print('Nodes in pruning: %d' % (len(self.pruning)))
    print('Actual impurity for pruning is: %.2f' %
          (self.get_pruning_impurity(y)))

    # TODO(lishal): implement multiple selection methods
    selected_nodes = set()
    weights = self.get_node_leaf_counts(self.pruning)
    probs = 1 - self.get_class_probability_pruning()
    weights = weights * probs
    weights = weights / sum(weights)
    batch = []

    print('Sampling batch')
    while len(batch) < N:
      node = np.random.choice(list(self.pruning), p=weights)
      children = self.get_child_leaves(node)
      children = [
          c for c in children if c not in self.y_labels and c not in batch
      ]
      if len(children) > 0:
        selected_nodes.add(node)
        batch.append(np.random.choice(children))
    self.selected_nodes = selected_nodes
    return batch

  def to_dict(self):
    output = {}
    output['node_dict'] = self.node_dict
    return output

  def Compute_tf_idf(self,X) :
    vectorizer = TfidfVectorizer(max_df=0.5,#0.7 700
                                 min_df=3,
                                 norm='l2',                                
                                 stop_words=self.nltk_gensim_stop_words,
                                 use_idf=True,
                                 ngram_range=(1,3),
                                 max_features=500
                                )
    y = vectorizer.fit_transform(X)
   # return np.array(y.toarray())
    svd = TruncatedSVD(n_components=2, n_iter=400)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(normalizer)
    y = lsa.fit_transform(y)
    #return y
    return np.array(y.toarray())

  def Compute_tf_idf_fit(self,X) :
    vectorizer = TfidfVectorizer(max_df=0.5,#0.7 700
                                 min_df=3,
                                 norm='l2',                                
                                 stop_words=self.nltk_gensim_stop_words,
                                 use_idf=True,
                                 ngram_range=(1,3),
                                 max_features=600
                                )
    y = vectorizer.fit(X)
    return y


  def Compute_tf_idf_fit2(self,X) :
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=3, 
                                 stop_words='english',
                                 use_idf=True,
                                 ngram_range=(1,3),
                                max_features=500
                                )
    y = vectorizer.fit(X)
    return y

  def Compute_tf_idf_fit3(self,X) :
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2, 
                                 stop_words='english',
                                 use_idf=True,
                                 ngram_range=(1,3),
                                  max_features=500
                                )
    y = vectorizer.fit(X)
    return y


  def Compute_tf_idf_fit4(self,X) :
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=4, 
                                 stop_words='english',
                                 use_idf=True,
                                 ngram_range=(1,3),
                                 max_features=600
                                )
    y = vectorizer.fit(X)
    return y


  def Compute_tf_idf_transform(self,vectorizer,X) :
    y = vectorizer.transform(X)
    return np.array(y.toarray())
    print(len(X))
    svd = TruncatedSVD(n_components=100,n_iter=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(normalizer)
    y = lsa.fit_transform(y)
    #return y
    return np.array(y.toarray())


  def Compute_tf_idf_transform3(self,vectorizer,X) :
    y = vectorizer.transform(X)
    svd = TruncatedSVD(n_components=min(len(X),100),n_iter=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(normalizer)
    y = lsa.fit_transform(y)
    #return y
    return np.array(y.toarray())