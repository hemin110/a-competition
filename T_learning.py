# -*- coding: utf-8 -*-
'''
auther@ AIbert
hantaoer@foxmail.com
'''
from sklearn.metrics import roc_auc_score as AUC
import pandas as pd
import numpy as np
 
class T_Learner(object):
    """docstring for T_Learner"""
    def __init__(self,clfA,clfB,clfC,step,bias,max_turns=5):
        self.clfA = clfA
        self.clfB = clfB
        self.clfC = clfC
        self.step = step
        self.bias = bias
        self.max_turns = max_turns
        self.score_a = 0
        self.score_b = 0
        self.score_c = 0

    def tlearn(self, train, test, validation, flag, featuresA, featuresB, featuresC,drop_rate):
        print len(featuresA),len(featuresB),len(featuresC)
        results_proba = pd.DataFrame()
        raw_test = test
        results = pd.DataFrame(raw_test['no'])
        features = list(set(featuresA+featuresB+featuresC))
        print "------- start Train-Validation-Test",len(train),len(validation),len(test),'------------'
        turn = 1
        while( turn <= self.max_turns):
            new = pd.DataFrame()

            self.clfA.fit(train[featuresA],train[flag])
            pred_a= self.clfA.predict_proba(train[featuresA])[:,1] 
            prob_a = self.clfA.predict_proba(test[featuresA])[:,1]
            prea = (prob_a > (np.max(prob_a)+np.min(prob_a))*0.6)
            valid_a = self.clfA.predict_proba(validation[featuresA])[:,1]
            print '>',

            self.clfB.fit(train[featuresB],train[flag])
            pred_b = self.clfB.predict_proba(train[featuresB])[:,1]
            prob_b = self.clfB.predict_proba(test[featuresB])[:,1]
            preb = (prob_a > (np.max(prob_b)+np.min(prob_b))*0.6)
            valid_b = self.clfB.predict_proba(validation[featuresB])[:,1]
            print '>',
            
            self.clfC.fit(train[features],train[flag])
            pred_c= self.clfC.predict_proba(train[features])[:,1]
            prob_c = self.clfC.predict_proba(test[features])[:,1]
            prec = (prob_c > (np.max(prob_c)+np.min(prob_c))*0.6)
            valid_c = self.clfC.predict_proba(validation[features])[:,1]
            print '>',
            
            valid_score_a = AUC(validation[flag],valid_a)
            valid_score_b = AUC(validation[flag],valid_b)
            valid_score_c = AUC(validation[flag],valid_c)
            valid_score = AUC(validation[flag], valid_a*valid_score_a + valid_b*valid_score_b + valid_c*valid_score_c)

            index1 = (prea==preb) & (prea==prec) # as long as test
            sum_va = valid_score_a+valid_score_b+valid_score_c
            prob = (prob_c[index1]*valid_score_c+prob_a[index1]*valid_score_a+prob_b[index1]*valid_score_b)/sum_va

            # an order of small to big
            alpha_low = np.sort(prob)[int(len(prob)*turn/2.0/self.max_turns)]-0.01
            alpha_high= np.sort(prob)[int(len(prob)*(1-turn/2.0/self.max_turns))]+0.01

            index2 = ((prob>alpha_high) | (prob<alpha_low)) 	# as long as prob

            new['no'] = test['no'][index1][index2]		# as long as selected Samples
            new['pred'] = prob[index2]
            results_proba = results_proba.append(new)

            # test['flag'] = prea
            rightSamples = test[index1][index2]
            rightSamples['flag'] = prea[index1][index2]

            score_sim = np.sum(abs(prob_a-prob_b)+abs(prob_a-prob_c)+abs(prob_b-prob_c)+0.1)/len(prob_a)# compute similarity            
            # drop worst samples in dataA from train set
            true_y = train.iloc[self.step:]['flag']
            train_prob = pred_a[self.step:]*valid_score_a + pred_b[self.step:]*valid_score_b + pred_c[self.step:]*valid_score_c
            train_score = AUC(true_y,train_prob)
            
            dropper = self.max_turns/(1+ drop_rate*np.exp(-self.max_turns)*valid_score)
            # Growth curve models
            loss_bias = 0
            if(self.step>0):
                true_y = train.iloc[0:self.step]['flag']
                temp = pred_a[0:self.step]*valid_score_a + pred_b[0:self.step]*valid_score_b + pred_c[0:self.step]*valid_score_c
                temp = (temp+0.1)/(max(temp)+0.2)
                temp = (true_y-1)*np.log(1-temp)-true_y*np.log(temp)
                # an order of small to big
                location = int(min(self.step, len(rightSamples)*dropper+2)*np.random.rand())
                loss_bias =  np.sort(temp)[-location]

                # print sum(temp >= loss_bias),len(train),self.step
                temp = np.append(temp,np.zeros(len(train)-self.step)-999)
                remain_index = (temp <= loss_bias)
                
                # print loss_bias,sum(remain_index),sum(1-remain_index),sum(temp==loss_bias)
                # print "dropper",dropper,location
                self.step = self.step-sum(1-remain_index)
            else:
                remain_index = []

            # build the new trainand test
            # print train.shape, rightSamples.shape
            # print train.columns==rightSamples.columns
            train = train[remain_index].append(rightSamples[features+[flag,'no']])
            test = test[~test.index.isin(rightSamples.index)]

            # print running details
            print turn,">>Drop",len(remain_index)-sum(remain_index),"Add",len(rightSamples),"Step",self.step,
            print 'Simlarity_Auc',score_sim,'Tr_Auc',train_score,'V_Auc',valid_score_a,valid_score_b,valid_score_c
            print ' >>>',len(train),len(test),"Bias",loss_bias
            turn += 1

        print '--------------- end -------------------'
        # compute test left
        prob_a = self.clfA.predict_proba(test[featuresA])[:,1]
        p_a = self.clfA.predict_proba(raw_test[featuresA])[:,1]
        valid_a = self.clfA.predict_proba(validation[featuresA])[:,1]

        prob_b = self.clfB.predict_proba(test[featuresB])[:,1]
        valid_b = self.clfB.predict_proba(validation[featuresB])[:,1]
        p_b = self.clfB.predict_proba(raw_test[featuresB])[:,1]

        prob_c = self.clfC.predict_proba(test[features])[:,1]
        valid_c = self.clfC.predict_proba(validation[features])[:,1]
        p_c = self.clfC.predict_proba(raw_test[features])[:,1]

        self.score_a = AUC(validation[flag],valid_a)
        # self.score_a = np.log(self.score_a/(1-self.score_a))
        self.score_b = AUC(validation[flag],valid_b)
        # self.score_b = np.log(self.score_b/(1-self.score_b))
        self.score_c = AUC(validation[flag],valid_c)
        
        print "score_a",self.score_a,"score_b",self.score_b,"score_c",self.score_c
        return p_a,p_b,p_c
