text_dir = "../text"
data_dir = "../data/"
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as Cvec
from sklearn.feature_extraction.text import TfidfVectorizer as Tvec
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
import gc
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score,classification_report,roc_curve
inter_dir = "..//intermediate_result//"
from collections import defaultdict
import pdb
label_field = "Cerebrovascular disease present"


def get_consolidated_doc_types():
    doc_type_consolidation = {
        'adult_triage_note': ['Adult Triage Note'],
        'discharge_summary': ['Discharge Summary - Medical', 'Discharge Summary.', 'Discharge Summary - General', 'Discharge Summary - Surgery Short', 'Discharge Summary - Orthopedic Surgery', 'Discharge Summary - Stroke Neurology', 'Discharge Summary Thoracic Surgery', 'Transfer Summary.'],
        'ed_physician_notes': ['ED Physician Handover Report'],
        'history_and_physical_examination':['History and Physical', 'History & Physical Examination.', 'Admitting Trauma History and Physical Assessment', 'History & Physical Examination'],
        'inpatient_consult_report':['Inpatient Consult Report.'],
        'inpatient_consultation':['Inpatient Consultation', 'Acute Pain Service Summary Note'],
        'inpatient_operative~procedure_report':['Inpatient Operative/Procedure Report'],
        'med_surg_met~notmet_assessment':['Med Surg MET/NOT-MET Assessment','Med Surg MET/NOT-MET Assessment Flowsheet', 'Mental Health MET/NOT-MET Assessment Flowsheet', 'Mental Health MET/NOT-MET Assessment'],
        'neurological_diagnostics':['EEG Preliminary Report', 'Neurological Diagnostics'],
        'non-physician_progress_notes':['MPR', 'MPR - OB'],
        'nursing_notes': ['Clinical Record', 'Pain Assessment', 'Intake and Output', 'Patient Care', 'Patient Assessment', 'AcuityPlus Inpatient Classification', 'Patient Assessment Tools', 'ED UCC - Intake and Output', 'Neurological Observation', 'Surgical Assessment and History - Nursing', 'Patient Assessment Neuro', 'Day Surgery / 24 Hour', 'AcuityPlus Mental Health Patient Classification'],
        'nursing_transfer_report':['Nursing Transfer Report - ED to IP', 'Nursing Transfer Report - IP to IP', 'Nursing Transfer Report - PACU to IP', 'Nursing Transfer Report - Mental Health'],
        'outpatient_consultation':['Outpatient Consultation'],
        'pharmacy_care_plan':['Pharmacy Care Plan'],
        'picc_~_midline_record':['PICC / Midline Record'],
        'rehabilitation':['Neuro Rehabilitation'],
        'social_work_assessment':['Social Work Assessment'],
        'urological_diagnostics':['Urological Diagnostics']
    }
    return doc_type_consolidation
    

#VISITIDCODE
def get_group_ids(merged_df,id_field = "VISITIDCODE"):
    pos_df = merged_df[merged_df[label_field] == "Yes"]
    neg_df = merged_df[merged_df[label_field] == "No"]
    pos_ids=list(pos_df[id_field].unique())
    neg_ids=list(neg_df[id_field].unique())
    return pos_ids,neg_ids

def prep_cui(merged_df,selected_types,pos_ids,neg_ids):
    '''
    Prepare the list of CUI as input and a list of 0/1 as the label for training supervised models. 
    INPUT:
        merged_df: data frame that contians text and the label of CEVD
        selected_types: list of strings
        pos_ids,neg_ids : list of ids
    OUTPUT:
        total_txt： list string, each element is the combination of one patient's all CUIs from documents within the selected types 
        total_label: list of 0 and 1.
    '''
    # get all pos and neg ids
    
    # get the selected columns, each type of document uses one field
    selected_cols = deepcopy(selected_types)
    selected_cols.append(label_field)
    selected_cols.append('VISITIDCODE')
    tmp_df = merged_df[selected_cols]
    pos_df = tmp_df[tmp_df[label_field] == "Yes"]
    neg_df = tmp_df[tmp_df[label_field] == "No"]
    
    pos_txt = [] # list of string, each string is the combination of all selected documents belong to one positive patient
    tmp_pos_ids = set(pos_df.VISITIDCODE.unique())
    for guid in list(pos_ids):
        if (guid in tmp_pos_ids):
            tmp_row = pos_df[pos_df.VISITIDCODE == guid].squeeze()
            tmp_list = [] # list of string, each element is CUIs from one document type
            for t in selected_types:
                if not pd.isna(tmp_row[t]):
                    tmp_list.append(tmp_row[t])
            tmp_s = ' '.join(tmp_list)
            pos_txt.append(tmp_s)
        else:
            pos_txt.append(" ")
    pos_label = [1] * len(pos_txt)
    neg_txt = []
    tmp_neg_ids = set(neg_df.VISITIDCODE.unique())
    for guid in list(neg_ids):
        if (guid in tmp_neg_ids):
            tmp_row = neg_df[neg_df.VISITIDCODE == guid].squeeze()
            tmp_list = []
            for t in selected_types:
                if not pd.isna(tmp_row[t]):
                    tmp_list.append(tmp_row[t])
            tmp_s = ' '.join(tmp_list)
            neg_txt.append(tmp_s)
        else:
            neg_txt.append(" ")
    neg_label = [0] * len(neg_txt)
    pos_txt.extend(neg_txt)
    total_txt = pos_txt
    pos_label.extend(neg_label)
    total_label = pos_label
    return total_txt,total_label

def prep_txt(merged_df,selected_types,pos_ids,neg_ids,txt_name = "no_neg_concept"):
    '''
    Prepare the list of documents as input and a list of 0/1 as the label for training supervised models. 
    INPUT:
        merged_df: data frame that contians text and the label of CEVD
        selected_types: list of strings
        pos_ids,neg_ids: list of ids
        txt_name: The column name of the column in the dataframe that contains the target text
    OUTPUT:
        total_txt： list string, each element is the combination of one patient's all documents within the selected types 
        total_label: list of 0 and 1.
    '''
    
    tmp_df = merged_df[merged_df.name.isin(set(selected_types))]
    #tmp_df is the data frame only has selected document types
    pos_df = tmp_df[tmp_df[label_field] == "Yes"]
    neg_df = tmp_df[tmp_df[label_field] == "No"]
    pos_group = pos_df.groupby('visit_guid')
    pos_txt = []
    tmp_pos_ids = set(pos_df.visit_guid.unique())
    for guid in list(pos_ids):
        if (guid in tmp_pos_ids): # some patient may do not exist in tmp_df because they do not have specific types of document
            tmp_list = pos_group.get_group(guid)[txt_name].tolist()
            tmp_s = ' '.join([str(sss) for sss in tmp_list])
            pos_txt.append(tmp_s)
        else:
            pos_txt.append(" ")
    pos_label = [1] * len(pos_txt)
    neg_group = neg_df.groupby('visit_guid')
    neg_txt = []
    tmp_neg_ids = set(neg_df.visit_guid.unique())
    for guid in list(neg_ids):
        if (guid in tmp_neg_ids):
            tmp_list = neg_group.get_group(guid)[txt_name].tolist()
            tmp_s = ' '.join([str(sss) for sss in tmp_list])
            neg_txt.append(tmp_s)
        else:
            neg_txt.append(" ")
    neg_label = [0] * len(neg_txt)
    pos_txt.extend(neg_txt)
    total_txt = pos_txt
    pos_label.extend(neg_label)
    total_label = pos_label
    return total_txt,total_label
    
def get_word_imp(total_txt,total_label,tmp_vec,tmp_model):
    '''
    Get feature importance from 5 models produced in 5 fold cross validation. And report the performance of each model as    
    classification_report in scikit-learn。
    INPUT:
        total_txt: input of models, list of string.
        total_label: label of models, list of binary value.
        tmp_vec: a vectorizer from scikit-learn
        tmp_model: a classification model
    OUTPUT:
        word_imp_dict: A dictionary, the key is the feature, which may be a word or CUI, depending on the content in total_txt. Value is the
        sum of the importance of the 5 models generated in 5 fold cross validation.
        cr_list: a list where each element is a classification report in dictionary form      
    '''
    word_imp_dict = defaultdict(lambda:0)
    # dictionary of importance. Key: words Value: average importances across 5 models
    kf = StratifiedKFold(n_splits = 5,shuffle=True,random_state=123)
    cr_list = []
    for (train_idx,test_idx) in tqdm(kf.split(total_txt,total_label)):
        p = 1
#          X_train, y_train = [], []
#         [(X_train.append(total_txt[i]), y_train.append(total_label[i])) for i in train_idx]
#         X_test, y_test = [], []
#         [(X_test.append(total_txt[i]), y_test.append(total_label[i])) for i in test_idx]
        X_train = [total_txt[i] for i in train_idx]
        X_test = [total_txt[i] for i in test_idx]
        y_train = [total_label[i] for i in train_idx]
        y_test = [total_label[i] for i in test_idx]
        bow_train = tmp_vec.fit_transform(X_train)
        bow_test = tmp_vec.transform(X_test)
        tmp_model.fit(bow_train,y_train)
        y_pred = tmp_model.predict_proba(bow_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1], pos_label=1)
        tpr_ind = np.max(np.where(fpr < 0.015)[0])
        tmp_dict = classification_report(y_test,y_pred[:,1]>=thresholds[tpr_ind],output_dict=True)
        tmp_dict['auc'] = roc_auc_score(y_test, y_pred[:,1])
        
        cr_list.append(tmp_dict)
        matched_tpr = tpr[tpr_ind]
        vocab = tmp_vec.get_feature_names_out()
        score = tmp_model.feature_importances_
        #high_score_idx = np.argsort(score)[-100:]

        for idx in range(len(score)):
            word_imp_dict[vocab[idx]] += score[idx]
        print(matched_tpr)
    #merged_cr = merge_cr_dict(cr_list)
    return word_imp_dict,cr_list

def run_models(total_txt,total_label,candi_vec,candi_models,fpr_thres = 0.005):
    '''
    Grid search of vectorizer and model, performance was measured by 5 fold cross validation.Grid search for vectorizer and model. Other 
    performance metrics are observed after adjusting the classification threshold to achieve a certain specificity. Performance was measured
    by 5 fold cross validation.
    INPUT：
        total_txt: list of text/cuis
        total_label: list of binary values
        candi_vec: list of scikit-learn's vectorizer
        candi_models: list of supervised models
        fpr_thres: threshold of false positive rate. Specificity = 1 - FPR 
    OUTPUT:
        mean_sens_dict: a dict of dict of list, {vectorizer : {model : [list of sensitivity]}}. Stores the list sensitivities of different
        combination of vectorizers and models
    '''
    kf = StratifiedKFold(n_splits = 5,shuffle=True,random_state=221)
    # mean_sens_dict stores the list sensitivities of different combination of vectorizers and models
    # {vectorizer : {model : [list of sensitivity]}}
    mean_sens_dict = defaultdict(lambda:defaultdict(lambda:[]))
    for (train_idx,test_idx) in tqdm(kf.split(total_txt,total_label)):
        p = 1
        X_train = [total_txt[i] for i in train_idx]
        X_test = [total_txt[i] for i in test_idx]
        y_train = [total_label[i] for i in train_idx]
        y_test = [total_label[i] for i in test_idx]
        #eval_set = [(X_train.iloc[indices[1]], Y_train.iloc[indices[1],0])]
        for vec_idx in range(len(candi_vec)):
            tmp_vec = candi_vec[vec_idx]
            bow_train = tmp_vec.fit_transform(X_train)
            bow_test = tmp_vec.transform(X_test)
            for model_idx in range(len(candi_models)):
                tmp_model = candi_models[model_idx]
                tmp_model.fit(bow_train,y_train)
                y_pred = tmp_model.predict_proba(bow_test)
                fpr,tpr,thres = roc_curve(y_test,y_pred[:,1])
                tpr_ind = np.max(np.where(fpr < fpr_thres))
                matched_tpr = tpr[tpr_ind]
                #tmp_auc = roc_auc_score(y_test,y_pred[:,1])
                #if (p):
                    #print(matched_tpr)
                    #p=0
                mean_sens_dict[vec_idx][model_idx].append(matched_tpr)
    return mean_sens_dict

def merge_cr_dict(dict_list):
    '''
    Merge classification report(cr) dictionaries, by averaging all metrics in report dictionaries in a list
    INPUT:
        dict_list: list of cr dict
    OUTPUT:
        result_dict: a single cr dict
    '''
    #log = r'{}:{}, mean {} +- {}, std {}'
    result_dict = defaultdict(lambda:[])
    result_dict = deepcopy(dict_list[0])
    result_dict['auc'] = [result_dict['auc']]
    count = len(dict_list)
    for k,v in result_dict.items():
        if type(v) == dict:
            for sub_k,sub_v in v.items():
                result_dict[k][sub_k] = [sub_v]
                for tmp_dict in dict_list[1:]:
                    result_dict[k][sub_k].append(tmp_dict[k][sub_k])
                result_dict[k][sub_k] = np.mean(result_dict[k][sub_k])
    for tmp_dict in dict_list[1:]:
        result_dict['auc'].append(tmp_dict['auc'])
    result_dict['auc'] = np.mean(result_dict['auc'])
    return result_dict
