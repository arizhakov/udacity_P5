#!/usr/bin/python
'''
    ---
    "POI Identifier". 
    Andri Rizhakov.
    DAN P5 IML. 
    Final Project.
    ---   
'''
###############################################################################
###############################################################################
###############################################################################
def main():
    print "=========="
    import sys
    #import os
    import pickle
    from time import time
    
    ## evaluation
    from sklearn.metrics import precision_score, recall_score 
    import matplotlib.pyplot as plt
    import pandas as pd
    #from ggplot import *
    import numpy as np
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    #from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.cross_validation import train_test_split
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ## current file running
    print "Running:", sys.argv[0].split("/")[-1]
    t_start_all = time()
    
    ### import helper functions
    sys.path.append("../tools/")
    from feature_format import featureFormat, targetFeatureSplit
    ## make sure 'tester' in same dir 
    from tester import dump_classifier_and_data
    
    ## moving loading dict code to be consistent with 'validate.py' ex from prev.
    ## lesson.
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    '''
    #Example structure of data_dict:
    >>> data_dict
            {'METTS MARK': {'salary': 365788, 
                            'to_messages': 807, 
                            'deferral_payments': 'NaN', 
                            'total_payments': 1061827, 
                            'exercised_stock_options': 'NaN', 
                            'bonus': 600000, 
                            'restricted_stock': 585062, 
                            'shared_receipt_with_poi': 702, 
                            'restricted_stock_deferred': 'NaN', 
                            'total_stock_value': 585062, 
                            'expenses': 94299, 
                            'loan_advances': 'NaN', 
                            'from_messages': 29, 
                            'other': 1740, 
                            'from_this_person_to_poi': 1, 
                            'poi': False, 
                            'director_fees': 'NaN', 
                            'deferred_income': 'NaN', 
                            'long_term_incentive': 'NaN', 
                            'email_address': 'mark.metts@enron.com', 
                            'from_poi_to_this_person': 38
                            }, 
            'BAXTER JOHN C': {'salary': 267102, 
                              'to_messages': 'NaN', 
                              'deferral_payments': 1295738, 
                              'total_payments': 5634343, 
                              'exercised_stock_options': 6680544, 
                              'bonus': 1200000, 
                              'restricted_stock': 3942714, 
                              'shared_receipt_with_poi': 'NaN', 
                              'restricted_stock_deferred': 'NaN', 
                              'total_stock_value': 10623258, 
                              'expenses': 11200, 
                              'loan_advances': 'NaN', 
                              'from_messages': 'NaN', 
                              'other': 2660303, 
                              'from_this_person_to_poi': 'NaN', 
                              'poi': False, 
                              'director_fees': 'NaN', 
                              'deferred_income': -1386055, 
                              'long_term_incentive': 1586055, 
                              'email_address': 'NaN', 
                              'from_poi_to_this_person': 'NaN'
                              },
            ...
    '''
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ##### Task 0. Data Exploration
    
    Rubric:
        ---
        Data Exploration (related mini-project: Lesson 5)
            Student response addresses the most important characteristics of the 
            dataset and uses these characteristics to inform their analysis. 
            Important characteristics include:
                total number of data points
                allocation across classes (POI/non-POI)
                number of features
                are there features with many missing values? etc.
        ---
    '''
    print "START: Task 0 - Explore data."
    t_start_0 = time()
    
    Boolean_doTask0 = False
    if Boolean_doTask0:
        ### Following L5, "explore_enron_data_16021614.py", do some data exploration.
        
        # How many data points (people) are in the dataset?
        #print "total number of data points, len(data_dict):", len(data_dict)
        #>>> 146
        
        # Display all keys:
        #print data_dict.keys()
        #>>> ['METTS MARK', 'BAXTER JOHN C', 'ELLIOTT STEVEN',.....]
        
        #print data_dict.items()
        #get list of dict items
        
        #print data_dict['METTS MARK'].keys()
        #>>> ['salary', 'to_messages', 'deferral_payments',
        
        #print "number of features, len(data_dict['METTS MARK'].keys()):", len(data_dict['METTS MARK'].keys())
        #>>> 21
        
        '''
        The poi feature records whether the person is a person of interest, 
        according to our definition. How many POIs are there in the E+F dataset? 
        In other words, count the number of entries in the dictionary where
        data[person_name]["poi"]==1
        '''
        list_count_poi = [key for key, value in data_dict.iteritems() if (data_dict[key]['poi']==True)]
        #print len(list_count_poi)
        #>>> 18
        #print "(POI/total), pre-outlier removal:", 1.0*len(list_count_poi)/(1.0*len(data_dict))
        #>>> 0.1233
        #print "(POI/non-POI), pre-outlier removal:", 1.0*len(list_count_poi)/(1.0*len(data_dict) - 1.0*len(list_count_poi))
        #>>> 0.1406
        
        #Whats the value of stock options exercised by Jeffrey Skilling?
        #print data_dict['SKILLING JEFFREY K'].keys()
        #print data_dict['SKILLING JEFFREY K']['exercised_stock_options']
        #>>> 19250000
        
        
        '''
        Of these three individuals (Lay, Skilling and Fastow), who took home the most 
        money (largest value of total_payments feature)?
        
        How much money did that person get?
        '''
        #print "Skilling total_payments:", data_dict['SKILLING JEFFREY K']['total_payments']
        #>>> 8682716
        #print "Lay total_payments:", data_dict['LAY KENNETH L']['total_payments']
        #>>> 103559793
        #print "Fastow total_payments:", data_dict['FASTOW ANDREW S']['total_payments']
        #>>> 2424083
        
        '''
        For nearly every person in the dataset, not every feature has a value. 
        How is it denoted when a feature doesnt have a well-defined value?
        #NaN
        '''
        
        #How many folks in this dataset have a quantified salary? 
        #What about a known email address?
        list_count_quantifiedSalary = [key for key, value in data_dict.iteritems() if (data_dict[key]['salary']!='NaN')]
        #print "len(list_count_quantifiedSalary):",len(list_count_quantifiedSalary)
        #>>> 95
        list_count_email_address = [key for key, value in data_dict.iteritems() if (data_dict[key]['email_address']!='NaN')]
        #print "len(list_count_email_address):",len(list_count_email_address)
        #>>> 111
        
        ## "allocation across classes" - take this to mean "how many features are 
        #non-NA"
        dict_summary = {}
        dict_summary2 = {}
        for feature in data_dict[data_dict.keys()[0]].keys():
            #print name
            dict_summary[feature] = [key for key, value in data_dict.iteritems() if (data_dict[key][feature]!='NaN')]
            #print "not NaN: len,", feature,":",len(dict_summary[feature])
            dict_summary2[feature] = len(dict_summary[feature])
        
        ## plot via pandas? see forum suggestions.
        ### can find meaningful trends, outliers in which features to use.
        #plt.scatter(ages, net_worths)
        #plt.show()
        
        columns = data_dict.keys()
        index = data_dict[data_dict.keys()[0]].keys()
        df1 = pd.DataFrame(index=index, 
                           columns=columns)
        df1 = df1.fillna(0) # with 0s rather than NaNs
        for name in columns:
            L_temp = []
            for feature in index:
                if data_dict[name][feature] == 'NaN':
                    L_temp.append(0.0)
                else:
                    L_temp.append(data_dict[name][feature])
            df1[name] = L_temp
        ##plot    
        #df1.transpose().plot(kind='scatter', x='salary', y='total_payments', color = 'poi')
        #bad df1 = df1.applymap(lambda x: 1 if x else 0)
        #df1.transpose().plot(kind='scatter', x='salary', y='total_payments', color = 'poi')
        
        ##ggplot way; needs ggplot to work
        #ggplot(aes(x='salary', y='total_payments', color='poi'), data=df1.transpose())  + geom_point()
        #ggplot not a 64-bit package!! cant use w 64-bit Anaconda
        ##need a non-ggplot approach to plotting by color. resume to matplotlib.
        df1T = df1.transpose()
        colors = np.where(df1T['poi'] == True, 'r', 'b')
        #plt.scatter(x=df1T['salary'], y=df1T['total_stock_value'], color = colors, alpha = 0.5)
        
        ##known outliers from manual check of enron.pdf
        df1T = df1T.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']) 
        
        #plt.scatter(x=df1T['salary'], y=df1T['total_stock_value'], color = colors, alpha = 0.5)
        #plt.scatter(x=df1T['exercised_stock_options'], y=df1T['total_payments'], color = colors, alpha = 0.5)
        
        #exampple df['race_label'] = df.apply (lambda row: label_race (row),axis=1)
        df1T['poi_float'] = df1T.apply(lambda row: 1.0 if row['poi'] == True else 0.0, axis = 1)
        #plt.scatter(x=df1T['exercised_stock_options'], y=df1T['poi_float'], color = colors, alpha = 0.5)
        
        '''
        [u'salary', u'to_messages', u'deferral_payments', u'total_payments',
               u'exercised_stock_options', u'bonus', u'restricted_stock',
               u'shared_receipt_with_poi', u'restricted_stock_deferred',
               u'total_stock_value', u'expenses', u'loan_advances', u'from_messages',
               u'other', u'from_this_person_to_poi', u'poi', u'director_fees',
               u'deferred_income', u'long_term_incentive', u'email_address',
               u'from_poi_to_this_person', u'poi_float']
        '''
        #plt.scatter(x=df1T['salary'], y=df1T['poi_float'], color = colors, alpha = 0.5)
        ## use this approach, but replace x-value with other variables.
        # salary is ok feature
        # to_messages is ok feature
        # deferral_payments is ok feature
        # total_payments is ok feature
        # exercised_stock_options is ok feature
        # bonus is ok feature
        # restricted_stock is ok feature
        # shared_receipt_with_poi is ok feature
        # restricted_stock_deferred is NOT ok feature
        #total_stock_value ok
        #expenses ok
        #loan_advances NOT ok
        #from_messages ~ok
        #other ~ok
        #from_this_person_to_poi ~ok
        #director_fees NOT ok
        #deferred_income ok
        #long_term_incentive ~ok
        #from_poi_to_this_person ~ok
        
        '''
        ##################### 
        ### Conclusions:
        #####################
        
        >> Feature selection:
        ok features: 
        salary, to_messages, deferral_payments, total_payments, 
        exercised_stock_options, bonus, restricted_stock, shared_receipt_with_poi, 
        total_stock_value, expenses, deferred_income
        
        not ok features: 
        restricted_stock_deferred, loan_advances, director_fees
        
        unsure features: 
        from_messages, other, from_this_person_to_poi, 
        long_term_incentive, long_term_incentive, from_poi_to_this_person
        
        >> Outlier detection:
        'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' need to be removed after manual 
        overview of the "enron61702insiderpay.pdf" doc. 
        '''
    
    print "END: Task 0 - Explore data."
    t_end_0 = time()
    print "Task 0 run time:", round(t_end_0 - t_start_0, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ##### Task 1: Select what features you'll use.
    
    Rubric:
        ---
        Intelligently select features (related mini-project: Lesson 11)
            Univariate or recursive feature selection is deployed, or features are 
            selected by hand (different combinations of features are attempted, and 
            the performance is documented for each one). Features that are selected 
            are reported and the number of features selected is justified. For an 
            algorithm that supports getting the feature importances 
            (e.g. decision tree) or feature scores (e.g. SelectKBest), those are 
            documented as well.
        Properly scale features (related mini-project: Lesson 9)
            If algorithm calls for scaled features, feature scaling is deployed.
        ---
    '''
    ## *features_list is a list of strings, each of which is a feature name.
    ## ** The first feature must be "poi".
    ## ** You will need to use more features
    print "START: Task 1 - Feature Selection."
    t_start_1 = time()
    
    ### for brevity, include all features, and deselect with automated tools, such 
    ### as KBest. Then, compare with anticipated features ("ok", "not ok") from T0.
    
    Boolean_doTask1 = True
    if Boolean_doTask1:
        features_all = data_dict[data_dict.keys()[0]].keys()
        ## errored. drop email address
        features_all.remove('email_address')
        features_all.remove('poi')
        features_list = ['poi'] + features_all
        
    print "END: Task 1 - Feature Selection."
    t_end_1 = time()
    print "Task 1 run time:", round(t_end_1 - t_start_1, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ### Task 2: Remove outliers
    
    Rubric:
        ---
        Outlier Investigation (related mini-project: Lesson 7)
            Student response identifies outlier(s) in the financial data, and 
            explains how they are removed or otherwise handled. Outliers are 
            removed or retained as appropriate.
        ---
    '''
    print "START: Task 2 - Remove Outliers."
    t_start_2 = time()
    
    Boolean_doTask2 = True
    if Boolean_doTask2:  
        del data_dict["TOTAL"]
        del data_dict["THE TRAVEL AGENCY IN THE PARK"]
        
    print "END: Task 2 - Remove Outliers."
    t_end_2 = time()
    print "Task 2 run time:", round(t_end_2 - t_start_2, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ### Task 3: Create new feature(s)
    
    Rubric:
        ---
        Create new features (related mini-project: Lesson 11)
            {} At least one new feature is implemented. Justification for that feature 
            is provided in the written response, and the effect of that feature on 
            the final algorithm performance is tested.
        ---
    '''
    print "START: Task 3 - Feature creation."
    t_start_3 = time()
    
    Boolean_doTask3 = True
    if Boolean_doTask3: 
        ### Store to my_dataset for easy export below.
        my_dataset = data_dict
        Boolean_doTask3_addNewFeatures = False
        if Boolean_doTask3_addNewFeatures:
            ### compute new features here, in "my_dataset", so not to disturb "data_dict"
            ## start: copy from studentCode_16030217.py, L11
            def computeFraction(poi_messages, all_messages):
                """ given a number messages to/from POI (numerator) 
                    and number of all messages to/from a person (denominator),
                    return the fraction of messages to/from that person
                    that are from/to a POI
               """
                ### you fill in this code, so that it returns either
                ###     the fraction of all messages to this person that come from POIs
                ###     or
                ###     the fraction of all messages from this person that are sent to POIs
                ### the same code can be used to compute either quantity
            
                ### beware of "NaN" when there is no known email address (and so
                ### no filled email features), and integer division!
                ### in case of poi_messages or all_messages having "NaN" value, return 0.
                if poi_messages == "NaN" or all_messages == "NaN":
                    fraction = 0.0
                else:
                    fraction = float(poi_messages)/float(all_messages)
                return fraction
            submit_dict = {}
            for name in my_dataset:
                data_point = my_dataset[name]
                ##from POI
                from_poi_to_this_person = data_point["from_poi_to_this_person"]
                to_messages = data_point["to_messages"]
                fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
                ##to POI
                from_this_person_to_poi = data_point["from_this_person_to_poi"]
                from_messages = data_point["from_messages"]
                fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
                ##populate dummy dict
                my_dataset[name]["fraction_from_poi"] = fraction_from_poi
                my_dataset[name]["fraction_to_poi"] = fraction_to_poi
            ## end: copy from studentCode_16030217.py    
            ## add newly generated features to past "features_list"
            features_list = features_list + ["fraction_from_poi", "fraction_to_poi"]
        ### Extract features and labels from dataset for local testing
        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        
    print "END: Task 3 - Feature creation."
    t_end_3 = time()
    print "Task 3 run time:", round(t_end_3 - t_start_3, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ### Task 4: Try a variety of classifiers
    
    Rubric:
        ---
        Pick an algorithm  (related mini-project: Lessons 1-3)
            {} At least 2 different algorithms are attempted and their performance 
            is compared, with the more performant one used in the final analysis.
        ---
    '''
    print "START: Task 4 - Classifier model study."
    t_start_4 = time()
    
    Boolean_doTask4 = True
    if Boolean_doTask4:
        models = []
        # append in |("name", clf_pipeline, param_grid)| format for each `models` 
        # entry. combine inside the loop, and then do CV, GridSearchCV, .fit, 
        # results.append(clf.score_)
        ### define scalars,arrays to use in GridSearch
        feature_min = 1
        if Boolean_doTask3_addNewFeatures:
            feature_max = 22 #change depending on how many new feat. if added = 22
        else:
            feature_max = 20 #20 is max for original num of features.
        max_svc_max_iter = int(1e5) #1e5 crashed. try 1e3. crashing due to non-scaling. return to 1e5.
        svm_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        svm_kernel = ["linear", "poly", "rbf", "sigmoid"]
        svm_gamma = [0.01, 0.1, 0.5, 0.9, 10, 100, 1000]            
        dt_min_samples_split = [2,5,10,20,40,80,100,200,500,1000]
        knn_n_neighbors = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,25,30,40,50,75,100]
        Boolean_doTask4_fullGridSearch = False
        if Boolean_doTask4_fullGridSearch:
            # SVM variants
            models.append(('MinMaxSclr_KBest_SVM',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", SelectKBest()),
                                     ("svm", SVC(max_iter=max_svc_max_iter))]), #SVM:max_iter=1000
                           dict(features__k = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma)
                            )
                           )
            models.append(('MinMaxSclr_PCA_SVM',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", PCA()),
                                     ("svm", SVC(max_iter=max_svc_max_iter))]),
                           dict(features__n_components = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma)
                            )
                           )
            models.append(('MinMaxSclr_PCAKBest_SVM',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("svm", SVC(max_iter=max_svc_max_iter))]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma) 
                            )
                           )
            models.append(('MinMaxSclr_KBestPCA_SVM',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("svm", SVC(max_iter=max_svc_max_iter))]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma) 
                            )
                           )
            '''
            models.append(('KBest_SVM',
                           Pipeline([("features", SelectKBest()),
                                     ("svm", SVC(max_iter=max_svc_max_iter))]), #SVM:max_iter=1000
                           dict(features__k = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma)
                            )
                           )
            ''' #removed because crash point.
            '''            
            models.append(('PCA_SVM',
                           Pipeline([("features", PCA()),
                                     ("svm", SVC(max_iter=max_svc_max_iter))]),
                           dict(features__n_components = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma)
                            )
                           )
            ''' #removed because crash point.
            '''
            models.append(('PCAKBest_SVM',
                           Pipeline([("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("svm", SVC(max_iter=max_svc_max_iter))]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma) 
                            )
                           )
            ''' #removed because crash point.
            '''
            models.append(('KBestPCA_SVM',
                           Pipeline([("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("svm", SVC(max_iter=max_svc_max_iter))]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                svm__C = svm_C, 
                                svm__kernel = svm_kernel, 
                                svm__gamma = svm_gamma) 
                            )
                           )
            ''' #removed because crash point.
            # GNB variants
            models.append(('MinMaxSclr_KBest_GNB',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", SelectKBest()),                                                                          
                                     ("gnb", GaussianNB())]),
                           dict(features__k = range(feature_min,feature_max))
                            )
                           )
            
            models.append(('MinMaxSclr_PCA_GNB',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", PCA()),                                                                          
                                     ("gnb", GaussianNB())]),
                           dict(features__n_components = range(feature_min,feature_max)) 
                            )
                           )
            models.append(('MinMaxSclr_PCAKBest_GNB',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("gnb", GaussianNB())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max)) 
                            )
                           )
            models.append(('MinMaxSclr_KBestPCA_GNB',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("gnb", GaussianNB())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max)) 
                            )
                           )
            models.append(('KBest_GNB',
                           Pipeline([("features", SelectKBest()),                                                                          
                                     ("gnb", GaussianNB())]),
                           dict(features__k = range(feature_min,feature_max))
                            )
                           )
            models.append(('PCA_GNB',
                           Pipeline([("features", PCA()),                                                                          
                                     ("gnb", GaussianNB())]),
                           dict(features__n_components = range(feature_min,feature_max)) 
                            )
                           )               
            models.append(('PCAKBest_GNB',
                           Pipeline([("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("gnb", GaussianNB())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max)) 
                            )
                           )
            models.append(('KBestPCA_GNB',
                           Pipeline([("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("gnb", GaussianNB())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max)) 
                            )
                           )
            # DT variants
            models.append(('KBest_DT',
                           Pipeline([("features", SelectKBest()),                                                                          
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__k = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split)
                            )
                           )
            models.append(('MinMaxSclr_PCA_DT',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", PCA()),                                                                          
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__n_components = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split) 
                            )
                           )               
            models.append(('MinMaxSclr_PCAKBest_DT',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split) 
                            )
                           )
            models.append(('MinMaxSclr_KBestPCA_DT',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split) 
                            )
                           )   
            models.append(('PCA_DT',
                           Pipeline([("features", PCA()),                                                                          
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__n_components = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split) 
                            )
                           )               
            models.append(('PCAKBest_DT',
                           Pipeline([("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split) 
                            )
                           )
            models.append(('KBestPCA_DT',
                           Pipeline([("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("dt", DecisionTreeClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                dt__min_samples_split = dt_min_samples_split) 
                            )
                           )
            # KNN variants
            models.append(('MinMaxSclr_KBest_KNN',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", SelectKBest()),                                                                          
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors)
                            )
                           )
            models.append(('MinMaxSclr_PCA_KNN',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", PCA()),                                                                          
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__n_components = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )               
            models.append(('MinMaxSclr_KBestPCA_KNN',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )
            models.append(('MinMaxSclr_PCAKBest_KNN',
                           Pipeline([("scale", MinMaxScaler(feature_range=(0, 1))),
                                     ("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )
            models.append(('KBest_KNN',
                           Pipeline([("features", SelectKBest()),                                                                          
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors)
                            )
                           )
            models.append(('PCA_KNN',
                           Pipeline([("features", PCA()),                                                                          
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__n_components = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )               
            models.append(('KBestPCA_KNN',
                           Pipeline([("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )
            models.append(('PCAKBest_KNN',
                           Pipeline([("features", FeatureUnion([("pca", PCA()), 
                                                                ("univ_select", SelectKBest())])),                                                                           
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )
        else:
            models.append(('KBestPCA_KNN',
                           Pipeline([("features", FeatureUnion([("univ_select", SelectKBest()),
                                                                ("pca", PCA())])),                                                                           
                                     ("knn", KNeighborsClassifier())]),
                           dict(features__pca__n_components = range(feature_min,feature_max),
                                features__univ_select__k = range(feature_min,feature_max),
                                knn__n_neighbors = knn_n_neighbors) 
                            )
                           )
        # prepare results reports                   
        best_estimators = []
        best_scores = []
        names = []
        cv = StratifiedShuffleSplit(y = labels, 
                                    n_iter = 10, #default is 10; change to 30 for increased fidelity (Rationale: approx min samples needed for good approx of Gaus distr). Failed w large SVM max? reduce and test w cv=1. {}
                                    test_size = 0.1, 
                                    random_state = 2016)
        # cycle through all grid searches
        for name, pipeline, param_grid in models:
            print "Start:", name
            grid_search = GridSearchCV(estimator = pipeline, 
                                       param_grid = param_grid, 
                                       verbose = 1,
                                       cv = cv,
                                       scoring = None, #default "scoring=None". try 'f1', 'recall' to combine both R and P.
                                       n_jobs = 1) # parallelize to lower runtime
            grid_search.fit(features, labels)
            best_estimators.append(grid_search.best_estimator_)
            best_scores.append([grid_search.best_score_])
            names.append(name)
            print "End:", name
            print "grid_search.best_score_:", grid_search.best_score_
        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(best_scores)
        ax.set_xticklabels(names)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.show()
    
    print "END: Task 4 - Classifier model study."
    t_end_4 = time()
    print "Task 4 run time:", round(t_end_4 - t_start_4, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ### Task 5: Tune your classifier to achieve better than 0.3 precision and recall 
    ### using our testing script. 
    
    Rubric:
        ---
        Tune the algorithm (related mini-project: Lessons 2, 3, 13)
            Response addresses what it means to perform parameter tuning and why it 
            is important. {} At least one important parameter tuned, with at least 
            3 settings investigated systematically, or any of the following are true:
                GridSearchCV used for parameter tuning
                Several parameters tuned
                Parameter tuning incorporated into algorithm selection (i.e. 
                    parameters tuned for more than one algorithm, and best 
                    algorithm-tune combination selected for final analysis)
        ---
    '''
    
    print "START: Task 5 - Classifier tuning."
    t_start_5 = time()
    
    Boolean_doTask5 = False
    if Boolean_doTask5:
        pass
        
    print "END: Task 5 - Classifier tuning."
    t_end_5 = time()
    print "Task 5 run time:", round(t_end_5 - t_start_5, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    print "----------"
    '''
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. 
    
    Rubric:
        ---
        Usage of Evaluation Metrics (related mini-project: Lesson 14)
            At least two appropriate metrics are used to evaluate algorithm 
                performance (e.g. precision and recall), and the student 
                articulates what those metrics measure in context of the project 
                task.
        Validation Strategy (related mini-project: Lesson 13)
            Response addresses what validation is and why it is important.
            Performance of the final algorithm selected is assessed by splitting 
            the data into training and testing sets or through the use of cross 
            validation, noting the specific type of validation performed.
        Algorithm Performance
            When tester.py is used to evaluate performance, precision and recall 
            are both at least 0.3.
        ---
    '''
    print "START: Task 6 - Dump classifier, dataset,and features_list."
    t_start_6 = time()
    
    Boolean_doTask6 = True
    if Boolean_doTask6:
        ###You do not need to change anything below, but make sure
        ### that the version of poi_id.py that you submit can be run on its own and
        ### generates the necessary .pkl files for validating your results.
        dump_classifier_and_data(best_estimators[best_scores.index(max(best_scores))], 
                                                 my_dataset, 
                                                 features_list)
        #output for display of finals results:
        print "names:", names
        print "best_scores:", best_scores
        print "names[best_scores.index(max(best_scores))]:", names[best_scores.index(max(best_scores))]
        print "best_scores[best_scores.index(max(best_scores))]:", best_scores[best_scores.index(max(best_scores))]
        print "best_estimators[best_scores.index(max(best_scores))]:", best_estimators[best_scores.index(max(best_scores))]
        print "best_estimators[best_scores.index(max(best_scores))].steps:", best_estimators[best_scores.index(max(best_scores))].steps
        
    print "END: Task 6 - Dump classifier, dataset,and features_list."
    t_end_6 = time()
    print "Task 6 run time:", round(t_end_6 - t_start_6, 16), "s"
    print "----------"
    ###############################################################################
    ###############################################################################
    ###############################################################################
    t_end_all = time()
    print "total run time:", round(t_end_all - t_start_all, 16), "s"
    print "=========="

if __name__ == '__main__':
    main()

