import numpy as np
import os
import json
import operator
import pandas as pd
import math
class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def convert_to_df(self, arr):
        return pd.DataFrame(arr)

    def get_sse_class(self, y, classes):
        min_sse = None
        for classv in classes:
            sse = ((y - classv)**2).sum().tolist()[0]
            if not min_sse or sse < min_sse:
                min_sse = sse
        return min_sse

    def get_mean_sse(self, y):
        mean = y.mean().tolist()[0]
        return ((y - mean)**2).sum().tolist()[0]

    def select_optimal_splitting(self, X_df, y_df):
        n_rows, feat_len = X_df.shape
        #X_df, y_df = self.convert_to_df(X), self.convert_to_df(y)
        #X_df.columns = [str(i)  for i in range(feat_len)]
        min_sse = None
        #sse_details = (Feature, value)
        sse_details = None
        for i in range(feat_len):
            #if str(i) in features:
            #    continue
            curr_col = X_df[str(i)]
            uniq_values = np.sort(curr_col.unique())
            for value in uniq_values[:-1]:
                indx_1, indx_2 = curr_col <= value, curr_col > value
                R_1, y_1 = curr_col[indx_1].dropna(), y_df[indx_1].dropna()
                R_2, y_2 = curr_col[indx_2].dropna(), y_df[indx_2].dropna()
                uniq_y1, uniq_y2 = y_1[0].unique(), y_2[0].unique()
                #sse_y1, sse_y2 = self.get_sse_class(y_1, uniq_y1), self.get_sse_class(y_2, uniq_y2)
                sse_y1, sse_y2 = self.get_mean_sse(y_1), self.get_mean_sse(y_2)
                #Then compute the sse of 2 regions
                sse = sse_y1 + sse_y2
                #if min_sse:
                #    print (min_sse - sse)
                if min_sse == None or sse < min_sse:
                    min_sse = sse
                    sse_details = (str(i), value)
        return {
                'idx': sse_details[0],
                'threshold': sse_details[1],
        }

    def calculate_c(self, y):
        n_rows = y.shape[0]
        return y.sum().values.flatten()[0]/n_rows

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)
        You should update the self.root in this function.
        '''
        n_rows, feat_len = X.shape
        X_df, y_df = self.convert_to_df(X), self.convert_to_df(y)
        X_df.columns = [str(i)  for i in range(feat_len)]
        self.root = self.fit_aux(X_df, y_df, 0)
        print(self.root)

    def fit_aux(self, X, y, height):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        curr_depth = 0
        #Create a function to select the optimal splitting given the dataset.
        split_info = self.select_optimal_splitting(X, y)
        #Update self.root
        resp = {
            'splitting_variable': int(split_info['idx']),
            'splitting_threshold': split_info['threshold']
        }
        

        left_split, y_left = X[X[split_info['idx']] <= split_info['threshold']].dropna(), y[X[split_info['idx']] <= split_info['threshold']].dropna()
        right_split, y_right = X[X[split_info['idx']] > split_info['threshold']].dropna(), y[X[split_info['idx']] > split_info['threshold']].dropna()
        
        #print(left_split, y_left)
        #print("-----------------------------------------------------------------------")
        #print(right_split, y_right)
        #print("--------------------------------------------------------------------")
        #print("********************************************************************")
        if left_split.shape[0] < self.min_samples_split or height + 1 == self.max_depth:
            #It is a leaf node
            resp.update({
                'left': self.calculate_c(y_left)
            })
        else:
            #Do a Recursive iteration
            resp.update({
                'left': self.fit_aux(left_split, y_left, height + 1)
            })

        if right_split.shape[0] < self.min_samples_split or height + 1 == self.max_depth:
            #It is a leaf node
            resp.update({
                'right': self.calculate_c(y_right)
            })
        else:
            #Do a Recursive iteration.
            resp.update({
                'right': self.fit_aux(right_split, y_right, height + 1)
            })
        return resp

    def pred_row(self, row):
        curr_node = self.root
        while type(curr_node) == dict:
            idx = curr_node['splitting_variable']
            val = row[idx]
            if val <= curr_node['splitting_threshold']:
                curr_node = curr_node["left"]
            else:
                curr_node = curr_node["right"]
        return curr_node

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = []
        for row in X:
            y_pred.append(self.pred_row(row))
        return np.array(y_pred)

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            #splitting a nodeplitting a node into two child nod into two child nod
            json.dump(model_dict, fp)


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_string = tree.get_model_string()

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            print(operator.eq(model_string, test_model_string))
            
            y_pred = tree.predict(x_train)
            print(y_pred)
            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(y_test_pred)
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
