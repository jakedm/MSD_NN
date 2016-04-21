import numpy as np
import sys
import os
import csv
from sklearn.feature_extraction import DictVectorizer

feat_names = ['track_id','artist_name','title','loudness','tempo']
label_names = ['genre']

def getData(MB=False):
    data_name = 'data.txt'
    col_name = 'genre_dataset_cnames.txt'    
    

    cols = open(col_name, 'r')

    cnames = cols.readline()
    print(cnames)
    columns = cnames.split(',')
    data = []
    features = []
    labels = []

    with open(data_name, 'r') as f:
        temp = csv.DictReader(f,fieldnames=columns)        
        for row in temp:
            cnt = 0
            for i in range(0, len(row)):
                features.append(row)
            

    with open(data_name, 'r') as l:
        temp = csv.DictReader(l,fieldnames=columns)
        for row in temp:
            cnt = 0
            for i in range(0, len(row)):
                labels.append(row)

        

    #print(len(data[0].keys()))
    #print(len(out[0].keys()))

    ## Remove keys that I don't want from the data
    for i in features:
        for f in i.keys():
            if f not in feat_names:
                del i[f]



    #print(features[0].keys())

    for j in labels:
        for l in j.keys():
            if l not in label_names:
                del j[l]
                
    #print(labels[0].keys())

    v = DictVectorizer(sparse=False)
    X = v.fit_transform(features)
    Y = v.fit_transform(labels)
        
    train_size = X.shape[0]

    train_feats = X[0:train_size,:]
    train_labels = Y[0:train_size,:]

    x_dim = X.shape[1]
    y_dim = Y.shape[1]

    if MB is False:

        f = []
        l = []
        for i in range(0, train_feats.shape[0]):
            f.append(train_feats[i,:].reshape(x_dim, 1))
            l.append(train_labels[i,:].reshape(y_dim, 1))

        train = [f, l]

        dev = [f, l]
            
        test = [f, l]
        
    else:
        print("Minibatching by 34.")
        
        f = []
        l = []

        for i in range(0, train_feats.shape[0], 34):
            end = i+34
            f.append(train_feats[i:(i+34),:].reshape(x_dim, 34))
            l.append(train_labels[i:(34+i),:].reshape(y_dim, 34))

        train = [f, l]
            
        dev = [f, l]

        test = [f, l]


    return train, dev, test

'''
def main():
    d_file = 'data.txt'
    c_file = 'genre_dataset_cnames.txt'
    data = file_reader(d_file, c_file)
    print("returning data")
    return data

if __name__ == main():
    main()
'''
