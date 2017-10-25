# That file is taken from:
# https://github.com/ilkarman/DeepLearningFrameworks/blob/master/common/utils.py
import numpy as np
import os
import tarfile
import pickle
import subprocess
import sys
if sys.version_info.major == 2:
    # Backward compatibility with python 2.
    from six.moves import urllib
    urlretrieve = urllib.request.urlretrieve
else:
    from urllib.request import urlretrieve


def read_batch(src):
    '''Unpack the pickle files
    '''
    with open(src, 'rb') as f:
        if sys.version_info.major == 2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
    return data


    
def download_cifar(src="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", mode=1):
    '''Load the training and testing data
    '''
    # FLAG: should we host this on azure?
    if mode == 1:
        print ('Downloading ' + src)
        fname, h = urlretrieve(src, './delete.me')
        print ('Done.')
    else:
        None
    try:
        print ('Extracting files...')
        with tarfile.open(fname) as tar:
            tar.extractall()
        print ('Done.')
        print ('Preparing train set...')
        train_list = [read_batch('./cifar-10-batches-py/data_batch_{0}'.format(i + 1)) for i in range(5)]
        x_train = np.concatenate([t['data'] for t in train_list])
        y_train = np.concatenate([t['labels'] for t in train_list])       
        print ('Preparing test set...')
        tst = read_batch('./cifar-10-batches-py/test_batch')
        x_test = tst['data']
        y_test = np.asarray(tst['labels'])
        print ('Done.')
    finally:
        os.remove(fname)
    return x_train, x_test, y_train, y_test



def cifar_for_library(channel_first=True, one_hot=False): 
    # Raw data
    x_train, x_test, y_train, y_test = download_cifar()
    # Scale pixel intensity
    x_train =  x_train/255.0
    x_test = x_test/255.0
    # Reshape
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)  
    # Channel last
    if not channel_first:
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
    # One-hot encode y
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test
