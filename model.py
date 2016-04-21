import numpy as np
import thread
import time
import os
import sys
import threading
import dataReader
import time
from multiprocessing import Process, Queue

numThreads = 6
HIDDEN = 4500
MB = 34
LR = 0.001
EPOCHS = 100
NUM_LAYERS = 3
BADCOUNT = 14

#-----------------------------------------------------------------------
## UTILITY FUNCTIONS ##

def dsigmoid(y):
    return y * (1.0 - y)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=0)
    return sf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initNormal(in_size, layer_size):
    '''
    Initialize weights based on a normal distribution.
    '''
    weights = np.random.standard_normal(size=(in_size, layer_size))
    bias = np.random.standard_normal(size=(layer_size))

    return weights, bias
    
    
def initNormalized(in_size, layer_size):
    '''
    Initialize weights using Normalized Distribution technique (Glorot and Bengio,
      2010). Paper found at:

    http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf

    '''
    min_val = np.negative(np.sqrt(6.0) / np.sqrt(in_size + layer_size))
    max_val = np.sqrt(6.0) / np.sqrt(in_size + layer_size)
    
    weights = np.random.random_integers(min_val, max_val,size=(in_size, layer_size))
    bias = np.random.standard_normal(min_val, max_val,size=(layer_size))

    return weights, bias

#-----------------------------------------------------------------------
## MODEL CLASS ##
class Net(object):
    '''
    This class is the Network class. It handles the training of the
      model, the gradient computation, the accuracy computation, 
    '''
    def __init__(self, num_layers, layer_size, feature_dim, label_dim):
        self._num_layers = num_layers
        self._feature_dim = feature_dim
        self._label_dim = label_dim
        self._layer_size = layer_size
        self._lr = LR


    def initialize(self, init_type):
        '''
        Initize all layers of the network.
        '''
        self._layers = []
        
        ## If only one layer
        if self._num_layers == 1:
            self._layers.append(Layer(self._layer_size, self._feature_dim, init_type))
            self._layers.append(Layer(self._label_dim, self._layer_size, init_type))

        ## If multiple layers
        else:
            self._layers.append(Layer(self._layer_size, self._feature_dim, init_type))

            for i in range(0, self._num_layers - 2):
                self._layers.append(Layer(self._layer_size, self._layer_size, init_type))

            self._layers.append(Layer(self._label_dim, self._layer_size, init_type))

        for i in range(0, len(self._layers)):
            self._layers[i].initializeParams()
        

    def feedForward(self, inputs):
        '''
        Push data through the network to get outputs.
        '''
        x = inputs
        
        for l in range(0, len(self._layers)):
            x = self._layers[l].getHidden(x)
        
        o = softmax(x)
        
        return o
        
        
    def getAcc(self, o, label):    
        '''
        Get the accuracy of a single output and a single label.
        '''
        m = -100000
        m_class = -1
        label_class = -1
        for i in range(0, self._label_dim):
            if o[i] > m:
                m = o[i]
                m_class = i

            if label[i] == 1:
                label_class = i

        return [m_class, label_class]
        
    def threadedGrad(self, i, deltas):
        '''
        Compute gradient. 

        This function is called when using threaded
          gradient computation.
        '''
        new_weights = self._layers[i].getWeights()
        grad = self._layers[i].getGrad()
        last_hidden = self._layers[i].getLastHidden()
        for j in range(self._layer_size): 
            for k in range(self._label_dim):
                change = deltas[k] * last_hidden[j]
                new_weights[j][k] -= self._lr * change + grad[j][k]
                grad[j][k] = change
        self._layers[i].updateGrad(grad)

    def backProp(self, o, label):
        '''
        This function updates the model using backpropagation.

        It first computes the error, using the negative log likelihood
          loss function.
          
        Once the error is computed at the output error, the error is then 
          backpropagated downward to the input layer. At that point, set 
          the weights of the model to the updated weights.
        '''
        if o.shape[0] != self._label_dim:
            print("Incorrect number of targets for number of labels.")
            print(o.shape[0])
            print(self._label_dim)
            exit(0)
            
        error = 0.0
        for i in range(0,label.shape[0]):
            error += -np.sum(label[i,:] * np.log(o[i,:]))

        ## Compute deltas for last layer
        output_deltas = [0.0] * self._label_dim        
        for i in range(0, self._label_dim):
            #error = -np.sum(label * np.log(o[i]))
            output_deltas[i] = dsigmoid(o[i]) * error
                    
        ## Compute hidden deltas
        hidden_deltas = []
        for i in range(0, self._num_layers - 1):
            temp = [0.0] * self._layer_size
            hidden_deltas.append(temp)


        for i in range(self._num_layers-1, 0):            
            for j in range(self._layer_size): 
                error = 0.0
                for k in range(self._label_dim):
                    error += output_deltas[k] * self._layers[i].getWeights()[j][k]
                hidden_deltas[i][j] = dsigmoid(self._layers[i].getLastHidden()[j]) * error

        
        if numThreads > 1:

            threads = []
            for i in range(self._num_layers, 0):
                if i == self._num_layers:
                    p = Process(target=threadedGrad, args=(i, output_deltas))
                elif i < self._num_layers:
                    p = Process(target=threadedGrad, args=(i, hidden_deltas))
                p.start()
                threads.append(p)

            for p in threads:
                p.join()

        else:

            for i in range(self._num_layers, 0):
                new_weights = self._layers[i].getWeights()
                grad = self._layers[i].getGrad()
                last_hidden = self._layers[i].getLastHidden()
                #last_hidden = self._layers[-1].getLastHidden()
                for j in range(self._layer_size): 
                    for k in range(self._label_dim):
                        change = 0
                        if i == self._num_layers:
                            change = output_deltas[k] * last_hidden[j]
                        elif i < self._num_layers:
                            change = hidden_deltas[k] * last_hidden[j]
                        new_weights[j][k] -= self._lr * change + grad[j][k]
                        grad[j][k] = change
                self._layers[i].updateGrad(grad)
                    

        return (error/MB)

    def train(self, train_feats, train_labels):
        '''
        Train the model using the gradient descent algorithm.
        '''
        error = 1000000
        ## These are the files to print the results to.
        #res = open("./results/multiThread_results.txt", "a")
        #ff_file = open("./results/feedForward_times.txt", "a")
        #bp_file = open("./results/backProp_times.txt", "a")
        t0 = time.clock()
        for i in range(EPOCHS):
            error = 1000000
            best_iter = -1
            best_o = 0
            best_y = 0
            e_start = time.clock()
            for f in range(0, len(train_feats)):                
                x = train_feats[f]
                y = train_labels[f]

                print("Starting feedforward")
                ff_start = time.clock()
                o = self.feedForward(x)
                ff_end = time.clock()
                ff_file.write("Time for one feedforward is " + str((ff_end - ff_start)) + " using " + str(numThreads) + " threads." + "\n\tfeature batch: " + str(f) + "; batchSize = " + str(MB) + "; numLayers: " + str(NUM_LAYERS) + "; hiddenSize: " + str(HIDDEN) + "\n")

                print("Starting backProp")
                bp_start = time.clock()
                #new_err, accuracy = self.backProp(o, y)
                new_err = self.backProp(o, y)
                bp_end = time.clock()
                bp_file.write("Time for one backprop is " + str((bp_end - bp_start)) + " using " + str(numThreads) + " threads." + "\n\tfeature batch: " + str(f) + "; batchSize = " + str(MB) + "; numLayers: " + str(NUM_LAYERS) + "; hiddenSize: " + str(HIDDEN) + "\n")

                print("new err: " + str(new_err))
                print("old err: " + str(error))
                #print("Accuracy: " + str(accuracy))
                print("Training sample " + str(f))
                print("Epoch : " + str(i))
                if new_err < error:
                    print("Error update: " + str(new_err))
                    print("iter: " + str(f))
                    error = new_err
                    best_iter = f
                    print(o)
                    print(y)
                    best_o = o
                    best_y = y

            e_end = time.clock()
            epoch_total = e_end - e_start        
            print("Final error for epoch " + str(i) + " is " + str(error)) 
            print("Total time for epoch " + str(i) + "using " + str(numThreads) + ": " + str(epoch_total))

            res.write("Final error for epoch " + str(i) + " is " + str(error) + "\n") 
            res.write("\tTotal time for epoch " + str(i) + "using " + str(numThreads) + ": " + str(epoch_total) + "\n")

        t1 = time.clock()
        total = t1 - t0
        res.write("\t\tTOTAL TRAIN TIME: " + str(total))
        res.close()
        return error



class Layer(object):
    '''
    This is the Layer class. It handles the initialization and changing of
      the weights, as well as the computation of the hidden representations.
    '''
    
    def __init__(self, in_size, layer_size, init_type):
        """
        :param in_size: input size dimension
        :param layer_size: hidden dim
        :param init_type: String denoting which type of weight initialization
          to use. Either 'basic' for choosing randomly from normal distribution,
          or 'normalized' to initialized based on layer sizes.
        """
        self._in_size = in_size
        self._layer_size = layer_size
        self._weights = np.zeros((self._layer_size, self._in_size))
        self._bias = np.random.randn(self._layer_size)
        self._last_hidden = np.zeros((MB, layer_size))
        self._gradient = np.zeros((in_size, layer_size))
        self._init = init_type


    def getGrad(self):
        return self._gradient

    def updateGrad(self, new_grad):        
        self._gradient = new_grad

    def getWeights(self):
        return self._weights

    def getBias(self):
        return self._bias

    def getLastHidden(self):
        return self._last_hidden

    def getInitType(self):
        return self._init

    def updateWeights(self, new_weights):
        if new_weights.shape[0] != self._weights.shape[0]:
            print("ERROR: New weights are the wrong size.(dim 0)")

        if new_weights.shape[1] != self._weights.shape[1]:
            print("ERROR: New weights are the wrong size.(dim 1)")

        self._weights = new_weights

    def initializeParams(self):
        if self._init == 'basic':
            self._weights, self._bias = initNormal(self._in_size, self._layer_size)
        elif self.init == 'normalized':
            self._weights, self._bias = initNormalized(self._in_size, self._layer_size)
            
    def mul(self, x, W, A, start, end):
        # This assumes that x is already a column vector
        # x.shape[1] = W.shape[0]

        for i in range (start, end):
            for j in range(0, W.shape[0]):
                A[j,:] += W[j][i] * x[i,:]


    def getHidden(self, inputs):
        '''
        This will return the hidden representation based on the input
          vectors given.
        
        This method handles the parallelized matrix multiplication, using the
          mul() method defined above. If numThreads > 1, the operations will be
          parallelized; otherwise, the multiplication will be done naively.
        '''
        start = 0
        end = 0
        W = self._weights
        b = self._bias
        x = inputs
        dim = W.shape[1]
        ## X must be multi dimensional, of size MB x feature_dim (MB >= 1)
        H = np.zeros((W.shape[0], x.shape[1]))
        A = np.zeros((W.shape[0], x.shape[1]))

        if dim != x.shape[0]:
            print("Error: Dimensions of x, W don't line up.")
            print("W dim: " + str(W.shape) + " x dim: " + str(x.shape))
            exit(-1)

        if dim%numThreads != 0:
            print("Error: Dimension not divisible by " + str(numThreads))
            print("dim: " + str(dim))
            exit(-1)

        if numThreads == 1:
            for i in range(0, W.shape[1]):
                for j in range(0, W.shape[0]):
                    A[j][0] += W[j][i] * x[i][0]
        else:
            ## NEW APPROACH
            chunk = dim / numThreads
            thread_list = []           
            for i in range(0, numThreads):            
                start = i * chunk
                end = (i + 1) * chunk            
                p = Process(target=self.mul, args=(x, W, A, start, end))
                p.start()
                thread_list.append(p)
                    
            for x in thread_list:
                x.join()

        H = sigmoid(A)
        self._last_hidden = H        
        return H

def main():    
    train, dev, test= dataReader.getData(True)
    feat_dim = train[0][0].shape[0]
    label_dim = train[1][0].shape[0]

    DNN = Net(NUM_LAYERS, HIDDEN, feat_dim, label_dim, 'normalized')
    DNN.initialize()
    
    print("starting training")
    error = DNN.train(train[0], train[1])

if __name__ == main():
    main()
