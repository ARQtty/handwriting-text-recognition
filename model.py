
# coding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow.layers import conv2d, max_pooling2d, batch_normalization
from tensorflow.nn import relu
import logging
tf.get_logger().setLevel(logging.ERROR)

import preprocess


# IAM dataset contains 79 chars
charList = preprocess.loadCharList() + '~'
print("Dataset contains %d chars" % len(charList))
maxTextLen = 32


# ## Example image
# from matplotlib import pyplot as plt
# get_ipython().magic('matplotlib inline')
# img = np.trunc(np.random.random((32, 128)) * 255)
# plt.imshow(img, cmap='gray')
class EarlyStop():
    def __init__(self, patience, minDelta):
        self.patience = patience
        self.minDelta = minDelta
        self.notIncrease = 0
        self.minLoss = float('inf')

    def register(self, lossVal):
        if lossVal < self.minLoss:
            self.minLoss = lossVal
            self.notIncrease = 0
        else:
            if lossVal - self.minLoss < self.minDelta:
                self.notIncrease += 1

    def shouldStop(self):

        return self.notIncrease >= self.patience



# Building model
class Model():
    def __init__(self):
        # build net's graph
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            self.inputImgsPlaceholder = tf.placeholder(tf.float32, shape=(None, 128, 32))
            self.buildCNN()
            self.buildRNN()
            self.buildCTC()

            self.learningRatePlaceholder = tf.placeholder(tf.float32, shape=[])
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.RMSPropOptimizer(self.learningRatePlaceholder).minimize(self.loss)

        self.snapID = 0
        self.saver = tf.train.Saver(max_to_keep=1) # saver saves model to file

        # init TF Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.EarlyStop = EarlyStop


    def buildCNN(self):
        # in (None, 32, 128, 1)
        # out(None, 1, 32, 256) -> (None, 32, 256)
        with tf.name_scope("CNN"):
            x = tf.expand_dims(input=self.inputImgsPlaceholder, axis=3)
            kernelVals = [5, 5, 3, 3, 3]
            featureVals = [1, 32, 64, 128, 128, 256]
            strideVals = poolVals = [(2,2), (2,2), (1, 2), (1, 2), (1, 2)]
            numLayers = len(strideVals)

            # create layers
            for i in range(numLayers):
                kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
                conv = tf.nn.conv2d(x, kernel, padding='SAME',  strides=(1,1,1,1))
                conv_norm = tf.layers.batch_normalization(conv)
                relu = tf.nn.relu(conv_norm)
                x = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        x = tf.squeeze(x, axis=2)
        self.cnnOutTensor = x
        tf.summary.image("2 training data examples", tf.expand_dims(self.cnnOutTensor, axis=-1), max_outputs=2)


    def buildRNN(self):
        with tf.name_scope("RNN"):
            # basic cells which is used to build RNN
            numHidden = 256
            cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

            # stack basic cells
            stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            # bidirectional RNN
            # BxTxF -> BxTx2H
            ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked,
                                                            cell_bw=stacked,
                                                            inputs=self.cnnOutTensor,
                                                            dtype=self.cnnOutTensor.dtype)

            # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
            concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

            # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
            kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(charList) + 1], stddev=0.1))
            output = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=2)


        self.rnnOutTensor = output


    def buildCTC(self):
        with tf.name_scope("CTC"):
            x = tf.transpose(self.rnnOutTensor, [1, 0, 2])
            # ground truth text as sparse tensor
            self.gtTextsPlaceholder = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]),
                                                      tf.placeholder(tf.int32, [None]),
                                                      tf.placeholder(tf.int64, [2]))

            # calc loss for batch
            self.seqLenPlaceholder = tf.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTextsPlaceholder,
                                                      inputs=x,
                                                      sequence_length=self.seqLenPlaceholder,
                                                      ctc_merge_repeated=True))

            # calc loss for each element to compute label probability
            self.savedCtcInput = tf.placeholder(tf.float32, shape=[maxTextLen, None, len(charList) + 1])
            self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTextsPlaceholder,
                                                 inputs=self.savedCtcInput,
                                                 sequence_length=self.seqLenPlaceholder,
                                                 ctc_merge_repeated=True)

            # decoder: either best path decoding or beam search decoding
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=x,
                                                         sequence_length=self.seqLenPlaceholder,
                                                         beam_width=50,
                                                         merge_repeated=False)

            tf.summary.scalar('batch_loss', self.loss)


    def saveModel(self, sess):
        # saves model to file
        self.saver.save(sess, 'saved_models/snapshot', global_step=self.snapID)
        self.snapID += 1


    def toSparse(self, gtTexts):
        # puts ground truth texts into sparse tensor for ctc_loss
        indices = []
        values = []
        shape = [len(gtTexts), 0] # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(gtTexts):
            # convert to string of label (i.e. class-ids)
            try:
                labelStr = [charList.index(c) for c in text]
            except Exception as e:
                print("cant find char in %s" % text)
                raise e
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)


    def trainBatch(self, batch):
        imgs = batch[0]
        gtTexts = batch[1]
        batchLen = len(imgs)
        sparse = self.toSparse(gtTexts)


        rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 3000 else 0.0001) # decay learning rate

        evalList = [self.optimizer, self.mergedLogs, self.loss]
        feedDict = {self.inputImgsPlaceholder : imgs,
                    self.gtTextsPlaceholder : sparse ,
                    self.seqLenPlaceholder : [maxTextLen] * batchLen,
                    self.learningRatePlaceholder : rate}
        (_, summaryLoss, lossVal) = self.sess.run(evalList, feedDict)

        self.batchesTrained += 1
        self.train_writer.add_summary(summaryLoss, self.batchesTrained)
        del summaryLoss, batch, _

        return lossVal


    def train(self, batchSize=50):
        # loggers for Tensorboard
        writer = tf.summary.FileWriter("logs", self.sess.graph)
        self.mergedLogs = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('logs/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', self.sess.graph)


        self.batchesTrained = 0
        earlyStop = self.EarlyStop(patience=6, minDelta=0.005)
        trainEpochLosses = [float("inf")]

        epoch = 0
        while not earlyStop.shouldStop():
            batchNumber = 0
            losses = []
            epoch += 1
            print("Epoch %d" % epoch)

            # Train
            for batch in preprocess.batchGenerator(batchSize=batchSize, mode='train'):
                batchNumber += 1
                lossVal = self.trainBatch(batch)

                losses.append(lossVal)
                print("  Batch %d loss: %.2f" % (batchNumber, lossVal))

            epochLoss = sum(losses)/len(losses)
            earlyStop.register(epochLoss)

            if trainEpochLosses[-1] > epochLoss:
                self.saveModel(self.sess)
            trainEpochLosses.append(epochLoss)

            print("Epoch %d train summary:  mean loss %.2f" % (epoch, epochLoss))

            # Test
            #
            # if epochLoss < trainEpochLosses[-1]:
            #      saveModel(sess)
            #


        print("\nFinish learning")


model = Model()
model.train(batchSize=50)
