'''
Creator: Nirvan S P Theethira
Date: 12/01/2019
Purpose:  CSCI 5266 Fall Group Project: Mimic
This files contains the code to train and test a seq2seq personality based chat bot.

PLEASE HAVE THE DATA IN THE SAME FOLDER AS THIS FILE BEFORE RUNNING

SAMPLE TRAIN RUN: 
python Mimic.py --trainCharachter joey --epochs 2 --batchSize 20 --saveEpochs 1 --modelSaveFile joey2

SAMPLE LOAD:
python Mimic.py --modelLoadFile joey2
'''

import numpy as np
import pickle
import keras
from keras import layers , activations , models , preprocessing
from keras import preprocessing , utils
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    '''
    Custom callback used to save model every few epochs.
    '''
    def __init__(self, mimic, saveFile, saveEpoch):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.mimic = mimic
        self.saveFile = saveFile
        self.saveEpoch =  saveEpoch

    def on_epoch_end(self, epoch, logs=None):
        '''
        Save model every few epochs
        '''
        self.mimic.accuracy = logs['accuracy']
        if epoch%self.saveEpoch==0:
            print('Saving model at {}'.format(self.saveFile))
            self.mimic.save(self.saveFile)

    def on_train_end(self, logs={}):
        '''
        Save model at the end of training
        '''
        print("Training sucessfull!! Saving model at {}".format(self.saveFile))
        self.mimic.save(self.saveFile)

class Mimic:
    '''
    The class us used to train a personality based chatbot.
    '''
    UNK = '<UNK>'
    START = '<START>'
    END = '<END>'

    def __init__(self, preProcessor, model=None, tokenizer=None, embeddingDim=200, metadata=None,):
        '''
        Used to initialize all required parameters.
        '''
        self.model = None
        self.maxInputLen = 0
        self.maxOutputLen = 0
        self.encoder = None
        self.decoder = None
        self.embeddingDim = embeddingDim
        self.preProcessor = preProcessor
        self.accuracy = -1
        if model!=None and tokenizer!=None and metadata!=None:
            self.model = model
            self.tokenizer = tokenizer
            self.vocabSize = len( self.tokenizer.word_index )+1
            self.maxInputLen = metadata['maxInputLen']
            self.maxOutputLen = metadata['maxOutputLen']
            self.embeddingDim = metadata['embeddingDim']
            self.accuracy = metadata['accuracy']
            self.extractChatbot()

    def extractEmbeddings(self, word2vecFile):
        '''
        Extract embedding weights from Glove vectors in the order of word tokens.
        '''
        print("Using {} for embedding weights".format(word2vecFile))
        embeddings = defaultdict(list,pickle.load(open(word2vecFile,'rb')))
        embeddingDim = len(list(embeddings.values())[0])
        mn = min([j for i in embeddings.values() for j in i])
        mx = max([j for i in embeddings.values() for j in i])
        embeddingMatrix = np.random.uniform(low=mn,high=mx,size=(self.vocabSize, embeddingDim))
        for word,index in self.tokenizer.word_index.items():
            if len(embeddings[word])>0:
                embeddingMatrix[index] = embeddings[word]
        return embeddingMatrix


    def build(self, inputs, outputs, word2vecFile=None):
        '''
        Creates tokenizer from corpus.
        Also creates all layers of the seq2seq model.
        '''
        processedInputs = self.preProcessor.cleanTexts(inputs)
        processedOutputs = self.preProcessor.cleanTexts(outputs, tokens=[self.START, self.END])

        self.tokenizer = preprocessing.text.Tokenizer(filters='\t\n', oov_token=self.UNK, lower=self.preProcessor.toLower)
        self.tokenizer.fit_on_texts(processedInputs + processedOutputs)
        self.vocabSize = len( self.tokenizer.word_index )+1
        print( 'Vocabulary size from corpus: {}'.format( self.vocabSize ))

        encoderInputs = keras.layers.Input(shape=( None , ))
        decoderInputs = keras.layers.Input(shape=( None , ))

        if word2vecFile==None:
            encoderEmbedding = keras.layers.Embedding(self.vocabSize, self.embeddingDim , mask_zero=True ) (encoderInputs)
            decoderEmbedding = keras.layers.Embedding( self.vocabSize, self.embeddingDim , mask_zero=True) (decoderInputs)
        else:
            embeddingMatrix = self.extractEmbeddings(word2vecFile)
            self.embeddingDim = len(embeddingMatrix[0])
            encoderEmbedding = keras.layers.Embedding(self.vocabSize, self.embeddingDim ,
                                                      mask_zero=True, weights=[embeddingMatrix]) (encoderInputs)
            decoderEmbedding = keras.layers.Embedding( self.vocabSize, self.embeddingDim ,
                                                      mask_zero=True, weights=[embeddingMatrix]) (decoderInputs)


        _ , state_h , state_c = keras.layers.LSTM( self.embeddingDim , return_state=True )( encoderEmbedding )
        encoderStates = [ state_h , state_c ]

        decoderLstm = keras.layers.LSTM( self.embeddingDim , return_state=True , return_sequences=True )
        decoderOutputs , _ , _ = decoderLstm ( decoderEmbedding , initial_state=encoderStates )

        decoderDense = keras.layers.Dense( self.vocabSize , activation=keras.activations.softmax )
        output = decoderDense ( decoderOutputs )

        self.model = keras.models.Model([encoderInputs, decoderInputs], output )
        self.model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics = ['accuracy'])
        self.extractChatbot()
        print(self.model.summary())
        return processedInputs, processedOutputs

    def getOneHot(self, tokenizedText):
        '''
        Convert tokenized words to one hot vectors.
        '''
        paddedText = np.zeros((tokenizedText.shape[0],tokenizedText.shape[1]))
        for i in range(len(tokenizedText)) :
            paddedText[i] = np.hstack((tokenizedText[i][1:],[0]))
        onehotText = utils.to_categorical( paddedText , self.vocabSize )
        return np.array( onehotText )

    def dataGen(self, tokenizedInputs, tokenizedOutputs, batchSize=10):
        '''
        Generates batches of data for training.
        '''
        paddedInputs = preprocessing.sequence.pad_sequences( tokenizedInputs , maxlen=self.maxInputLen , padding='pre' )
        encoderInput = np.array( paddedInputs )

        paddedAnswers = preprocessing.sequence.pad_sequences( tokenizedOutputs , maxlen=self.maxOutputLen , padding='post' )
        decoderInput = np.array( paddedAnswers )

        totalBatches = len(encoderInput)/batchSize
        counter=0
        while(True):
            prev = batchSize*counter
            nxt = batchSize*(counter+1)
            counter+=1
            decoderOutput = self.getOneHot(decoderInput[prev:nxt])
            yield [encoderInput[prev:nxt], decoderInput[prev:nxt]], decoderOutput
            if counter>=totalBatches:
                counter=0

    def fit(self, inputs, outputs, batchSize = 10, epochs = 20, saveFile=None, plot=False, saveEpoch=1):
        '''
        The main training function.
        '''
        tokenizedInputs = self.tokenizer.texts_to_sequences( inputs )
        mx = max( [ len(x) for x in tokenizedInputs ] )
        if mx>self.maxInputLen:
            self.maxInputLen = mx

        tokenizedOutputs = self.tokenizer.texts_to_sequences( outputs )
        mx = max( [ len(x) for x in tokenizedOutputs ] )
        if mx>self.maxOutputLen:
            self.maxOutputLen = mx
        if saveFile:
            callBack = [LossAndErrorPrintingCallback(self, saveFile, saveEpoch)]
        else:
            callBack = []

        evalutaion = self.model.fit_generator(self.dataGen(tokenizedInputs, tokenizedOutputs, batchSize=batchSize),
                                              epochs=epochs, steps_per_epoch = len(tokenizedInputs)/batchSize,
                                              callbacks=callBack)
        self.accuracy = evalutaion.history['accuracy'][-1]

        if plot:
            plt.plot(evalutaion.history['accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.show()

            plt.plot(evalutaion.history['loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()

    def extractChatbot(self):
        '''
        Create inference model.
        '''
        _, stateH, stateC = self.model.layers[4](self.model.layers[2](self.model.inputs[0]))
        self.encoder = keras.models.Model(self.model.inputs[0], [stateH, stateC])

        inputH = keras.layers.Input(shape=(self.embeddingDim,), name='inpH')
        inputC = keras.layers.Input(shape=(self.embeddingDim,), name='inpC')


        decoderOut, stateH2, stateC2 = self.model.layers[5](self.model.layers[3](self.model.inputs[-1]),
                                                       initial_state=[inputH, inputC])

        self.decoder = keras.models.Model([self.model.inputs[-1]] + [inputH, inputC],
                                   [self.model.layers[-1](decoderOut)] + [stateH2, stateC2])


    def chat(self, sentence):
        '''
        Pass input text through the model and extract output.
        '''
        sentence = self. preProcessor.cleanTexts([sentence])[0]
        padSentence = preprocessing.sequence.pad_sequences([self.tokenizer.texts_to_sequences([sentence])[0]] ,
                                                           maxlen=self.maxInputLen , padding='pre')
        #print([self.tokenizer.index_word[j] for j in padSentence[0] if j!=0])
        statesValues = self.encoder.predict(padSentence)
        inpTargetSeq = np.zeros( ( 1 , 1 ) )
        inpTargetSeq[0, 0] = self.tokenizer.word_index[self.START]
        reply = ''
        while (1):
            decOut , h , c = self.decoder.predict([ inpTargetSeq ] + statesValues )
            predIndex = np.argmax(decOut[0][0])
            predWord = self.tokenizer.index_word[predIndex]

            if predWord == self.END or len(reply.split()) > self.maxOutputLen:
                break

            reply += ' {}'.format(predWord)
            inpTargetSeq[ 0 , 0 ] = predIndex
            statesValues = [ h , c ]
        return reply

    def save(self, saveFile):
        if self.accuracy==-1:
            print("Cannot save untrained model!!!")
        else:
            if not os.path.isdir(saveFile):
                os.makedirs(saveFile)
            self.model.save(saveFile+'/model.h5')
            metaData = {
                        'maxInputLen':self.maxInputLen,
                        'maxOutputLen':self.maxOutputLen,
                        'preProcessor':self.preProcessor,
                        'embeddingDim':self.embeddingDim,
                        'accuracy':self.accuracy
                       }
            pickle.dump(metaData, open(saveFile+'/metaData.pkl', 'wb'))
            pickle.dump(self.tokenizer, open(saveFile+'/tokenizer.pkl', 'wb'))

    @classmethod
    def load(cls, loadFile):
        model = keras.models.load_model(loadFile+'/model.h5')
        metaData = pickle.load(open(loadFile+'/metaData.pkl', 'rb'))
        tokenizer = pickle.load(open(loadFile+'/tokenizer.pkl', 'rb'))
        return cls(preProcessor= metaData['preProcessor'], model=model, tokenizer=tokenizer, metadata=metaData)

class Preprocessor:
    '''
    Used to clean text before training.
    '''
    def __init__(self, lower=False, keepPunct='[.,!?;]'):
        self.toLower = lower
        self.keepPunct = keepPunct

    def cleanTexts(self, textList, tokens=None):
        '''
        Takes an input text and adds a space between words and keep punctuations.
        '''
        cleanText = []
        for sent in textList:
            sent = re.sub('[0-9]','',sent)
            if self.toLower:
                sent = sent.lower()
            words = re.findall(r"[\w]+|"+self.keepPunct, sent)
            if tokens:
                words = [tokens[0]]+words+[tokens[1]]
            cleanText.append(' '.join(words))
        return cleanText

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a seq2seq personality chatbot')
    parser.add_argument('--trainCharachter', type=str, help='Name of character to train on. \
                         This helps load data to train model on. eg:joey')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train on. eg:100')
    parser.add_argument('--batchSize', type=int, help='Number of batches to split data into. eg:10')
    parser.add_argument('--saveEpochs', type=int, help='Number of epochs to save after. eg:10')
    parser.add_argument('--modelSaveFile', type=str, help='File to save model to. eg:joey400')

    parser.add_argument('--modelLoadFile', type=str, help='File to load model from. eg:joey400')

    args = parser.parse_args()

    if args.trainCharachter:
        genericQuestions = pickle.load(
                           open('data/{}Data/genericQuestionsTrain.pkl'
                           .format(args.trainCharachter), 'rb'))
        genericAnswers = pickle.load(
                         open('data/{}Data/genericAnswersTrain.pkl'
                         .format(args.trainCharachter), 'rb'))
        print("\nGeneric data loaded: input :{}, output:{}".format(len(genericQuestions),len(genericAnswers)))
        r = np.random.randint(0,len(genericQuestions))
        print("Sample Generic data input: {}".format(genericQuestions[r]))
        print("Sample Generic data output: {}".format(genericAnswers[r]))

        personInput = pickle.load(
                      open('data/{}Data/{}InputTrain.pkl'
                      .format(args.trainCharachter,args.trainCharachter), 'rb'))         
        personOutput = pickle.load(
                       open('data/{}Data/{}OutputTrain.pkl'
                       .format(args.trainCharachter,args.trainCharachter), 'rb'))
        print("{} data loaded: input :{}, output:{}".format(args.trainCharachter,len(personInput),len(personOutput)))
        r = np.random.randint(0,len(personInput))
        print("Sample {} data input: {}".format(args.trainCharachter, personInput[r]))
        print("Sample {} data output: {}".format(args.trainCharachter, personOutput[r]))

        print("Training model on {} epochs in batches of {}".format(args.epochs, args.batchSize))
        mic = Mimic(Preprocessor())
        preQ, preA= mic.build(genericQuestions + personInput, genericAnswers + personOutput,
                               word2vecFile='data/{}Data/word2vec.pkl'.format(args.trainCharachter))

        e = mic.fit(preQ[0:100],preA[0:100],batchSize=args.batchSize,epochs=args.epochs,plot=True,
                    saveFile=args.modelSaveFile,
                    saveEpoch=args.saveEpochs)
        print("Done training starting chat bot. Have fun talking to {}".format(args.trainCharachter))

    elif args.modelLoadFile:
        mic = Mimic.load(args.modelLoadFile)
        print("Loaded model accuracy: {}".format(mic.accuracy))
    else:
        print("Error with input parameters. Please type: python Mimic.py --help")

    print("\nStarting chat bot")

    while(1):
        inp = input("\nInput test sentence or 0 to quit: ")
        if inp=='0':
            break
        print(mic.chat(inp))


