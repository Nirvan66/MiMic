import numpy as np
import random
import pickle
import re
import os
import subprocess

import nltk.translate.bleu_score as bleu_score
from rouge.rouge import rouge_n_sentence_level
from rouge.rouge import rouge_l_sentence_level

class AutomaticMetricTester:
    def __init__(self, model, statements, references, tempDataFolder='temp/', sample=True, sample_size=10):
        self._model = model
        pairs = list(zip(statements, references))
        random.shuffle(pairs)

        if not sample:
            sample_size = len(pairs)
        else:
            sample_size = min(sample_size, len(pairs))
        self._statements = [pair[0] for pair in pairs[0:sample_size]]
        self._references = self._model.preProcessor.cleanTexts([pair[1] for pair in pairs[0:sample_size]])
        self._candidates = self.breakTexts([self._model.chat(text) for text in self._statements])
        self._tempDataFolder = tempDataFolder

        self._blue_score = None

        self._rouge_1_recall = []
        self._rouge_1_precision = []
        self._rouge_1_f1 = []

        self._rouge_4_recall = []
        self._rouge_4_precision = []
        self._rouge_4_f1 = []

        self._rouge_el_recall = []
        self._rouge_el_precision = []
        self._rouge_el_f1 = []

        self._meteor_precision = None
        self._meteor_recall = None
        self._meteor_f1 = None
        self._meteor_fmean = None
        self._meteor_score = None

        self._wer_error = None

    # TODO: Move to preproc
    def breakTexts(self, textList):
        return [text.split(' ') for text in self._model.preProcessor.cleanTexts(textList)]

    def packTexts(self, textList):
        return [[text] for text in textList]

    def unBreakTexts(self, textList):
        return [' '.join(text) for text in textList]

    def computeBLUEScore(self):
        blue_references = self.packTexts(self.breakTexts(self._references))
        blue_candidates = self._candidates #self.breakTexts(self._candidates)
        #print(blue_references)
        #print(blue_candidates)
        #print(blue_references)
        #print(blue_candidates)
        self._blue_score = bleu_score.corpus_bleu(blue_references, blue_candidates, smoothing_function=bleu_score.SmoothingFunction().method1)

    def computeROGUEScores(self):
        self._rouge_1_recall = []
        self._rouge_1_precision = []
        self._rouge_1_f1 = []

        self._rouge_4_recall = []
        self._rouge_4_precision = []
        self._rouge_4_f1 = []

        self._rouge_el_recall = []
        self._rouge_el_precision = []
        self._rouge_el_f1 = []

        rogue_references = self.breakTexts(self._references)
        for (candidate, reference) in zip(self._candidates, rogue_references):
            # print(candidate, reference)
            # We will consider the 1-gram version
            recall, precision, rouge = rouge_n_sentence_level(candidate, reference, 1)
            self._rouge_1_recall.append(recall)
            self._rouge_1_precision.append(precision)
            self._rouge_1_f1.append(rouge)

            # We will consider the 4-gram version
            recall, precision, rouge = rouge_n_sentence_level(candidate, reference, 4)
            self._rouge_4_recall.append(recall)
            self._rouge_4_precision.append(precision)
            self._rouge_4_f1.append(rouge)

            # We will consider the l version
            recall, precision, rouge = rouge_l_sentence_level(candidate, reference)
            self._rouge_el_recall.append(recall)
            self._rouge_el_precision.append(precision)
            self._rouge_el_f1.append(rouge)

    def computeMETEORScore(self):
        os.makedirs(self._tempDataFolder, exist_ok=True)

        f_ref = open(self._tempDataFolder + '/ref.dat', 'w+')
        f_cand = open(self._tempDataFolder + '/cand.dat', 'w+')

        met_candidates = self.unBreakTexts(self._candidates)
        for (reference, candidate) in zip(self._references, met_candidates):
            f_ref.write(reference + '\n')
            f_cand.write(candidate + '\n')

        f_ref.close()
        f_cand.close()

        meteor_cmd = ['java', '-jar', '-Xmx2G', 'resources/meteor-1.5.jar', \
                        'temp/ref.dat', 'temp/cand.dat', '-l', 'en', '-norm']
        meteor_output = subprocess.run(meteor_cmd, capture_output=True)

        if meteor_output.stderr.decode() != "":
            print("Error occured when running meteor tests!")
            print(meteor_output.stderr.decode())
        else:
            output = {}
            for line in meteor_output.stdout.decode().split('\n'):
                if ': ' in line:
                    key,value = line.split(': ')
                    output[key] = value

            self._meteor_precision = float(output['Precision'])
            self._meteor_recall = float(output['Recall'])
            self._meteor_f1 = float(output['f1'])
            self._meteor_fmean = float(output['fMean'])
            self._meteor_score = float(output['Final score'])

    #   taken from https://web.archive.org/web/20171215025927/http://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html
    def wer(self, references, candidates ,debug=False):
        r = references.split()
        h = candidates.split()
        # print(r)
        # print(h)
        #costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        SUB_PENALTY = 1
        INS_PENALTY = 1
        DEL_PENALTY = 1

        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r)+1):
            costs[i][0] = DEL_PENALTY*i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    costs[i][j] = costs[i-1][j-1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                    insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                    deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if debug:
            print("OP\tREF\tHYP")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i-=1
                j-=1
                if debug:
                    lines.append("OK\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub +=1
                i-=1
                j-=1
                if debug:
                    lines.append("SUB\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j-=1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i-=1
                if debug:
                    lines.append("DEL\t" + r[i]+"\t"+"****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("#cor " + str(numCor))
            print("#sub " + str(numSub))
            print("#del " + str(numDel))
            print("#ins " + str(numIns))
        return (numSub + numDel + numIns) / (float) (len(r))

    def computeWER(self):
        wer_candidates = self.unBreakTexts(self._candidates)

        self._wer_error = []
        for (reference,candidate) in zip(self._references, wer_candidates):
            if len(reference) != 0:
                self._wer_error.append(self.wer(reference, candidate, debug=False))

    def compileScores(self,cheat=False):
        if cheat:
            self._candidates = self.breakTexts(self._references)
        self.computeBLUEScore()
        self.computeROGUEScores()
        self.computeMETEORScore()
        self.computeWER()

    def printScores(self):
        print("Sample Size: ", len(self._statements))
        print("BLUE SCORE: ", self._blue_score)

        print("ROGUE 1 Recall Average: ", np.average(self._rouge_1_recall))
        print("ROGUE 1 Precision Average: ", np.average(self._rouge_1_precision))
        print("ROGUE 1 F1: ", np.average(self._rouge_1_f1))
        print("ROGUE 4 Recall Average: ", np.average(self._rouge_4_recall))
        print("ROGUE 4 Precision Average: ", np.average(self._rouge_4_precision))
        print("ROGUE 4 F1: ", np.average(self._rouge_4_f1))
        print("ROGUE el Recall Average: ", np.average(self._rouge_el_recall))
        print("ROGUE el Precision Average: ", np.average(self._rouge_el_precision))
        print("ROGUE el F1: ", np.average(self._rouge_el_f1))

        print("METEOR Precision: ", self._meteor_precision)
        print("METEOR Recall: ", self._meteor_recall)
        print("METEOR f1: ", self._meteor_f1)
        print("METEOR fmean: ", self._meteor_fmean)
        print("METEOR Score: ", self._meteor_score)

        print("WER Average Error: ", np.average(self._wer_error))
