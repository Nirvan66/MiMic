import numpy as np
import random
import pickle
import re
import os
import subprocess

import nltk.translate.bleu_score as bleu_score
from rouge.rouge import rouge_n_sentence_level
from rouge.rouge import rouge_l_sentence_level
import ipywidgets as widgets
from datetime import datetime

class AutomaticMetricTester:
    def __init__(self, model, statements, references, tempDataFolder='temp/', sample=True, sample_size=10):
        self._model = model
        pairs = list(zip(statements, references))
        random.shuffle(pairs)
        self._statements = [pair[0] for pair in pairs[0:min(sample_size, len(pairs))]]
        self._references = [pair[1] for pair in pairs[0:min(sample_size, len(pairs))]]
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

        for (candidate, reference) in zip(self._candidates, self._references):
            #print(candidate, reference)

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

        meteor_cmd = ['java', '-jar', '-Xmx2G', 'meteor-1.5.jar', \
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

    # Code taken from https://web.archive.org/web/20171215025927/http://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html
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

    def compileScores(self):
        self.computeBLUEScore()
        self.computeROGUEScores()
        self.computeMETEORScore()
        self.computeWER()

    def printScores(self):
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

class HumanMetricTester:

    def __init__(self, model, statements, references, saveDataFolder='save_data/', sample=True, sample_size=10):
        self._model = model
        pairs = list(zip(statements, references))
        random.shuffle(pairs)
        self._pairs = pairs[0:min(sample_size,len(pairs))]
        #self._references = references
        self._save_data_name = saveDataFolder + '_save_data_' + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self._savable = False

        self._namebox = None
        self._save_button = None
        self._title = None

        self._results = []

        self._answer_title_as = []
        self._answer_title_bs = []
        self._problem_titles = []
        self._answer_buttons = []
        self._answer_key = []

        self._correct_count = 0

    def on_name_enter(self, wdgt):
        self._save_data_name = wdgt.value + save_data_name
        self._savable = True

    def question_update(self, _):
        self._save_button.icon='times'

    def save_button_on_click(self,wdgt):
        wdgt.icon='check'

        self._correct_count = 0
        for idx in range(0,len(self._pairs)):
            if self._answer_key[idx] == self._answer_buttons[idx].value:
                self._correct_count += 1

    def getScore(self):
        return self._correct_count/len(self._pairs)

    def generateSurvey(self):
        self._namebox = widgets.Text(
            value='',
            placeholder='Name',
            description='Name:',
            disabled=False
        )

        self._namebox.on_submit(self.on_name_enter)

        self._title = widgets.HTML(
            value="<h2>For each of the following statements select the response which seems the most appropriate.</h2>"
        )

        self._save_button = widgets.Button(
            description='Score/Save',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='times'
        )

        self._save_button.on_click(self.save_button_on_click)

        count = -1
        for (statement, reference) in self._pairs:
            count += 1
            self._problem_titles.append(widgets.HTML(
            value="<h3>{}. Statement: \"{}\"</h3>".format(count, statement)))

            answers = [self._model.chat(statement)] + [reference]
            self._answer_key.append(np.random.randint(2))
            #s.append(list(range(0, len(answers))))
            #random.shuffle(self._answer_keys[-1])

            if self._answer_key[-1] == 0:
                self._answer_title_as.append(widgets.HTML(value="0. \"{}\"".format(self._model.chat(statement).strip())))
                self._answer_title_bs.append(widgets.HTML(value="1. \"{}\"".format(reference)))
            else:
                self._answer_title_as.append(widgets.HTML(value="0. \"{}\"".format(reference)))
                self._answer_title_bs.append(widgets.HTML(value="1. \"{}\"".format(self._model.chat(statement).strip())))

            self._answer_buttons.append(widgets.RadioButtons(
                options=[0,1],
                disabled=False,
            ))

            self._answer_buttons[-1].observe(self.question_update)

    def displaySurvey(self):
        #display(self._namebox)
        display(self._title)

        for idx in range(0, len(self._pairs)):
            display(self._problem_titles[idx])
            display(self._answer_title_as[idx])
            display(self._answer_title_bs[idx])
            display(self._answer_buttons[idx])
            #display(self._answer_keys[idx])

        display(self._save_button)
