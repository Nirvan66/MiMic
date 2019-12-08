import numpy as np
import random
import pickle
import ipywidgets as widgets

class HumanMetricTester:

    def __init__(self, model, statements, references, saveDataFolder='save_data/', sample=True, sample_size=10):
        self._model = model
        pairs = list(zip(statements, self._model.preProcessor.cleanTexts(references)))
        random.shuffle(pairs)
        self._pairs = pairs[0:min(sample_size,len(pairs))]

        self._save_button = None
        self._grade_button = None
        self._title = None
        self._score_box = None

        self._results = []

        self._answer_title_as = []
        self._answer_title_bs = []
        self._problem_titles = []
        self._answer_buttons = []
        self._answer_key = []

        self._correct_count = 0

    def question_update(self, _):
        self._save_button.icon='times'

    def save_button_on_click(self,wdgt):
        wdgt.icon='check'

        self._correct_count = 0
        for idx in range(0,len(self._pairs)):
            if self._answer_key[idx] == self._answer_buttons[idx].value:
                self._correct_count += 1
        self._score_box.value = self._correct_count/len(self._pairs)

    def grade_button_on_click(self,wdgt):
        wdgt.icon='check'
        self._save_button.disabled=True
        self._grade_button.disabled=True

        self._correct_count = 0
        for idx in range(0,len(self._pairs)):
            self._answer_buttons[idx].disabled=True
            if self._answer_key[idx] == self._answer_buttons[idx].value:
                self._correct_count += 1
                self._problem_titles[idx].value += ' CORRECT '
            else:
                self._problem_titles[idx].value += ' WRONG '
        self._score_box.value = self._correct_count/len(self._pairs)

    def getScore(self):
        return self._correct_count/len(self._pairs)

    def generateSurvey(self):
        self._title = widgets.HTML(
            value="<h2>For each of the following statements select the response which seems the most appropriate.</h2>"
        )

        self._save_button = widgets.Button(
            description='Score',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Compute Score',
            icon='times'
        )

        self._grade_button = widgets.Button(
            description='Grade',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Computers score and marks answers',
            icon='times'
        )

        self._save_button.on_click(self.save_button_on_click)
        self._grade_button.on_click(self.grade_button_on_click)
        self._score_box = widgets.FloatText(value=0,description='Score:',disabled=True)

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
        display(self._save_button, self._grade_button)
        display(self._score_box)
