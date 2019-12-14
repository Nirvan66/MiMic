# MiMic
MiMic is a chat-bot that mimics the personality of TV show characters.

You can mostly ignore all but the source and documentation files. The other files were staging grounds for individual code contributions.

In sources you will find several files.

The Data_Scrapper file shows how we extracted our test set.

The AutomaticMetrics file contains the code for running automatic metrics and examples can be found in EvaluateModel.
You will need both the data folder (which is not part of this repo) and resources folder (which is not part of this repo). The resources folder contains the meteor.jar file along with the paraphrase libraries (which you can get from https://www.cs.cmu.edu/~alavie/METEOR/).

For more information on the metrics used, see Charlie/AutomaticMetrics_Samples.

The HumanMetrics file contains code for generating the human evaluation scripts. You can run this code using the HumanTester file.

Mimic, Mimic_T, Smart and Stupid all contain models. To load Mimic and Mimic_T you will need the models which are not publicly available. Please contact authors for more information.
