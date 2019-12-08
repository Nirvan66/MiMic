from Mimic_T import Mimic_T

model = Mimic_T()
model.processModel()
model.loadWeights()
out = model.predict("what should i do?")
print(out)