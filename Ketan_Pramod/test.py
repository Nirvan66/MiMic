from Mimic_T import Mimic_T

model = Mimic_T()
model.processModel("")
model.loadWeights("joey.h5")
out = model.predict("what should i do?")
print(out)