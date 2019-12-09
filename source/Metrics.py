import pickle

def CreateDataFile(model, statements, outputFile):
    candidates = [model.chat(statement) for statement in statements]
    pickle.dump( candidates, open(outputFile, "wb+" ) )
