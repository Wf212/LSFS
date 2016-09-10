import pickle

def from_pickle(file_name):
    try :
        with open(file_name, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            data = unpickler.load()
    except EOFError:
        pass
    return data
    
    
def to_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, file=f)