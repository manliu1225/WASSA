def read_file(X_file, y_file = None):
    with open(X_file, 'r') as inputf: X = [e.strip() for e in inputf.readlines()]
    if y_file: 
        with open(y_file, 'r') as inputf: y = [e.strip() for e in inputf.readlines()]
    else: y = []
    return np.array(X), np.array(y)

class VSM:  # lookup provider for self-trained or pre-trained VSMs, e.g. http://nlp.stanford.edu/data/glove.6B.zi
    def __init__(self, src):
        self.map = {}
        self.dim = None
        self.source = src.split("/")[-1] if src is not None else "NA"
        # create dictionary for mapping from word to its embedding
        if src is not None:
            with open(src) as f:
                i = 0
                for line in f:
                    word = line.split()[0]
                    embedding = line.split()[1:]
                    self.map[word] = np.array(embedding, dtype=np.float32)
                    i += 1
                self.dim = len(embedding)
        else:
            self.dim = 1