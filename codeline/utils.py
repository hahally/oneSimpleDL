from nltk.text import TextCollection

# 读取文件
def read_file(file):
    lines = []
    with open(file, mode='r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            if line.isspace():
                continue
            lines.append(line)
            line = f.readline().strip()

    return lines

# 保存文件
def save_text(lines,file):
    with open(file=file, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line+'\n')

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
class GetTFidf():
    def __init__(self):
        super(TFIDF).__init__()
        self.tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True)
        
    def train_fit(self,corpus):
        
        return self.tfidf.fit_transform(corpus)
    
    def test_fit(self, sent:list):
        
        return self.tfidf.transform(sent)
    
    def get_vocab(self):
        
        return self.tfidf.get_feature_names()
    
    
    
