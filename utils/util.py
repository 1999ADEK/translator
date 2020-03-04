import os
from torchtext.data import Field, TabularDataset
from spacy.lang.zh import Chinese
from janome.tokenizer import Tokenizer

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(dataroot, isTest = False):
    nlp_zh = Chinese()
    tokenizer_zh = nlp_zh.Defaults.create_tokenizer(nlp_zh)
    tokenizer_ja = Tokenizer().tokenize
    
    def tokenize_zh(sentence):
        return [tok.text for tok in tokenizer_zh(sentence)]
    def tokenize_ja(sentence):
        return [tok for tok in tokenizer_ja(sentence,  wakati=True)]
    
    ZH_TEXT = Field(tokenize=tokenize_zh)
    JA_TEXT = Field(tokenize=tokenize_ja)
    
    data_fields = [('Chinese', ZH_TEXT), ('Japanese', JA_TEXT)]
    if isTest:
        test = TabularDataset.splits(path=dataroot, test='test.csv', format='csv', fields=data_fields)
    else:
        train, val = TabularDataset.splits(path=dataroot, train='train.csv', validation='val.csv', format='csv', fields=data_fields)
    
    ZH_TEXT.build_vocab(train)
    JA_TEXT.build_vocab(train)
    
    data_fields = {'Chinese': ZH_TEXT, 'Japanese': JA_TEXT}
    datasets = {'test': test} if isTest else {'train': train, 'val': val}
    return (data_fields, datasets)