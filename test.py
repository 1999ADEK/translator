import os
import torch
from torch.autograd import Variable
from models.nmt_model import NMT
from options.test_options import TestOptions
from utils.util import *
from utils.batch_iter import MyIterator, batch_size_fn

def token2string(sent, ZH_TEXT):
    string = ""
    for i in sent[1:]:
        sym = ZH_TEXT.vocab.itos[i.item()]
        if sym == "<eos>":
            break
        string += sym
    return string

def translate(test, model, JA_TEXT, ZH_TEXT, out_path):
    model.eval()
    print("Translating test data...")
    result_file = os.path.join(out_path, 'translations.txt')
    for i in range(1, len(test)):
        if i % 10 == 0:
            print(f"Translating the {i}th sentence...")
        sent = test[i].Japanese
        src = torch.LongTensor([[JA_TEXT.vocab.stoi[w] for w in sent]])
        src = Variable(src)
        src_mask = (src != JA_TEXT.vocab.stoi["<pad>"]).unsqueeze(-2)
        out = model.greedy_decode(src, src_mask, max_len=60, start_symbol=ZH_TEXT.vocab.stoi["<sos>"])
        with open(result_file, 'a+', encoding='utf-8') as f:
            message = "Translation: " + token2string(out[0], ZH_TEXT) + "\n"
            gold = ""
            for w in test[i].Chinese:
                gold += w
            message += "{:>11}".format("Gold") + ": " + gold + "\n\n"
            f.write(message)
    print(f"Finish. Results file can be found at {result_file}")

if __name__ == '__main__':
    # parse options
    opt = TestOptions().parse()
    TestOptions.print_options(opt)
    dataroot = opt.dataroot
    expr_name = opt.name
    checkpoints_dir = opt.checkpoints_dir
    batch_size = opt.batch_size
    load_epoch = opt.epoch
    results_dir = opt.results_dir
    
    expr_path = os.path.join(checkpoints_dir, expr_name)
    out_path = os.path.join(results_dir, expr_name)
    mkdir(out_path)
    
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load datasets
    datafields, datasets = load_data(dataroot, isTest=True)
    ZH_TEXT = datafields['Chinese']
    JA_TEXT = datafields['Japanese']
    test = datasets['test']
    
    # model
    model = NMT(len(JA_TEXT.vocab), len(ZH_TEXT.vocab))
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(os.path.join(expr_path, f'{expr_name}_{load_epoch}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    translate(test, model, JA_TEXT, ZH_TEXT, out_path)