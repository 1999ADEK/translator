import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchtext import data, datasets
from models.nmt_model import NMT
from models.layer_utils import *
from optims.NoamOpt import NoamOpt
from options.train_options import TrainOptions
from utils.util import *
from utils.batch_iter import *


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()
    TrainOptions.print_options(opt)
    dataroot = opt.dataroot
    expr_name = opt.name
    checkpoints_dir = opt.checkpoints_dir
    batch_size = opt.batch_size
    load_epoch = opt.epoch
    save_latest_freq = opt.save_latest_freq
    save_epoch_freq = opt.save_epoch_freq
    continue_train = opt.continue_train
    num_epoch = opt.num_epoch
    epoch_count = opt.epoch_count
    
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load datasets
    data_fields, datasets = load_data(dataroot)
    ZH_TEXT = data_fields['Chinese']
    JA_TEXT = data_fields['Japanese']
    train, val = datasets['train'], datasets['val']
    
    # defiine criterion and move to GPU if available
    criterion = LabelSmoothing(size=len(ZH_TEXT.vocab), padding_idx=0, smoothing=0.0)
    if torch.cuda.is_available():
        criterion.cuda()
    
    # model and optimizer
    model = NMT(len(JA_TEXT.vocab), len(ZH_TEXT.vocab))
    if torch.cuda.is_available():
        model.cuda()
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    expr_path = os.path.join(checkpoints_dir, expr_name)
    
    # load state dictionary if continue training
    if continue_train:
        checkpoint = torch.load(os.path.join(expr_path, f'{expr_name}_{load_epoch}.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model_opt._step = checkpoint['model_step']
        model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    train_iter = MyIterator(train, batch_size=batch_size, device=device,
                            repeat=False, sort_key=lambda x: (len(x.Japanese), len(x.Chinese)),
                            batch_size_fn=batch_size_fn, train=True)
    val_iter = MyIterator(val, batch_size=batch_size, device=device,
                            repeat=False, sort_key=lambda x: (len(x.Japanese), len(x.Chinese)),
                            batch_size_fn=batch_size_fn, train=False)
    
    pad_idx = ZH_TEXT.vocab.stoi["<pad>"]
    for epoch in range(num_epoch):
        print(f'------------Epoch {epoch_count + epoch}------------')
        # train for one epoch
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model, SimpleLossCompute(model.generator, criterion, opt=model_opt))
        
        # evaluate on val data
        print('On validation set:')
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in val_iter), model, SimpleLossCompute(model.generator, criterion, opt=model_opt))
        print('Loss:', loss)
        
        # save model
        if epoch % save_latest_freq == 0:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_step': model_opt._step,
                        'optimizer_state_dict': model_opt.optimizer.state_dict()
                        }, os.path.join(expr_path, f'{expr_name}_latest.pt'))
        if epoch % save_epoch_freq == 0:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_step': model_opt._step,
                        'optimizer_state_dict': model_opt.optimizer.state_dict()
                        }, os.path.join(expr_path, f'{expr_name}_{epoch_count+epoch}.pt'))