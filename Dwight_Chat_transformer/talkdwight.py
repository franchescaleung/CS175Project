import math, copy, sys
import torch
import nltk
nltk.download('wordnet')
from MoveData import *
from Transformer import *
from TalkTrain import *

opt = Options(batchsize=16, device=torch.device("cpu"), epochs=50, 
              lr=0.01, max_len = 25, save_path = 'saved/weights/transformer_custom_weights')
data_iter, infield, outfield, opt = json2datatools(path = 'saved/custompairs.json', opt=opt)
emb_dim, n_layers, heads, dropout = 32, 4, 8, 0.1
dwight = Transformer(len(infield.vocab), len(outfield.vocab), emb_dim, n_layers, heads, dropout)
dwight.load_state_dict(torch.load(opt.save_path))

if __name__ == "__main__":

    while True:
        tell_dwight = input("You > ")
        dwight_reply = talk_to_chloe(tell_dwight, dwight, opt, infield, outfield)
        if ("bye dwight" in tell_dwight or "bye ttyl" in dwight_reply):
            print('Dwight > '+ dwight_reply + '\n')
            break
        else:
            print('Dwight > '+ dwight_reply + '\n')
