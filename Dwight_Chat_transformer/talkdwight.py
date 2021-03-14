#chatbot adapted from this tutorial: https://github.com/clam004/chat-transformer
# MoveData, Transform, and TalkTrain are all code used directly from the above tutorial

import math, copy, sys
import torch
import nltk
nltk.download('wordnet')
from MoveData import *
from Transformer import *
from TalkTrain import *

opt = Options(batchsize=16, device=torch.device("cpu"), epochs=50, lr=0.01, max_len = 25, save_path = 'saved/weights/transformer_custom_weights') #initialize our options for the chatbot
data_iter, infield, outfield, opt = json2datatools(path = 'saved/custompairs.json', opt=opt) #make out infield/outfield vocabulary from our custom query/response pairings
emb_dim, n_layers, heads, dropout = 16, 4, 8, 0.1 #won't directly be used except in training, but needs to be defined for chatbot
dwight = Transformer(len(infield.vocab), len(outfield.vocab), emb_dim, n_layers, heads, dropout) #initialize the chatbot with its vocabulary
dwight.load_state_dict(torch.load(opt.save_path)) #load weights and options into chatbot

if __name__ == "__main__":

    while True: #initialize while loop
        tell_dwight = input("You > ") #get input from user
        dwight_reply = talk_to_chloe(tell_dwight, dwight, opt, infield, outfield) #get response from chatbot, #talk to chloe is a 
        if ("bye dwight" in tell_dwight or "bye" in dwight_reply): #check to see if user wants to exit or dwight thinks its the end of the conversation
            print('Dwight > '+ dwight_reply + '\n') #print response and exit
            break
        else:
            print('Dwight > '+ dwight_reply + '\n') #if not exit, just print response and loop
