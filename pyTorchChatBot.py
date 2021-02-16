from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from vocab import *
from toTensor import *
from Models import *


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def printLines(file, n=10):
	with open(file, 'rb') as datafile:
		lines = datafile.readlines()
	for line in lines[:n]:
		print(line)

def loadLines(fileName, fields):
	lines = {}
	with open(fileName, 'r', encoding='iso-8859-1') as f:
		for line in f:
			values = line.split(" +++$+++ ")
			# Extract fields
			lineObj = {}
			for i, field in enumerate(fields):
				lineObj[field] = values[i]
			lines[lineObj['lineID']] = lineObj
	return lines

def loadConversations(fileName, lines, fields):
	conversations = []
	with open(fileName, 'r', encoding='iso-8859-1') as f:
		for line in f:
			values = line.split(" +++$+++ ")
			# Extract fields
			convObj = {}
			for i, field in enumerate(fields):
				convObj[field] = values[i]
			# Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
			utterance_id_pattern = re.compile('L[0-9]+')
			lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
			# Reassemble lines
			convObj["lines"] = []
			for lineId in lineIds:
				convObj["lines"].append(lines[lineId])
			conversations.append(convObj)
	return conversations

def extractSentencePairs(conversations):
	qa_pairs = []
	for conversation in conversations:
		# Iterate over all the lines of the conversation
		for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
			inputLine = conversation["lines"][i]["text"].strip()
			targetLine = conversation["lines"][i+1]["text"].strip()
			# Filter wrong samples (if one of the lists is empty)
			if inputLine and targetLine:
				qa_pairs.append([inputLine, targetLine])
	return qa_pairs


if __name__ == '__main__':
	
	corpus_name = "cornell movie-dialogs corpus"
	corpus = os.path.join("data", corpus_name)
	
	# Define path to new file
	datafile = os.path.join(corpus, "formatted_movie_lines.txt")
	
	delimiter = '\t'
	# Unescape the delimiter
	delimiter = str(codecs.decode(delimiter, "unicode_escape"))

	# Initialize lines dict, conversations list, and field ids
	lines = {}
	conversations = []
	MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
	MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

	# Load lines and process conversations
	print("\nProcessing corpus...")
	lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
	print("\nLoading conversations...")
	conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

	# Write new csv file
	print("\nWriting newly formatted file...")
	with open(datafile, 'w', encoding='utf-8') as  outputfile:
		writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
		for pair in extractSentencePairs(conversations):
			writer.writerow(pair)

	# Print a sample of lines
	# print("\nSample lines from file:")
	# printLines(datafile)

	# Load/Assemble voc and pairs
	save_dir = os.path.join("data", "save")
	voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
	# Print some pairs to validate
	# print("\npairs:")
	# for pair in pairs[:10]:
	#     print(pair)

	# Trim voc and pairs
	pairs = trimRareWords(voc, pairs, MIN_COUNT)

	# # Example for validation
	# small_batch_size = 5
	# batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
	# input_variable, lengths, target_variable, mask, max_target_len = batches

	# print("input_variable:", input_variable)
	# print("lengths:", lengths)
	# print("target_variable:", target_variable)
	# print("mask:", mask)
	# print("max_target_len:", max_target_len)

	# Configure models
	model_name = 'cb_model'
	attn_model = 'dot'
	#attn_model = 'general'
	#attn_model = 'concat'
	hidden_size = 500
	encoder_n_layers = 2
	decoder_n_layers = 2
	dropout = 0.1
	batch_size = 64
	
	# Set checkpoint to load from; set to None if starting from scratch
	loadFilename = None
	checkpoint_iter = 4000
	#loadFilename = os.path.join(save_dir, model_name, corpus_name,
	#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
	#                            '{}_checkpoint.tar'.format(checkpoint_iter))


	# Load model if a loadFilename is provided
	if loadFilename:
	    # If loading on same machine the model was trained on
	    checkpoint = torch.load(loadFilename)
	    # If loading a model trained on GPU to CPU
	    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
	    encoder_sd = checkpoint['en']
	    decoder_sd = checkpoint['de']
	    encoder_optimizer_sd = checkpoint['en_opt']
	    decoder_optimizer_sd = checkpoint['de_opt']
	    embedding_sd = checkpoint['embedding']
	    voc.__dict__ = checkpoint['voc_dict']
	
	
	print('Building encoder and decoder ...')
	# Initialize word embeddings
	embedding = nn.Embedding(voc.num_words, hidden_size)
	if loadFilename:
	    embedding.load_state_dict(embedding_sd)
	# Initialize encoder & decoder models
	encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
	decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
	if loadFilename:
	    encoder.load_state_dict(encoder_sd)
	    decoder.load_state_dict(decoder_sd)
	# Use appropriate device
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	print('Models built and ready to go!')

	clip = 50.0
	teacher_forcing_ratio = 1.0
	learning_rate = 0.0001
	decoder_learning_ratio = 5.0
	n_iteration = 8000
	print_every = 1
	save_every = 500
	
	# Ensure dropout layers are in train mode
	encoder.train()
	decoder.train()
	
	# Initialize optimizers
	print('Building optimizers ...')
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	if loadFilename:
	    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
	    decoder_optimizer.load_state_dict(decoder_optimizer_sd)
	
	# If you have cuda, configure cuda to call
	for state in encoder_optimizer.state.values():
	    for k, v in state.items():
	        if isinstance(v, torch.Tensor):
	            state[k] = v.cuda()
	
	for state in decoder_optimizer.state.values():
	    for k, v in state.items():
	        if isinstance(v, torch.Tensor):
	            state[k] = v.cuda()
	
	# Run training iterations
	print("Starting Training!")
	trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename)