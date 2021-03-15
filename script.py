from pyTorchChatBot import *

if __name__ == "__main__":
	corpus_name = "cornell movie-dialogs corpus"
	corpus = os.path.join("data", corpus_name)
	
	datafile = os.path.join(corpus, "dwight_text_RNN.txt")
	
	
	save_dir = os.path.join("data", "save")
	voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
	

	
	small_batch_size = 5
	batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
	input_variable, lengths, target_variable, mask, max_target_len = batches

	model_name = 'cb_model'
	attn_model = 'dot'
	hidden_size = 500
	encoder_n_layers = 2
	decoder_n_layers = 2
	dropout = 0.1
	batch_size = 64
	checkpoint = None
	checkpoint_iter = 3000
	loadFilename = os.path.join(save_dir, model_name, corpus_name,'{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),'{}_checkpoint.tar'.format(checkpoint_iter))


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
	
	
	# print('Building encoder and decoder ...')
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
	# print('Models built and ready to go!')

	clip = 50.0
	learning_rate = 0.001
	decoder_learning_ratio = 5.0
	n_iteration = 8000
	print_every = 1000
	save_every = 1000
	
	# # Ensure dropout layers are in train mode
	encoder.train()
	decoder.train()
	
	# Initialize optimizers
	# print('Building optimizers ...')
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	if loadFilename:
	    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
	    decoder_optimizer.load_state_dict(decoder_optimizer_sd)
	encoder.eval()
	decoder.eval()
	
	# Initialize search module
	searcher = GreedySearchDecoder(encoder, decoder)
	# searcher = nucleusSampling(encoder, decoder)
	
	# Begin chatting (uncomment and run the following line to begin)
	evaluateInput(encoder, decoder, searcher, voc)