import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

'''
pack_padded_sequence => pack, pad_packed_sequence => unpack
'''

class Encoder_LSTM(nn.Module):
	def __init__(self, embedding_size, hidden_size, device):
		super(Encoder_LSTM, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.device = device
		self.lstm_embed2hidden = nn.LSTMCell(embedding_size, hidden_size)
		self.lstm_hidden2hidden_fc1 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc2 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc3 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc4 = nn.LSTMCell(hidden_size, hidden_size)


	def forward(self, x):
		packed_x, batch_sizes, sorted_indices, unsorted_indices = x

		output = []
		hx1, cx1, hx2, cx2, hx3, cx3, hx4, cx4 = [], [], [], [], [], [], [], [],
		for i, batch_size in enumerate(batch_sizes):
			fc1_hx, fc1_cx = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device), torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
			fc2_hx, fc2_cx = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device), torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
			fc3_hx, fc3_cx = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device), torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
			fc4_hx, fc4_cx = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device), torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)

			fc1_hx, fc1_cx = self.lstm_embed2hidden(packed_x[i*batch_size : i*batch_size + batch_size], (fc1_hx, fc1_cx))
			fc2_hx, fc2_cx = self.lstm_hidden2hidden_fc2(fc1_hx, (fc2_hx, fc2_cx))
			fc3_hx, fc3_cx = self.lstm_hidden2hidden_fc3(fc2_hx, (fc3_hx, fc3_cx))
			fc4_hx, fc4_cx = self.lstm_hidden2hidden_fc4(fc3_hx, (fc4_hx, fc4_cx))

			output.append(fc4_hx)
			hx1.append(fc1_hx), cx1.append(fc1_cx)
			hx2.append(fc2_hx), cx2.append(fc2_cx)
			hx3.append(fc3_hx), cx3.append(fc3_cx)
			hx4.append(fc4_hx), cx4.append(fc4_cx)

		output = torch.cat(output, dim=0)
		hx1 = torch.cat(hx1, dim=0)
		cx1 = torch.cat(cx1, dim=0)
		hx2 = torch.cat(hx2, dim=0)
		cx2 = torch.cat(cx2, dim=0)
		hx3 = torch.cat(hx3, dim=0)
		cx3 = torch.cat(cx3, dim=0)
		hx4 = torch.cat(hx4, dim=0)
		cx4 = torch.cat(cx4, dim=0)

		output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
		hx1 = PackedSequence(hx1, batch_sizes, sorted_indices, unsorted_indices)
		cx1 = PackedSequence(cx1, batch_sizes, sorted_indices, unsorted_indices)
		hx2 = PackedSequence(hx2, batch_sizes, sorted_indices, unsorted_indices)
		cx2 = PackedSequence(cx2, batch_sizes, sorted_indices, unsorted_indices)
		hx3 = PackedSequence(hx3, batch_sizes, sorted_indices, unsorted_indices)
		cx3 = PackedSequence(cx3, batch_sizes, sorted_indices, unsorted_indices)
		hx4 = PackedSequence(hx4, batch_sizes, sorted_indices, unsorted_indices)
		cx4 = PackedSequence(cx4, batch_sizes, sorted_indices, unsorted_indices)
		state = ((hx1, cx1), (hx2, cx2), (hx3, cx3), (hx4, cx4))

		return output, state


class Decoder_LSTM(nn.Module):
	def __init__(self, embedding_size, hidden_size, device):
		super(Decoder_LSTM, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.device = device
		self.lstm_embed2hidden = nn.LSTMCell(embedding_size, hidden_size)
		self.lstm_hidden2hidden_fc1 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc2 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc3 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc4 = nn.LSTMCell(hidden_size, hidden_size)

	def forward(self, x, state):
		x = x.reshape(-1, self.embedding_size) # batch_size, embedding_size

		fc1_hx, fc1_cx = state[0]
		fc2_hx, fc2_cx = state[1]
		fc3_hx, fc3_cx = state[2]
		fc4_hx, fc4_cx = state[3]

		fc1_hx, fc1_cx = self.lstm_embed2hidden(x, (fc1_hx, fc1_cx))
		fc2_hx, fc2_cx = self.lstm_hidden2hidden_fc2(fc1_hx, (fc2_hx, fc2_cx))
		fc3_hx, fc3_cx = self.lstm_hidden2hidden_fc3(fc2_hx, (fc3_hx, fc3_cx))
		fc4_hx, fc4_cx = self.lstm_hidden2hidden_fc4(fc3_hx, (fc4_hx, fc4_cx))
		
		output = fc4_hx
		state = ((fc1_hx, fc1_cx), (fc2_hx, fc2_cx), (fc3_hx, fc3_cx), (fc4_hx, fc4_cx))

		return output, state


class attn_LSTM(nn.Module):
	def __init__(self, embedding_size, hidden_size, device):
		super(attn_LSTM, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.device = device
		self.lstm_embed2hidden = nn.LSTMCell(embedding_size, hidden_size)
		self.lstm_hidden2hidden_fc1 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc2 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc3 = nn.LSTMCell(hidden_size, hidden_size)
		self.lstm_hidden2hidden_fc4 = nn.LSTMCell(hidden_size, hidden_size)

	def forward(self, x, state):
		x = x.reshape(-1, self.embedding_size)  # batch_size, embedding_size

		fc1_hx, fc1_cx = state[0]
		fc2_hx, fc2_cx = state[1]
		fc3_hx, fc3_cx = state[2]
		fc4_hx, fc4_cx = state[3]

		fc1_hx, fc1_cx = self.lstm_embed2hidden(x, (fc1_hx, fc1_cx))
		fc2_hx, fc2_cx = self.lstm_hidden2hidden_fc2(fc1_hx.clone(), (fc2_hx.clone(), fc2_cx.clone()))
		fc3_hx, fc3_cx = self.lstm_hidden2hidden_fc3(fc2_hx.clone(), (fc3_hx.clone(), fc3_cx.clone()))
		fc4_hx, fc4_cx = self.lstm_hidden2hidden_fc4(fc3_hx.clone(), (fc4_hx.clone(), fc4_cx.clone()))

		output = fc4_hx
		state = ((fc1_hx, fc1_cx), (fc2_hx, fc2_cx), (fc3_hx, fc3_cx), (fc4_hx, fc4_cx))

		return output, state


class Encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, device, num_layers=4):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size #512
		self.num_layers = num_layers #4
		self.device = device
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		self.rnn = Encoder_LSTM(self.hidden_size, self.hidden_size, self.device)

	def init_hidden(self, batch_size):
		return (torch.zeros(4, batch_size,  self.hidden_size, device='cuda'),torch.zeros(4, batch_size, self.hidden_size, device='cuda'))
	
	def forward(self, x):
		""" TO DO: feed the unpacked input x to Encoder """
		inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1)
		embedded = self.embedding(x)
		packed_input = pack(embedded, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
		packed_output, state = self.rnn(packed_input)
		# print(packed_output.data.shape)
		output, outputs_lengths = unpack(packed_output, batch_first=True)

		# print(output.shape)

		hx1, _ = unpack(state[0][0], batch_first=True)
		cx1, _ = unpack(state[0][1], batch_first=True)
		hx2, _ = unpack(state[1][0], batch_first=True)
		cx2, _ = unpack(state[1][1], batch_first=True)
		hx3, _ = unpack(state[2][0], batch_first=True)
		cx3, _ = unpack(state[2][1], batch_first=True)
		hx4, _ = unpack(state[3][0], batch_first=True)
		cx4, _ = unpack(state[3][1], batch_first=True)

		hx1 = hx1.permute(1, 0, 2)[-1]
		cx1 = cx1.permute(1, 0, 2)[-1]
		hx2 = hx2.permute(1, 0, 2)[-1]
		cx2 = cx2.permute(1, 0, 2)[-1]
		hx3 = hx3.permute(1, 0, 2)[-1]
		cx3 = cx3.permute(1, 0, 2)[-1]
		hx4 = hx4.permute(1, 0, 2)[-1]
		cx4 = cx4.permute(1, 0, 2)[-1]

		last_state = ((hx1, cx1), (hx2, cx2), (hx3, cx3), (hx4, cx4))

		return output, last_state, outputs_lengths
	

class Decoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, device):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.device = device
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		self.rnn = Decoder_LSTM(self.hidden_size, self.hidden_size, self.device)
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=1)
		)

	def forward(self, x, state, encoder_output):
		""" TO DO: feed the input x to Decoder """
		x = self.embedding(x)
		output, state = self.rnn(x, state)
		output = self.classifier(output)
		return output, state, output


class AttnDecoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, device, max_length, dropout_p=0.1):
		super(AttnDecoder, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.device = device
		self.max_length = max_length
		self.dropout_p = dropout_p
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=1)
		)

		self.embedding = nn.Embedding(vocab_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.rnn = attn_LSTM(self.hidden_size, self.hidden_size, self.device)
		self.out = nn.Linear(self.hidden_size, self.vocab_size)

	def forward(self, x, state, encoder_outputs, encoder_output_length):
		embedded = self.embedding(x).reshape(-1, self.hidden_size)

		attn_weights4 = F.softmax(
			self.attn(torch.cat((embedded, state[3][0]), dim=1)), dim=1)
		#print(encoder_outputs.shape) # encoder_output = batch, sequence, embed/
		#print(encoder_outputs.permute(0, 2, 1).shape)
		#print(attn_weights4.unsqueeze(2).shape)
		#attn_applied4 = torch.bmm(encoder_outputs.permute(0, 2, 1), attn_weights4.unsqueeze(2))
		# print(attn_weights4.unsqueeze(1).shape)
		# print(encoder_outputs.shape)
		attn_applied4 = torch.bmm(attn_weights4.unsqueeze(1), encoder_outputs)
		output4 = torch.cat((embedded, attn_applied4.reshape(-1, self.hidden_size)), 1)
		output4 = self.attn_combine(output4)

		output4 = F.relu(output4)
		output, hidden = self.rnn(output4, state)
		output = self.classifier(output)
		#print(output.shape)
		encoder_output_length
		attn_weights = attn_weights4
		return output, hidden, attn_weights




