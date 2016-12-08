# coding:utf-8

import tensorflow as tf
# from tensorflow.python.ops import rnn, rnn_cell
import model.rnn2 as rnn
import model.rnn_cell2 as rnn_cell

## Method 2 of bilstm-attention: feed attention weights to cells before lstm calculation is done.


def biLSTM_attention(input_q, inputs_a, W=None):     # input_q is pool_q (B, 2h), input_a (B, Sa, E)

	w = int(input_q.get_shape()[1])
	hidden_size = w/2
	h_a = int(inputs_a.get_shape()[1])

	inputs_a = tf.transpose(inputs_a, [1,0,2])
	inputs_a = tf.unpack(inputs_a)        # Sa*(B, E)


	# 参数：
	# 	x: [batch, height, width]   / [batch, step, input]
	# 	hidden_size: lstm隐藏层节点个数
	# 输出：
	# 	output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

	# input transformation

	# define the forward and backward lstm cells
	lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	# print ("type",type(lstm_fw_cell))
	lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	output, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs_a, dtype=tf.float32, q=input_q, W_att=W)

	# output transformation to the original tensor type
	output = tf.pack(output)
	output = tf.transpose(output, [1, 0, 2])
	return output

# return 1 output of lstm cells after pooling
def getFeature(pool_q, inputs_a, W=None):

	lstm_out_a = biLSTM_attention(pool_q, inputs_a, W)  # (batch, step, input)
	height, width = int(lstm_out_a.get_shape()[1]), int(
		lstm_out_a.get_shape()[2])  # (step, length of input for one step)

	# do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
	lstm_out_a = tf.expand_dims(lstm_out_a, -1)
	pool_a = tf.nn.max_pool(
		lstm_out_a,
		ksize=[1, height, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID')
	pool_a = tf.reshape(pool_a, [-1, width])
	return pool_a



