
# coding:utf-8

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

## define lstm model and reture related features


# return n outputs of the n lstm cells
def biLSTM(x, hidden_size):
	# biLSTM：
	# 功能：添加bidirectional_lstm操作
	# 参数：
	# 	x: [batch, height, width]   / [batch, step, input]
	# 	hidden_size: lstm隐藏层节点个数
	# 输出：
	# 	output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

	# input transformation
	input_x = tf.transpose(x, [1, 0, 2])
	# input_x = tf.reshape(input_x, [-1, w])
	# input_x = tf.split(0, h, input_x)
	input_x = tf.unpack(input_x)

	# define the forward and backward lstm cells
	lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	output, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)

	# output transformation to the original tensor type
	output = tf.pack(output)
	output = tf.transpose(output, [1, 0, 2])
	return output



# return 1 output of lstm cells after pooling
def getFeature(embedding_q, lstm_hidden_size):
	lstm_out = biLSTM(embedding_q, lstm_hidden_size)        # (batch, step, input)
	height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

	# do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
	lstm_out = tf.expand_dims(lstm_out, -1)
	output = tf.nn.max_pool(
		lstm_out,
		ksize=[1, height, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID')

	output = tf.reshape(output, [-1, width])

	return output