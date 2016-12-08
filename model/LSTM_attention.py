# coding:utf-8

import tensorflow as tf


def getFeature(input_q, input_a, W):
	h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
	h_a = int(input_a.get_shape()[1])

	reshape_q = tf.expand_dims(input_q, -1)
	pool_q = tf.nn.avg_pool(
		reshape_q,
		ksize=[1, h_q, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID')
	output_q = tf.reshape(pool_q, [-1, w])

	reshape_q = tf.expand_dims(output_q, 1)
	reshape_q = tf.tile(reshape_q, [1, h_a, 1])
	reshape_q = tf.reshape(reshape_q, [-1, w])
	reshape_a = tf.reshape(input_a, [-1, w])

	M = tf.tanh(tf.add(tf.matmul(reshape_q, W['Wqm']), tf.matmul(reshape_a, W['Wam'])))
	M = tf.matmul(M, W['Wms'])

	S = tf.reshape(M, [-1, h_a])
	S = tf.nn.softmax(S)

	S_diag = tf.batch_matrix_diag(S)
	attention_a = tf.batch_matmul(S_diag, input_a)
	attention_a = tf.reshape(attention_a, [-1, h_a, w, 1])

	output_a = tf.nn.avg_pool(
		attention_a,
		ksize=[1, h_a, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID')

	output_a = tf.reshape(output_a, [-1, w])

	return output_q, output_a