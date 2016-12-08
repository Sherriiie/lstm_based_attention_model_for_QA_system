# -*- coding: UTF-8 -*-

import datetime
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os
import dataPro
import model.LSTM as QA_LSTM
import model.calSimilarities as calSim
import model.LSTM_attention as QA_LSTM_attention
import model.LSTM_attention2 as QA_LSTM_attention2
import xlrd

margin_value = 0.1				# 0.1
embedding_size = 100			# 100
lstm_hidden_size = 141 			# 141
attention_matrix_size = 100  #50 #20       # 141*2

batch_size = 100
test_batch_size = 500       # ratio
valid_batch_size = 100      # 2000
sequence_length_q = 30
sequence_length_a = 240
num_epochs = 38800    # 48800 # 194001
evaluate_every = 1000
learning_rate = 0.1    # 0.1


#=====================
checkpoint_every = 1000
save_loss_every = 1000

# attention params
W = {
	'Wam': tf.Variable(tf.truncated_normal([2 * lstm_hidden_size, attention_matrix_size], stddev=0.1)),
	'Wqm': tf.Variable(tf.truncated_normal([2 * lstm_hidden_size, attention_matrix_size], stddev=0.1)),
	'Wms': tf.Variable(tf.truncated_normal([attention_matrix_size, 1], stddev=0.1))
}


W2 = {
	'Wam': tf.Variable(tf.truncated_normal([lstm_hidden_size, attention_matrix_size], stddev=0.1)),
	'Wqm': tf.Variable(tf.truncated_normal([2 * lstm_hidden_size, attention_matrix_size], stddev=0.1)),
	'Wms': tf.Variable(tf.truncated_normal([attention_matrix_size, 1], stddev=0.1))
}

#=====================

projectpath = '/home/sherrie/PycharmProjects/tensorflow/test_char/'
datapath = os.path.join(projectpath, 'data/')
checkpoint_dir = os.path.join(projectpath, 'checkpoint/')

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

insuranceQA = dataPro.DataSet(datapath, sequence_length_q, sequence_length_a)

vocab_size = len(insuranceQA.vocab)
print '词典大小为：', vocab_size


########################################
## input part
input_q = tf.placeholder(tf.int32, [None, sequence_length_q], name='input_q')
input_ap = tf.placeholder(tf.int32, [None, sequence_length_a], name='input_ap')
input_an = tf.placeholder(tf.int32, [None, sequence_length_a], name='input_an')

# test input
test_q = tf.placeholder(tf.int32, [None, sequence_length_q], name='test_q')
test_a = tf.placeholder(tf.int32, [None, sequence_length_a], name='test_a')

with tf.device('/cpu:0'), tf.name_scope("embedding"):
	embedding_w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
	embedding_q = tf.nn.embedding_lookup(embedding_w, input_q)
	embedding_ap = tf.nn.embedding_lookup(embedding_w, input_ap)
	embedding_an = tf.nn.embedding_lookup(embedding_w, input_an)
	embedding_test_q = tf.nn.embedding_lookup(embedding_w, test_q)
	embedding_test_a = tf.nn.embedding_lookup(embedding_w, test_a)

embedding_q = tf.tile(embedding_q, [batch_size, 1, 1])
embedding_test_q = tf.tile(embedding_test_q, [test_batch_size, 1, 1])


############################################
## model LSTM

with tf.variable_scope('LSTM_model', reuse=None):
	pool_q = QA_LSTM.getFeature(embedding_q, lstm_hidden_size)

with tf.variable_scope('LSTM_model', reuse=True):
	pool_ap = QA_LSTM.getFeature(embedding_ap, lstm_hidden_size)
	pool_an = QA_LSTM.getFeature(embedding_an, lstm_hidden_size)
	pool_test_q = QA_LSTM.getFeature(embedding_test_q, lstm_hidden_size)
	pool_test_a = QA_LSTM.getFeature(embedding_test_a, lstm_hidden_size)



###############################################
## model LSTM-attention
'''
with tf.variable_scope('LSTM_model', reuse=None):
	lstm_q = QA_LSTM.biLSTM(embedding_q, lstm_hidden_size)
with tf.variable_scope('LSTM_model', reuse=True):
	lstm_ap = QA_LSTM.biLSTM(embedding_ap, lstm_hidden_size)
	lstm_an = QA_LSTM.biLSTM(embedding_an, lstm_hidden_size)
	lstm_test_q = QA_LSTM.biLSTM(embedding_test_q, lstm_hidden_size)
	lstm_test_a = QA_LSTM.biLSTM(embedding_test_a, lstm_hidden_size)
pool_q, pool_ap = QA_LSTM_attention.getFeature(lstm_q, lstm_ap, W)
pool_q, pool_an = QA_LSTM_attention.getFeature(lstm_q, lstm_an, W)
pool_test_q, pool_test_a = QA_LSTM_attention.getFeature(lstm_test_q, lstm_test_a, W)
'''


###############################################
## model LSTM-attention2
'''
with tf.variable_scope('LSTM_model', reuse=None):
	pool_q = QA_LSTM.getFeature(embedding_q, lstm_hidden_size)
with tf.variable_scope('LSTM_model', reuse=True):
	pool_ap = QA_LSTM_attention2.getFeature(pool_q, embedding_ap, W2)
	pool_an = QA_LSTM_attention2.getFeature(pool_q, embedding_an, W2)

	pool_test_q = QA_LSTM.getFeature(embedding_test_q, lstm_hidden_size)
	pool_test_a = QA_LSTM_attention2.getFeature(pool_test_q, embedding_test_a, W2)
'''


###############################################
## computes similarities
cos_sim_q_ap = calSim.feature2cos_sim(pool_q, pool_ap)
cos_sim_q_an = calSim.feature2cos_sim(pool_q, pool_an)
zero = tf.zeros_like(cos_sim_q_ap)
margin = margin_value * tf.ones_like(cos_sim_q_ap)
losses = tf.maximum(zero, tf.sub(margin, tf.sub(cos_sim_q_ap, cos_sim_q_an)))
loss = tf.reduce_sum(losses)
correct = tf.equal(zero, losses)
accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

# model evaluation
cos_sim_q_a = calSim.feature2cos_sim(pool_test_q, pool_test_a)
pre_index = tf.argmax(cos_sim_q_a, 0, name='pre_index')

## optimization

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
saver = tf.train.Saver(tf.all_variables(), reshape=True)
# Initializing the variables
init = tf.initialize_all_variables()

# f = open(checkpoint_dir + 'lossText.txt', 'w')
# f.close()
f = open(checkpoint_dir + 'lossText.txt', 'a')
f2 = open(checkpoint_dir + 'accText.txt', 'a')

# launch the grapp
with tf.Session() as sess:

	sess.run(init)

	# saver.restore(sess, checkpoint_dir + 'acc_0.68-48000')
	# print ("restored, begin to train")
	trainLossList = []
	accList = [0]
	all_train_los = 0

	for step in range(0, num_epochs):
		batch_q, batch_ap, batch_an = insuranceQA.next_batch(batch_size)
		los, _ = sess.run([loss, optimizer], feed_dict={input_q: batch_q,
		                                                input_ap: batch_ap,
		                                                input_an: batch_an})
		print ('=' * 10 + 'step{}, loss = {}'.format(step, float(los / batch_size)))
		all_train_los += los
		if step % save_loss_every == 0:
			los_avg = all_train_los/(save_loss_every*batch_size)
			print step, 'Train Loss: ', los_avg
			trainLossList.append(los_avg)
			all_train_los = 0
			# path = saver.save(sess, checkpoint_dir + 'loss_' + str(los_avg), global_step=step)
			# print("Saved model checkpoint to {}\n".format(path))
			f.write("step " + str(step) + " los_avg=" + str(los_avg) + '\n')

		if step%evaluate_every ==0:
		# if current_epoch < insuranceQA.train_epoch:
		# 	current_epoch = insuranceQA.train_epoch
			acc = 0
			for i in range(1, valid_batch_size+1):
				batch_test_q, batch_test_a, answerLen = \
					insuranceQA.next_test_batch(insuranceQA.validQ, test_batch_size)
				p_idx = sess.run(pre_index, feed_dict={test_q: batch_test_q,
													   test_a: batch_test_a})
				acc += 1 if p_idx < answerLen else 0

				# time_str = datetime.datetime.now()
				# time_str = time_str.strftime("%Y-%m-%d %H:%M:%S")
				# print time_str, i

			time_str = datetime.datetime.now()
			time_str = time_str.strftime("%Y-%m-%d %H:%M:%S")
			print("{}  Step {}, Validation Accuracy = {:.3f}".format(time_str, step, 1.0 * acc / i))

			f2.write("step " + str(step) + " acc=" + str(1.0 * acc / i)+ '\n')

			if (1.0 * acc / i)>max(accList):
				path = saver.save(sess, checkpoint_dir + 'acc_' + str(1.0 * acc / i), global_step=step)
				print("Saved model checkpoint to {}\n".format(path))
			accList.append(1.0 * acc / i)

	x1 = range(len(trainLossList))
	# f = open(checkpoint_dir + 'lossText.txt', 'w')
	for i in x1:
		f.write(str(i) + '\t' + str(trainLossList[i]) + '\n')
	f.close()

	x2 = range(len(accList))
	# f = open(checkpoint_dir + 'accText.txt', 'w')
	for i in x2:
		f2.write(str(i) + '\t' + str(accList[i]) + '\n')
	f2.close()
	print ("PROPRAM FINISHED! ")
