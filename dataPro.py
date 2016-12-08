# coding: utf-8

import numpy as np
import random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def getTxt(filepath):

	l2afile = filepath + 'label.answer'
	trainfile = filepath + 'question.label.train'
	validfile = filepath + 'question.label.valid'
	testfile = filepath + 'question.label.test'
	vocabfile = filepath + 'vocabulary'
	vocabDic = {}
	for line in open(vocabfile):
		line = line.decode('utf-8')  # str to unicode
		idx, word = line.rstrip().split('\t')
		vocabDic[idx] = word

	def idx2word(idxStr):
		sentence = list()
		idx_list = idxStr.strip().split(' ')
		# for idx in idxStr.split(' '):
		for idx in idx_list:

		    sentence.append(vocabDic[idx])
		sentence = ' '.join(sentence)
		return sentence

		# return ' '.join([vocabDic[idx] for idx in idxStr.split(' ')])

	l2aDic = {}
	for line in open(l2afile):
		idx, answer = line.rstrip().split('\t')
		l2aDic[idx] = idx2word(answer)

	def getTestTxt(write_part, readfilename, qaDic):
		qaPartDic = {}
		f = open(filepath + write_part, 'w')
		for line in open(readfilename):
			d, q, ap, an = line.rstrip().split('\t')
			q = idx2word(q)
			a = set([l2aDic[anslabel] for anslabel in ap.strip().split(' ')])
			qaDic[q] = a
			qaPartDic[q] = a

			f.write(q+'\t'+ap+'\t'+an+'\n')
		f.close()
		return qaPartDic

	qaDic = {}          # {question, answer} , question is with String type, answer is with Set type
	qaTrainDic = getTestTxt('train', trainfile, qaDic)
	qaValidDic = getTestTxt('valid', validfile, qaDic)
	qaTestDic = getTestTxt('test', testfile, qaDic)

	return l2aDic, qaDic, qaTrainDic, qaValidDic, qaTestDic


def getVocab(qaTrainDic):
	vocabDic = {'NULL': 0}
	reverseVocabDic = {0: 'NULL'}
	idx = 1
	for q, a in qaTrainDic.items():
		words = q.split(' ')
		for ai in a:
			words += ai.split(' ')
		for word in words:
			word = word.lower()
			if word not in vocabDic:
				vocabDic[word] = idx
				reverseVocabDic[idx] = word
				idx += 1
	return vocabDic, reverseVocabDic


def str2input(qa_str, length, vocab):
	result = qa_str.split(' ')
	for i in range(len(result)):
		word = result[i].lower()
		result[i] = vocab[word] if word in vocab else 0
	if len(result) > length:
		result = result[:length]
	else:
		result += [0] * (length - len(result))
	return result


class DataSet(object):

	def __init__(self, filepath, sequence_length_q, sequence_length_a):

		l2aDic, qaDic, qaTrainDic, qaValidDic, qaTestDic = getTxt(filepath)
		vocab, reVocab = getVocab(qaTrainDic)

		trainQ = qaTrainDic.keys()
		validQ = qaValidDic.keys()
		testQ = qaTestDic.keys()
		trainA = set()
		for a in qaTrainDic.values():
			trainA = trainA | a
		trainA = list(trainA)
		validA = set()
		for a in qaValidDic.values():
			validA = validA | a
		validA = list(validA)
		testA = set()
		for a in qaTestDic.values():
			testA = testA | a
		testA = list(testA)

		self.sequence_length_q = sequence_length_q
		self.sequence_length_a = sequence_length_a
		self.vocab = vocab
		self.reVocab = reVocab
		self.qaDic = qaDic
		self.answerDic = l2aDic
		self.trainQ = trainQ
		self.validQ = validQ
		self.testQ = testQ
		self.trainA = trainA
		self.validA = validA
		self.testA = testA
		self.train_index = 0
		self.train_epoch = 0

		# sherrie
		self.totalA = list(set(trainA)|set(validA)|set(testA))

	def next_batch(self, batch_size):

		if self.train_index == len(self.trainQ):
			# Shuffle the train data
			perm = range(len(self.trainQ))
			random.shuffle(perm)
			self.trainQ = [self.trainQ[new_idx] for new_idx in perm]

			self.train_epoch += 1
			self.train_index = 0
			print 'epoch：', self.train_epoch

		q = self.trainQ[self.train_index]
		ap = self.qaDic[q]
		#sherrie an = random.sample(self.trainA, batch_size)
		an = random.sample(self.totalA, batch_size)
		while len(set(an) | ap) < len(ap)+len(an):
			#sherrie an = random.sample(self.trainA, batch_size)
			an = random.sample(self.totalA, batch_size)
		# print ("Q:" )
		# print q
		# print ("AP:" )
		# for posans in ap:
		# 	print posans
		#
		# print ("AN:" )
		# for negans in an:
		# 	print negans
		q_out = [str2input(q, self.sequence_length_q, self.vocab)]
		ap = [str2input(a, self.sequence_length_a, self.vocab) for a in ap]
		ap_out = ap * (batch_size / len(ap)) + ap[:batch_size % len(ap)]
		an_out = [str2input(a, self.sequence_length_a, self.vocab) for a in an]
		self.train_index += 1
		return np.array(q_out), np.array(ap_out), np.array(an_out)

	def next_test_batch(self, qList, batch_size):
		q = random.sample(qList, 1)[0]
		ap = self.qaDic[q]

		#sherrie an = random.sample(self.testA, batch_size-len(ap))
		an = random.sample(self.totalA, batch_size - len(ap))
		while len(set(an) | ap) < batch_size:
			#sherrie an = random.sample(self.testA, batch_size-len(ap))
			an = random.sample(self.totalA, batch_size - len(ap))
		a = list(ap) + an

		q_out = [str2input(q, self.sequence_length_q, self.vocab)]
		a_out = [str2input(ai, self.sequence_length_a, self.vocab) for ai in a]

		return np.array(q_out), np.array(a_out), len(ap)


	def print_qa(self, batch_test_q, batch_test_a, p_idx):
		sentence_q = list()
		sentence_a = list()
		for idx in batch_test_q[0]:
			sentence_q.append(self.reVocab[idx])
		sentence_q = ''.join(sentence_q)
		for idx in batch_test_a[p_idx]:
			sentence_a.append(self.reVocab[idx])
		sentence_a = ''.join(sentence_a)
		print ('Q: ')
		print  sentence_q
		print ('A: ')
		print sentence_a
		return

if __name__ == '__main__':
	filepath = '/home/sherrie/PycharmProjects/tensorflow/test_char/data/'
	projectpath = '/home/sherrie/PycharmProjects/tensorflow/test_char/'
	datapath =  '/home/sherrie/PycharmProjects/tensorflow/test_char/data/'

	insuranceQA = DataSet(datapath, 30, 240)
	for i in range(5):
		batch_q, batch_ap, batch_an = insuranceQA.next_batch(5)


	filepath = '/home/sherrie/PycharmProjects/MyQA/data/'

	getTxt(filepath)

	data = DataSet(filepath, 30, 150)

	print '全部答案数量为：'+str(len(data.testA))
	print '训练词典中单词个数为：' + str(len(data.vocab))
	print '全部问题个数为：'+str(len(data.qaDic))
	print '训练集问题个数为：'+str(len(data.trainQ)), '; 训练集答案个数为：'+str(len(data.trainA))
	print '验证集问题个数为：' + str(len(data.validQ)), '; 验证集答案个数为：' + str(len(data.validA))
	print '测试集问题个数为：' + str(len(data.testQ)), '; 测试集答案个数为：' + str(len(data.testA))

	q, ap, an = data.next_batch(100)
	print q.shape, ap.shape, an.shape

	for i in range(10):
		testQ, testA, answerLen = data.next_test_batch(data.testQ, 500)
		print testQ.shape, testA.shape, answerLen

	aSet = data.qaDic.values()
	maxLen = 0
	for a in aSet:
		if maxLen < len(a):
			maxLen = len(a)
	print '拥有答案最多的问题的答案数目为：', maxLen








