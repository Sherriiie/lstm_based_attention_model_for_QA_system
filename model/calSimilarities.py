# coding:utf-8

import tensorflow as tf


def feature2cos_sim(feat_q, feat_a):
	norm_q = tf.sqrt(tf.reduce_sum(tf.mul(feat_q, feat_q), 1))
	norm_a = tf.sqrt(tf.reduce_sum(tf.mul(feat_a, feat_a), 1))
	mul_q_a = tf.reduce_sum(tf.mul(feat_q, feat_a), 1)
	cos_sim_q_a = tf.div(mul_q_a, tf.mul(norm_q, norm_a))
	return cos_sim_q_a

