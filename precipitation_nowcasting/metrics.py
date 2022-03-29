# coding: utf-8
import tensorflow as tf
import numpy as np 

threshold=10


def MARE(y_true, y_pred):
    loss = tf.reduce_mean(tf.abs(y_true - y_pred) / tf.minimum(y_true,y_pred))
    return loss



def TP(y_true, y_pred):
    true=tf.where(y_true > threshold, tf.ones_like(y_true), tf.zeros_like(y_true))
    pred=tf.where(y_pred > threshold, tf.ones_like(y_pred), tf.zeros_like(y_pred))    
    TP = tf.math.count_nonzero(pred * true)
    return TP

def TN(y_true, y_pred):
    true=tf.where(y_true > threshold, tf.ones_like(y_true), tf.zeros_like(y_true))
    pred=tf.where(y_pred > threshold, tf.ones_like(y_pred), tf.zeros_like(y_pred))    
    TN = tf.math.count_nonzero((pred - 1) * (true - 1))
    return TN

def FP(y_true, y_pred):
    true=tf.where(y_true > threshold, tf.ones_like(y_true), tf.zeros_like(y_true))
    pred=tf.where(y_pred > threshold, tf.ones_like(y_pred), tf.zeros_like(y_pred))    
    FP = tf.math.count_nonzero(pred * (true - 1))
    return FP

def FN(y_true, y_pred):
    true=tf.where(y_true > threshold, tf.ones_like(y_true), tf.zeros_like(y_true))
    pred=tf.where(y_pred > threshold, tf.ones_like(y_pred), tf.zeros_like(y_pred))    
    FN = tf.math.count_nonzero((pred - 1) * true)
    return FN

def precision(y_true, y_pred):
    tp=TP(y_true, y_pred)
    fp=FP(y_true, y_pred)
    precision=tp / (tp + fp)
    return  precision

def recall(y_true, y_pred):
    tp=TP(y_true, y_pred)
    fn=FN(y_true, y_pred)
    recall=tp / (tp + fn)
    return  recall

def F1(y_true, y_pred):
    p=precision(y_true, y_pred)
    r=recall(y_true, y_pred)
    return 2*p*r/(p+r)

def accuracy(y_true, y_pred):
    tp=TP(y_true, y_pred)
    fp=FP(y_true, y_pred)
    fn=FN(y_true, y_pred)
    tn=TN(y_true, y_pred)
    return (tp+tn)/(tp+tn+fp+fn)


def weighted_mean(y_true, y_pred):
    diff=tf.math.abs(y_true-y_pred)
    #w=tf.math.multiply(diff,tf.nn.relu(y_true-threshold))
    w1=tf.math.multiply(diff,tf.nn.relu(y_true-threshold))
    w2=tf.math.multiply(diff,tf.nn.relu(y_pred-threshold))
    return tf.math.reduce_mean((w1+w2)/2)
 

def shifted_sigmoid(y_true, y_pred):
    #y_true=tf.math.sigmoid(y_true-threshold)
    #y_pred=tf.math.sigmoid(y_pred-threshold)
    diff=tf.math.abs(y_true-y_pred)
    diff=tf.math.sigmoid(diff)
    return tf.math.reduce_mean(diff)
   


def weighted_sigmoid(y_true, y_pred):
    diff=tf.math.abs(y_true-y_pred)
    wt=tf.math.multiply(diff,tf.math.sigmoid(y_true*.4-16))
    wp=tf.math.multiply(diff,tf.math.sigmoid(y_pred*.4-16))
    max_=tf.math.maximum(wt,wp)
    return tf.math.reduce_mean(max_)



