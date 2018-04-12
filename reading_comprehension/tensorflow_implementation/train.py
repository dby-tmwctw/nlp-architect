# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from __future__ import division
from __future__ import print_function
from contextlib import closing

from random import shuffle

import os
import numpy as np
from utils import *
from layers import Match_LSTM_AnswerPointer

import math
import argparse
import tensorflow as tf
import os
import numpy as np

# parse the command line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_path',
    default='/nfs/site/home/snitturs/squad/code/data/',
    help='enter path for training data')

parser.add_argument('--gpu_id', default="2",
                    help='enter gpu id')

parser.add_argument('--max_para_req', default=300,type=int,
                    help='enter the max length of paragraph')

parser.add_argument('--epochs', default=30,type=int,
                    help='enter the number of epochs')

parser.add_argument('--select_device', default='GPU',
                    help='enter the device to execute on')


parser.add_argument('--train_set_size',default=None, type=int,
                    help='enter the length of training set size')


parser.add_argument('--hidden_size',default=150, type=int,
                    help='enter the number of hidden units')

parser.add_argument('--embed_size',default=300, type=int,
                    help='enter the size of embeddings')

parser.add_argument('--model_dir',default='trained_model', type=str,
                    help='enter path to model(save or restore)')

parser.add_argument('--restore_training',default=False, type=bool,
                    help='Choose whether to restore training from a previously saved model')



parser.add_argument('--batch_size',default=64,type=int,
                     help='enter the batch size')

parser.set_defaults()

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

hidden_size = args.hidden_size
gradient_clip_value = 15
embed_size = args.embed_size
#create a dictionary of all parameters
params_dict = {}
params_dict['batch_size'] = args.batch_size
params_dict['embed_size'] = args.embed_size
params_dict['pad_idx'] = 0
params_dict['hidden_size'] = hidden_size
params_dict['glove_dim'] = 300
params_dict['iter_interval'] = 8000
params_dict['num_iterations'] = 500000
params_dict['max_para']=args.max_para_req
params_dict['epoch_no']=args.epochs
# Set Configs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#paths for data files
path_gen = args.data_path
train_para_file = os.path.join(path_gen + 'squad/train.context')
train_para_ids = os.path.join(path_gen + 'squad/train.ids.context')
train_ques_file = os.path.join(path_gen + 'squad/train.question')
train_ques_ids = os.path.join(path_gen + 'squad/train.ids.question')
answer_file = os.path.join(path_gen + 'squad/train.span')
vocab_file = os.path.join(path_gen + 'squad/vocab.dat')
val_paras_ids = os.path.join(path_gen + 'squad/dev.ids.context')
val_ques_ids = os.path.join(path_gen + 'squad/dev.ids.question')
val_ans_file = os.path.join(path_gen + 'squad/dev.span')
vocab_file = os.path.join(path_gen + 'squad/vocab.dat')

#Create lists for train and validation sets
data_train = create_squad_training(train_para_ids, train_ques_ids, answer_file)
data_dev = create_squad_training(val_paras_ids, val_ques_ids, val_ans_file)

if args.train_set_size == None:
    params_dict['train_set_size']=len(data_train)

# Combine train and dev data
data_total = data_train + data_dev

#obtain maximum length of question
_, max_question = max_values_squad(data_total)
params_dict['max_question'] = max_question

# Load embeddings for vocab
print ('Loading Embeddings')
embeddingz = np.load(
    os.path.join(
        path_gen +
        "squad/glove.trimmed_zeroinit.300.npz"))
embeddings = embeddingz['glove']
print("creating training Set ")
train = get_data_array_squad(params_dict, data_train,set_val='train')
dev = get_data_array_squad(params_dict,data_dev,set_val='val')

#Define Model Graph
with tf.device('/device:'+args.select_device+':0'):
    model = Match_LSTM_AnswerPointer(params_dict, embeddings)


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    init = tf.global_variables_initializer()
    model_saver = tf.train.Saver()

    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')

    train_dir = os.path.join(model_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and args.restore_training and (tf.gfile.Exists(ckpt.model_checkpoint_path)
                 or tf.gfile.Exists(v2_path)):
        model_saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Loading from previously stored session")
    else:
        sess.run(init)

    dev_dict = create_data_dict(dev)

    print ("Begin Training")

    for epoch in range(params_dict['epoch_no']):
        print ("Epoch Number is ", epoch)
        #Shuffle Datset
        shuffle(train)
        train_dict = create_data_dict(train)
        model.run_loop(sess, train_dict, mode='train', dropout=0.6)

        print ("Saving Weights")
        model_saver.save(sess, "%s/best_model.chk" % train_dir)
        print ("\nRunning Validation Test")
        model.run_loop(sess, dev_dict, mode='val', dropout=1)
