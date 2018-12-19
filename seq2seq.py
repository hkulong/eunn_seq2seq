import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.tensorboard.plugins import projector

# useful libraries for data preprocessing
import tensorlayer as tl
from tensorlayer.layers import *

import numpy as np
import time
import os

from gen_data import max_len

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3


# this is the normal seq2seq model
class seq2seq_model():
    def __init__(self,
                vocab_size,
                embedding_size,
                num_layer,
                max_gradient_norm,
                batch_size_num,
                learning_rate,
                dropout):

        self.batch_size = batch_size_num

        with tf.variable_scope('seq2seq') as scope:
            self.encoder_input = tf.placeholder(tf.int32, [None, None])
            self.decoder_output = tf.placeholder(tf.int32, [None, None])
            self.decoder_input = tf.placeholder(tf.int32,[None, None])
            self.target_weight = tf.placeholder(tf.float32, [None, None]) # for training or updating
            
            self.encoder_length = retrieve_seq_length_op2(self.encoder_input)
            self.decoder_length = retrieve_seq_length_op2(self.decoder_output)
            
            batch_size = batch_size_num

            decoder_output = self.decoder_output
            target_weight = self.target_weight
            self.embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
            
            # encoder and decoder share the same weight
            encoder_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
            decoder_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
           
            with tf.variable_scope('encoder'):
                encoder_cell = tf.nn.rnn_cell.LSTMCell(embedding_size)
                encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embedded, 
                    self.encoder_length, dtype=tf.float32)

            with tf.variable_scope('decoder') as decoder_scope:
                # train or evaluate
                decoder_cell = tf.nn.rnn_cell.LSTMCell(embedding_size)

                helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedded, self.decoder_length)
                projection_layer = layers_core.Dense(vocab_size)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)

                output, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope, impute_finished=True)
                logits = output.rnn_output
                
                self.result_train = output.sample_id
                self.decoder_state = decoder_state

                # inference (sample decode)
                helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding, 
                    start_tokens=tf.fill([batch_size], GO_ID), end_token=EOS_ID)
                decoder_sample = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper_sample, encoder_state,
                    output_layer=projection_layer)
                output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_sample, impute_finished=True,
                     scope=decoder_scope)
                self.result_sample = output.sample_id

            params = scope.trainable_variables()

            # update for training
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_output, logits=logits) # max_len * batch
            self.loss_train = tf.reduce_sum(target_weight * cross_entropy) / tf.cast(batch_size, tf.float32)
            self.perplexity = tf.exp(tf.reduce_sum(target_weight * cross_entropy) / tf.reduce_sum(target_weight))

            # gradient clipping
            #optimizer = tf.train.AdamOptimizer(learning_rate)
            #gvs = optimizer.compute_gradients(self.loss_train, params)
            #capped_gvs = [(tf.clip_by_value(grad, -max_gradient_norm, max_gradient_norm), var) for grad, var in gvs]
            #self.opt_train = optimizer.apply_gradients(capped_gvs)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.opt_train = optimizer.minimize(self.loss_train)

            # Save down the summary of the model
            tf.summary.scalar('loss_wif_weight',self.loss_train)
            tf.summary.scalar('perplexity',self.perplexity)
            # Merge all summary into one
            self.summary_op = tf.summary.merge_all()
            
    def train(self, sess, encoder_input, decoder_output, decoder_input, mask):
        feed_dict = {}
        feed_dict[self.encoder_input] = encoder_input
        feed_dict[self.decoder_output] = decoder_output
        feed_dict[self.decoder_input] = decoder_input
        feed_dict[self.target_weight] = mask

        
        perplexity, result, loss, state, _ , summary= sess.run([self.perplexity, self.result_train, self.loss_train, self.decoder_state, self.opt_train, self.summary_op], feed_dict=feed_dict)
        return(perplexity, loss, summary)
    
    def test(self,sess, encoder_input, decoder_output, decoder_input, mask):
        feed_dict = {}
        feed_dict[self.encoder_input] = encoder_input
        feed_dict[self.decoder_output] = decoder_output
        feed_dict[self.decoder_input] = decoder_input
        feed_dict[self.target_weight] = mask

        perplexity, result, loss, stat= sess.run([self.perplexity, self.result_train, self.loss_train, self.decoder_state], feed_dict=feed_dict)
        return(perplexity, loss)

    def inference(self, sess, encoder_input, mode):
        feed_dict = {}
        feed_dict[self.encoder_input] = encoder_input
        result = None
        if mode == 'sample':
            result = sess.run(self.result_sample, feed_dict=feed_dict)
        return result

# functions for data processing
def load_vocab(filename):
    '''
    input : folder name
    output : vocab dictionary in the form of {vocab : index}
    '''
    vocab = {}
    with open(filename, encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab

def sentence2idx(line, vocab, chinese=False):
    '''
    input : whole sentence, vocab dictionary
    output : return a list of sentence with index, e.g. [24,199,256]
    '''
    # return unknown key if the token does not exist in the vocab folder
    if not chinese:
        return [vocab.get(token, UNK_ID) for token in line.split()]
    else:
        return [vocab.get(token, UNK_ID) for token in list(line)]

def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}
    
if __name__ == "__main__":
    batch_size = 64
    
    model_name = 'seq2seq_{}'.format(max_len)
    chinese = False
    inputs = open('data/reverse_{}/train/input.txt'.format(max_len), encoding='utf-8', errors='ignore').read().split('\n')
    targets = open('data/reverse_{}/train/output.txt'.format(max_len), encoding='utf-8', errors='ignore').read().split('\n')
    vocab = load_vocab('data/reverse_{}/train/vocab.txt'.format(max_len))
    
    test_inputs = open('data/reverse_{}/test/input.txt'.format(max_len), encoding='utf-8', errors='ignore').read().split('\n')
    test_targets  = open('data/reverse_{}/test/output.txt'.format(max_len), encoding='utf-8', errors='ignore').read().split('\n')
    #print(inputs[:5])
    #print(targets[:5])

    rev_vocab = get_rev_vocab(vocab)
    trainX = [sentence2idx(x, vocab, chinese) for x in inputs]
    trainY = [sentence2idx(y, vocab, chinese) for y in targets]

    testX = [sentence2idx(x, vocab, chinese) for x in test_inputs]
    testY = [sentence2idx(y, vocab, chinese) for y in test_targets]

    #print(len(trainX))
    #print(trainY[:5])

    # training for the seq2seq model
    seq2seq = seq2seq_model(vocab_size=len(vocab),
                            embedding_size=200,
                            num_layer=4,
                            max_gradient_norm=5,
                            batch_size_num=batch_size,
                            learning_rate=0.0001,
                            dropout=0.5
                            )
    n_epoch = 100
    n_step = int(len(trainX)/batch_size)
    n_test_step = int(len(testX)/batch_size)

    saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
    sess = tf.Session()
    try:
        loader = tf.train.import_meta_graph('{}/model.ckpt.meta'.format(model_name))
        loader.restore(sess, tf.train.latest_checkpoint('{}'.format(model_name)))
        print('load finished')
    except:
        sess.run(tf.global_variables_initializer())
        print('load failed')
    # initialize summary
    summary_writer = tf.summary.FileWriter('{}'.format(model_name),graph=sess.graph)
    with open('{}/metadata.tsv'.format(model_name),'w',encoding='utf-8') as f:
        for word in vocab:
            f.write('{}\n'.format(word))
        f.write('\n')
    # training
    summary_time = 0
    try:
        for epoch in range(n_epoch):
            n_iter = 0
            perplexity, loss = 0, 0
            for X, Y in tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False):
                X = tl.prepro.pad_sequences(X)
                _decoder_output = tl.prepro.sequences_add_end_id(Y, end_id=EOS_ID)
                _decoder_output = tl.prepro.pad_sequences(_decoder_output)

                _decoder_input = tl.prepro.sequences_add_start_id(Y, start_id=GO_ID, remove_last=False)
                _decoder_input = tl.prepro.pad_sequences(_decoder_input)
                _target_weight = tl.prepro.sequences_get_mask(_decoder_output)
                train_perplexity,train_loss,summary = seq2seq.train(sess,X,_decoder_output, _decoder_input, _target_weight)
                summary_writer.add_summary(summary,summary_time)
                perplexity += train_perplexity
                loss += train_loss
                n_iter +=1
                summary_time +=1
                '''
                # generate result during training 
                if n_iter % 100 == 0:
                    results = seq2seq.inference(sess, X ,'sample')
                    for inputs,targets in zip(X[:10],results[:10]):
                        try:
                            tmp_i = [rev_vocab[x] for x in inputs if rev_vocab[x]!='<PAD>']
                            tmp_o = [rev_vocab[x] for x in targets]
                            print('I : {}'.format(" ".join(tmp_i)))
                            print('O : {}'.format(" ".join(tmp_o[:tmp_o.index('<EOS>')])))
                        except:
                            print('error when decoding')
                '''
            test_iter = 0
            test_perplexity, test_loss = 0,0
            for X, Y in tl.iterate.minibatches(inputs=testX, targets=testY, batch_size=batch_size, shuffle=False):
                X = tl.prepro.pad_sequences(X)
                _decoder_output = tl.prepro.sequences_add_end_id(Y, end_id=EOS_ID)
                _decoder_output = tl.prepro.pad_sequences(_decoder_output)

                _decoder_input = tl.prepro.sequences_add_start_id(Y, start_id=GO_ID, remove_last=False)
                _decoder_input = tl.prepro.pad_sequences(_decoder_input)
                _target_weight = tl.prepro.sequences_get_mask(_decoder_output)
                test_perplexity,test_loss = seq2seq.test(sess,X,_decoder_output, _decoder_input, _target_weight)
                test_perplexity += test_perplexity
                test_loss += test_loss
                test_iter += 1

                # generate result during training 
                if test_iter % 10 == 0:
                    results = seq2seq.inference(sess, X ,'sample')
                    for inputs,targets in zip(X[:10],results[:10]):
                        try:
                            tmp_i = [rev_vocab[x] for x in inputs if rev_vocab[x]!='<PAD>']
                            tmp_o = [rev_vocab[x] for x in targets]
                            print('I : {}'.format(" ".join(tmp_i)))
                            print('O : {}'.format(" ".join(tmp_o[:tmp_o.index('<EOS>')])))
                        except:
                            print('error when decoding')

            print("epoch perplexity : %.5f" % (perplexity/n_iter))
            print("epoch loss: %.5f"  % (loss/n_iter))
            print("epoch test perplexity : %.5f" % (test_perplexity/test_iter))
            print("epoch test loss: %.5f"  % (test_loss/test_iter))
            saver.save(sess, '{}/model.ckpt'.format(model_name))
            config = projector.ProjectorConfig()
            embedding_config = config.embeddings.add()
            embedding_config.tensor_name = seq2seq.embedding.name
            embedding_config.metadata_path = 'metadata.tsv'
            projector.visualize_embeddings(summary_writer,config)
    except KeyboardInterrupt:
        saver.save(sess, '{}/model.ckpt'.format(model_name))
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = seq2seq.embedding.name
        embedding_config.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer,config)
