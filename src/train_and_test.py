#!/usr/bin/python
from gensim import models
from copy import deepcopy
from math import ceil,floor
from itertools import permutations
import random
import sys, getopt
import numpy as np
from numpy import exp, dot

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("compute.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)s] - %(message)s')
# fh.setFormatter(fm)
# ch.setFormatter(fm)
logger.addHandler(fh)

iter_num = 10
blank_percentage = 0.05
pediction_set_size = 10
window_size = 1

def remove_random_actions(plan):
    blank_count = int(ceil(len(plan) * blank_percentage + 0.5))
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = random.randrange(1, len(plan)-1)
        if missing_action_index in indices:
            # making sure that the indices generated are unique
            continue
        else:
            incomplete_plan[ missing_action_index ] = u'###'
            indices.append(missing_action_index)
            cnt += 1
    return blank_count, indices, incomplete_plan

# p = permutation of actions
# ip = incomplete plan
def getTentativePlan(p, ip, indices):
    for i in range(len(indices)):
        ip[indices[i]] = p[i]
    return ip

# def permuteOverMissingActions(actions, blank_count, indices):
#     ''' Exausts 64 GB of RAM when
#         blank_count >= 3,
#         #( actions ) >= 1250
#     '''
#     action_set = []
#     tentative_plans = []
#     for p in permutations(actions, blank_count):
#      action_set.append(p)
#      tentative_plans.append(getTentativePlan(p, incomplete_plan, indices))
#     return action_set, tentative_plans

# def predictAndVerify(indices, tentative_plans, action_set):
#     for i in range(len(indices)):
#         window_sized_plans = []
#         for tp in tentative_plans:
#             window_sized_plans.append( tp[indices[i]-window_size:indices[i]+window_size+1] )
#         scores = model.score( window_sized_plans )
#         best_indices = scores.argsort()[-1*pediction_set_size:][::-1]
#         for j in best_indices:
#             if action_set[j][i] == plan[indices[i]]:
#                 correct += 1
#                 break;
#     return correct

def min_uncertainty_distance_in_window_size(indices):
    # Makes sure that within a window size there is only one missing action
    # Optimized code from http://stackoverflow.com/questions/15606537/finding-minimal-difference
    if len(indices) <= window_size:
        return 2
    idx = deepcopy(indices)
    idx = sorted(idx)
    res = [ idx[i+window_size]-idx[i] for i in xrange(len(idx)) if i+window_size < len(idx) ]
    return min(res)


def score_sg_grad(model, target_word, current_word, target_weight, current_weight):
    l1 = model.syn0[current_word.index]
    l2a = deepcopy(model.syn1[target_word.point])  # 2d matrix, codelen x layer1_size
    sgn = -1.0**target_word.code  # ch function, 0-> 1, 1 -> -1
    # a = (1 - 1.0 / (1.0 + exp(-sgn*dot( current_weight * l1, target_weight * l2a.T))) )
    # print a.shape
    # b = dot(l2a * l1)
    # print b.shape
    # print dot(l2a, l1).shape
    grads = (1 - 1.0 / (1.0 + exp(-sgn*dot( current_weight * l1, target_weight * l2a.T))) ) * dot(l2a, l1)
    return sum(grads)


def compute_gradient(model, blank_index, sample_word, target_weight, current_weight, incompelete_plan):
    grad = 0.0
    vocab = model.vocab
    current_word = vocab[sample_word]
    context_words = [ vocab[incompelete_plan[blank_index-1]], vocab[incompelete_plan[blank_index+1]] ]
    for target_word in context_words:
        grad += score_sg_grad(model, target_word, current_word, target_weight, current_weight)
        grad += score_sg_grad(model, current_word, target_word, current_weight, target_weight)
    # print grad
    return grad


def train_and_test(domain, shouldTrain, set_number):
    '''
    The function trains a model on training data and then tests the models accuracy on the testing data.
    Since training is time consuming, we save the model and load it later for further testing
    '''

    print "\n=== Set : %s ===\n" % str(set_number)

    # Train a model based on training data
    if shouldTrain == True:
        sentences = models.word2vec.LineSentence(domain+'/train'+str(set_number)+'.txt')
        model = models.Word2Vec(sentences=sentences, min_count=1, workers=4, hs=1, window=window_size, iter=10)
        model.save(domain+'/model'+str(set_number)+'.txt')
    else:
        # OR load a mode
        model = models.Word2Vec.load(domain+'/model'+str(set_number)+'.txt')

    print "Training : COMPLETE!"

    # Evaluate model on test data
    plans = open(domain+'/test'+str(set_number)+'.txt').read().split("\n")
    list_of_actions = [[unicode(actn, "utf-8") for actn in plan_i.split()] for plan_i in plans]
    actions = model.vocab.keys()
    vocab_size = len(actions)
    correct = 0
    total = 0

    print "Testing : RUNNING . . ."
    list_of_actions = [x for x in list_of_actions if len(x) != 0]
    
    for itr in xrange(len(list_of_actions)):
        logger.info("\n\n")
        logger.info("--------------------------------------------------------------")

        plan = list_of_actions[itr]
        # This reduces the combinatorial burst as all permutations do not need to be checked
        # This is the logic used for the paper's (http://rakaposhi.eas.asu.edu/aamas16-hankz.pdf) code
        while True:
            blank_count, indices, incomplete_plan = remove_random_actions(plan)
            if min_uncertainty_distance_in_window_size(indices) > window_size:
                # print "min_uncertainty > window_size"
                break

        # logger.info("incomplete_plan:%s", incomplete_plan)
        # logger.info("complete_plan:%s", plan)
        total += blank_count
        weights = np.ones(vocab_size * blank_count).reshape(vocab_size, blank_count) / vocab_size
        for i in range(iter_num):
            gradients = np.zeros(vocab_size * blank_count).reshape(vocab_size, blank_count)
            for k in range(vocab_size):
                sample_indexs = []
                for blank_order in range(blank_count):
                    index = np.random.choice(np.arange(vocab_size), p=weights[:, blank_order])
                    sample_indexs.append(index)

                # compute gradients
                # print indices
                # print "blank_count:%s" % blank_count
                for blank_order in range(blank_count):
                    blank_index = indices[blank_order]
                    sample_index = sample_indexs[blank_order]
                    sample_word = actions[sample_index]
                    current_weight = weights[sample_index][blank_order]
                    grad = compute_gradient(model, blank_index, sample_word, 1,
                                            current_weight, incomplete_plan)
                    logger.debug("blank_word:%s\tsample_word:%s\tgrad:%f", plan[blank_index], sample_word, grad)
                    if plan[blank_index] == sample_word:
                        logger.debug("*****************************************************")
                        logger.debug("blank_word:%s\tsample_word:%s\tgrad:%f", plan[blank_index], sample_word, grad)
                    # print "blank_word:%s\tsample_word:%s\tgrad:%f" % (actions[blank_index], sample_word, grad)
                    gradients[sample_index][blank_order] += grad
            # update weights
            weights += gradients
            # min-max normalize to 0-1
            mins = np.amin(weights, axis=0)
            maxs = np.amax(weights, axis=0)
            weights = (weights - mins) / (maxs - mins)

            # normalize to distribution
            column_sum = weights.sum(axis=0)
            weights = weights / column_sum[np.newaxis, :]

        logger.info("full weights")
        logger.info("%s", weights)
        logger.info("max")
        logger.info("%s", np.amax(weights, axis=0))
        logger.info("min")
        logger.info("%s", np.min(weights, axis=0))
        logger.info("column sum")
        logger.info("%s", np.sum(weights, axis=0))

        logger.info("best weights")
        logger.info("%s", np.sort(weights, axis=0)[-1*pediction_set_size:][:])
        sorted_weights = np.sort(weights, axis=0)

        best_plan_args = np.argsort(weights, axis=0)[-1*pediction_set_size:][:]
        logger.info("best args")
        logger.info("%s", best_plan_args)
        for i in range(blank_count):
            blank_index = indices[i]
            logger.info("%d blank word:%s", i, plan[blank_index])
            logger.info("predict word\tweights")
            for j in range(pediction_set_size):
                logger.info("%s\t%f", actions[best_plan_args[j][i]], sorted_weights[j+vocab_size-10][i])

        for blank_order in range(blank_count):
            blank_index = indices[blank_order]
            for sample_index in best_plan_args[:][blank_order]:
                if actions[sample_index] == plan[blank_index]:
                    correct += 1
                    break

        # for i in indices:
        #     tentative_plans = []
        #     tentative_actions = []
        #     temp_plan = deepcopy(incomplete_plan)
        #     for a in actions:
        #         temp_plan[i] = a
        #         # tentative_plans.append( temp_plan[ max(0,i-window_size) : min(i+window_size+1,len(plan)) ] )
        #         tentative_plans.append( temp_plan[ max(0,i-window_size) : min(i+window_size+1,len(plan)) ] )
        #         tentative_actions.append(a)
        #     scores = model.score( tentative_plans )
        #     best_plan_args = scores.argsort()[-1*pediction_set_size:][::-1]
        #     for p in best_plan_args:
        #         if tentative_actions[p] == plan[i]:
        #             correct += 1
        #             break

        # Print at certain time intervals
        if (itr*100)/len(list_of_actions) % 10 == 0:
            sys.stdout.write( "\rProgress: %s %%" % str( (itr*100)/len(list_of_actions) ) )
            sys.stdout.flush()

        #action_set, tentative_plans = permuteOverMissingActions(actions, blank_count, indices)
        #correct = predictAndVerify( indices, tentativePlans, action_set)

    sys.stdout.write( "\r\rTesting : COMPLETE!\n")
    sys.stdout.flush()
    print "\nUnknown actions: %s; Correct predictions: %s" % (str(total), str(correct))
    print "Set Accuracy: %s\n" % str( float(correct*100)/total )
    return total, correct

def main(argv):
    #print argv
    domain = argv[0]
    train = True if len(argv)==2 and argv[1]=='t' else False
    k = 10

    print "\n=== Domain : %s ===\n" % domain

    total_unknown_actions = 0
    total_correct_predictions = 0
    for i in range(k):
        ua, cp = train_and_test( domain, train, i )
        total_unknown_actions += ua
        total_correct_predictions += cp

    print "\n==== FINAL STATISTICS ===="
    print "\nTotal unknown actions: %s; Total correct predictions: %s" % (str(total_unknown_actions), str(total_correct_predictions))
    print "ACCURACY: %s\n" % str( float(total_correct_predictions*100)/total_unknown_actions )

if __name__ == "__main__":
    main(sys.argv[1:])

