"""
This module defines the naive bayes method for text classification without call any pre-defined models.

Author:
    Hailiang Zhao
"""
import numpy as np
import re
import random
import feedparser
import operator


def create_dataset():
    post_lists = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    abused = [0, 1, 0, 1, 0, 1]
    return post_lists, abused


def creat_vocab(dataset):
    """
    Create the vocabulary from the posts.

    :param dataset: the input posts
    :return: the vocabulary
    """
    vocab_set = set([])
    for sentence in dataset:
        # the Union of two sets
        vocab_set = vocab_set | set(sentence)
    return list(vocab_set)


def words2vec(vocab_list, input_set):
    """
    Convert the words (in a sentence) to a 0-1 vector by detecting that whether the
    word is in the sentence.

    :param vocab_list:
    :param input_set:
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('The word \'%s\' is not in vocabulary!' % word)
    return return_vec


def words2bag(vocab_list, input_set):
    return_bag = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_bag[vocab_list.index(word)] += 1
        else:
            print('The word \'%s\' is not in vocabulary!' % word)
    return return_bag


def get_conditional_prob(features, labels):
    """
    For the abusive word detection problem, calculate the conditional probability by
    Bayes Theorem.

    :param features: the vectors of sentences
    :param labels: the class of the sentence (abusive or not)
    :return:
    """
    train_instances_num = len(features)
    words_num = len(features[0])
    # abusive stores p(c_1)
    abusive = sum(labels) / len(labels)
    # we do not use zeros
    abuse_per_word, non_abuse_per_word = np.ones(words_num), np.ones(words_num)
    abuse_words_num, non_abuse_words_num = 2., 2.
    for i in range(train_instances_num):
        if labels[i] == 1:
            abuse_per_word += features[i]
            abuse_words_num += sum(features[i])
        else:
            non_abuse_per_word += features[i]
            non_abuse_words_num += sum(features[i])
    # element-wise (we use log() to avoid underflow)
    # abuse_probs stores p(w_i | c_1) for each word i (the label is abusive)
    abuse_probs = np.log(abuse_per_word / abuse_words_num)
    # non_abuse_probs stores p(w_i | c_0) for each word i (the label is non-abusive)
    non_abuse_probs = np.log(non_abuse_per_word / non_abuse_words_num)
    return non_abuse_probs, abuse_probs, abusive


def classify(input_feature, non_abuse_probs, abuse_probs, abusive):
    abused = sum(input_feature * abuse_probs) + np.log(abusive)
    non_abused = sum(input_feature * non_abuse_probs) + np.log(1. - abusive)
    if abused > non_abused:
        return 1
    else:
        return 0


def abusive_words_test():
    my_dataset, my_labels = create_dataset()
    my_vocab_list = creat_vocab(my_dataset)
    print(my_vocab_list)
    matrix = []
    for sentence in my_dataset:
        matrix.append(words2vec(my_vocab_list, sentence))
    p0, p1, pa = get_conditional_prob(matrix, my_labels)
    test_entry = ['love', 'my', 'dalmation']
    test_feature = np.array(words2vec(my_vocab_list, test_entry))
    print(test_entry, 'is classified as:', classify(test_feature, p0, p1, pa))
    test_entry = ['stupid', 'garbage']
    test_feature = np.array(words2vec(my_vocab_list, test_entry))
    print(test_entry, 'is classified as:', classify(test_feature, p0, p1, pa))


def txt_parse(sentences):
    words_list = re.split(r'\W*', sentences)
    return [word.lower() for word in words_list if len(word) > 2]


def spam_email_test():
    """
    The function provides an example to classify emails into spam and ham by Naive Bayes.

    :return: overall error rate
    """
    # doc_list and labels_list are the same with the output of create_dataset()
    # full_txt is a one-dimensional list where every word stores
    doc_list, labels_list, full_txt = [], [], []
    for i in range(1, 26):
        words_list = txt_parse(open('../../../dataset/email/spam/%d.txt' % i).read())
        doc_list.append(words_list)
        full_txt.extend(words_list)
        labels_list.append(1)

        words_list = txt_parse(open('../../../dataset/email/ham/%d.txt' % i).read())
        doc_list.append(words_list)
        full_txt.extend(words_list)
        labels_list.append(0)
    vocab_list = creat_vocab(doc_list)

    # 10 for test, 40 for train
    train_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_idx = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_idx])
        del(train_set[rand_idx])

    matrix, labels = [], []
    for doc_idx in train_set:
        matrix.append(words2vec(vocab_list, doc_list[doc_idx]))
        labels.append(labels_list[doc_idx])
    p0, p1, p_spam = get_conditional_prob(np.array(matrix), np.array(labels))
    error_count = 0.
    for doc_idx in test_set:
        input_feature = words2vec(vocab_list, doc_list[doc_idx])
        classified = classify(np.array(input_feature), p0, p1, p_spam)
        print(doc_list[doc_idx], 'is classified as', classified, ',its label is', labels_list[doc_idx])
        if classify(np.array(input_feature), p0, p1, p_spam) != labels_list[doc_idx]:
            error_count += 1
    print('The error rate is: %f%%' % (float(error_count) / len(test_set) * 100))


def get_most_freq(vocab_list, full_txt):
    """
    Calculate frequency of occurrence.

    :param vocab_list:
    :param full_txt:
    :return: the top 30 frequent words
    """
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_txt.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def rss_feed_test():
    """
    This function provides an example of RSS feed classifier by Naive bayes.
    ===> Problem exist! The website for parsing is useless! <===

    :return:
    """
    rss0 = feedparser.parse('https://newyork.craigslist.org/d/computer-services/search/stn/cps')
    rss1 = feedparser.parse('https://sfbay.craigslist.org/d/computer-services/search/cps')
    # doc_list and labels_list are the same with the output of create_dataset()
    # full_txt is a one-dimensional list where every word stores
    doc_list, labels_list, full_txt = [], [], []
    min_len = min(len(rss1['entries']), len(rss0['entries']))
    for i in range(min_len):
        words_list = txt_parse(rss1['entries'][i]['summary'])
        doc_list.append(words_list)
        full_txt.extend(words_list)
        labels_list.append(1)

        words_list = txt_parse(rss0['entries'][i]['summary'])
        doc_list.append(words_list)
        full_txt.extend(words_list)
        labels_list.append(0)
    vocab_list = creat_vocab(doc_list)
    most_freq_words = get_most_freq(vocab_list, full_txt)
    for word_with_count in most_freq_words:
        if word_with_count[0] in vocab_list:
            vocab_list.remove(word_with_count[0])

    train_set = list(range(2 * min_len))
    test_set = []
    for i in range(20):
        rand_idx = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_idx])
        del(train_set[rand_idx])

    matrix, labels = [], []
    for doc_idx in train_set:
        matrix.append(words2bag(vocab_list, doc_list[doc_idx]))
        labels.append(labels_list[doc_idx])
    p0, p1, p_spam = get_conditional_prob(np.array(matrix), np.array(labels))
    error_count = 0.
    for doc_idx in test_set:
        input_feature = words2bag(vocab_list, doc_list[doc_idx])
        classified = classify(np.array(input_feature), p0, p1, p_spam)
        print(doc_list[doc_idx], 'is classified as', classified, ',its label is', labels_list[doc_idx])
        if classify(np.array(input_feature), p0, p1, p_spam) != labels_list[doc_idx]:
            error_count += 1
    print('The error rate is: %f%%' % (float(error_count) / len(test_set) * 100))
    return vocab_list, p0, p1


if __name__ == '__main__':
    rss_feed_test()
