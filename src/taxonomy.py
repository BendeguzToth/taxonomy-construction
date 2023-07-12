import codecs
import time
import argparse
import os
import sys
import datetime



ap = argparse.ArgumentParser(add_help=False)
sys.path.append("..")

TASK_DIR = os.path.split(os.getcwd())[0]

ap.add_argument('--corpus_prefix', default=TASK_DIR+'/corpus/3in1_twodatasets/3in1_twodatasets',
                help='path to the corpus resource')
ap.add_argument('--dataset_prefix', default=TASK_DIR+'/datasets/wn-bo', help='path to the train/test/val/rel data')

ap.add_argument('--model_prefix_file', default='3in1_subseqFeat', help='where to store the result')
#ap.add_argument('--embeddings_file', default='../../Wikipedia_Word2vec/glove.6B.50d.txt', help='path to word embeddings file')

ap.add_argument('--debug', default=False, help='debug or normal run')
ap.add_argument('--trainname', default='train_wnbo_hyper', help='name of training data')
ap.add_argument('--valname', default='dev_wnbo_hyper', help='name of val data')
ap.add_argument('--testname', default='test_wnbo_hyper', help='name of test data')
#args = ap.parse_args()

# dimension parameters
ap.add_argument('--NUM_LAYERS', default=2, help='number of layers of LSTM')
ap.add_argument('--HIST_LSTM_HIDDEN_DIM', default=60, type=int)
ap.add_argument('--POS_DIM', default=4)
ap.add_argument('--DEP_DIM', default=5)
ap.add_argument('--DIR_DIM', default=1)
ap.add_argument('--MLP_HIDDEN_DIM', default=60)
ap.add_argument('--PATH_LSTM_HIDDEN_DIM', default=60, type=int)

# parameters that are OUTDATED. may or may not affect performance
ap.add_argument('--word_dropout_rate', default=0.25, help='replace a token with <unk> with specified probability')
ap.add_argument('--path_dropout_rate', default=0, help='dropout of LSTM path embedding')
ap.add_argument('--no_training', default=False, help='load sample trees for training')
ap.add_argument('--n_rollout_test', type=int, default=5, help='beam search width')
ap.add_argument('--discard_rate', default=0., help='discard a pair w.o path info by discard_rate')
ap.add_argument('--set_max_height', default=False, help='limit the max height of tree')
ap.add_argument('--use_height_ebd', default=False, help='consider the height of each node')
ap.add_argument('--use_history', default=False, help='use history of taxonomy construction')
ap.add_argument('--use_sibling', default=True, help='use sibling signals')
ap.add_argument('--require_info', default=False, help='require there has to be info to infer...')
ap.add_argument('--given_root_train', default=False, help='[outdated]give gold root or not')
ap.add_argument('--given_root_test', default=False, help='[outdated]give gold root or not')
ap.add_argument('--filter_root', default=False, help='[outdated]filter root by term counts')
ap.add_argument('--one_layer', default=False, help='only one layer after pair representation')
ap.add_argument('--update_word_ebd', default=False, help='update word embedding or use fixed pre-train embedding')
ap.add_argument('--use_candidate', default=True, help='use candidates instead of considering all remaining pairs')
ap.add_argument('--height_ebd_dim', default=30)

# model settings
ap.add_argument('--max_paths_per_pair', type=int, default=200,
                help='limit the number of paths per pair. Invalid when loading from pkl')
ap.add_argument('--gamma', default=0.4)
ap.add_argument('--n_rollout', type=int, default=2, help='run for each sample')
ap.add_argument('--actor_lr', default=5e-4, help='learning rate of actor')
ap.add_argument('--critic_lr', default=1e-4, help='learning rate of critic')
ap.add_argument('--choose_max', default=True, help='choose action with max prob when testing')
ap.add_argument('--allow_up', default=True, help='allow to attach some term as new root')
ap.add_argument('--reward', default='edge', choices=['hyper', 'edge', 'binary', 'fragment'])
ap.add_argument('--reward_form', default='diff', choices=['last', 'per', 'diff'])

# ablation parameters
ap.add_argument('--allow_partial', default=True, help='allow only partial tree is built')
ap.add_argument('--use_freq_features', default=True, help='use freq features')
ap.add_argument('--use_features', default=True, help='use surface features')
ap.add_argument('--use_path', default=True, help='use path-based info')
ap.add_argument('--use_xy_embeddings', default=True, help='use word embeddings')
# misc
ap.add_argument('--test_semeval', default=True, help='run tests on semeval datasets')
ap.add_argument('--load_model_file', default=None,
                help='if not None, load model from a file')
ap.add_argument('--load_opt', default=False, help='load opt along with the loaded model')

ap.add_argument('--use_bert', default=False, type=bool)
ap.add_argument("--log_file", default=None)
ap.add_argument("--save_file", default=None)

args = ap.parse_args()
opt = vars(args)

# Load the relations -->train_RL.py (main)
with codecs.open(args.dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = {relation: i for i, relation in enumerate(relations)}

# Load the datasets --->train_RL.py (main)
from lstm_common import load_dataset
if args.debug:
        trainname = '../datasets/wn-bo/train_sample.tsv'
        print( 'Loading the dataset...', trainname, '*' * 10)
        train_set = load_dataset(trainname, relations)
        val_set = load_dataset(trainname, relations)
        test_set = load_dataset(trainname, relations)
else:
        print('calling load dataset from : MCRel_LSTM ---(in else)')
        trainname = '/' + args.trainname + '.tsv'
        valname = '/' + args.valname + '.tsv'
        testname = '/' + args.testname + '.tsv'
        train_set = load_dataset(args.dataset_prefix + trainname, relations)
        print ('Loading the training dataset...', trainname, '*' * 10, 'length',len(train_set))
        val_set = load_dataset(args.dataset_prefix + valname, relations)
        print ('Loading the validation dataset...', valname, '*' * 10, 'length',len(val_set))
        test_set = load_dataset(args.dataset_prefix + testname, relations)
        print ('Loading the test dataset...', testname, '*' * 10, 'length',len(test_set))

y_train = [relation_index[label] for label in train_set.values()]
y_val = [relation_index[label] for label in val_set.values()]
y_test = [relation_index[label] for label in test_set.values()]

from collections import Counter
myC=Counter(train_set.values())
print(Counter(y_train))

dataset_keys=[]
dataset_keys.extend(train_set.keys())
dataset_keys.extend(val_set.keys())
dataset_keys.extend(test_set.keys())

assert(len(dataset_keys) == len(train_set)+len(test_set)+len(val_set))

#add (x, root) to dataset_keys
vocab = set()
for x,y in dataset_keys:
        vocab.add(x)
        vocab.add(y)
dataset_keys += [(term, 'root007') for term in vocab]  #To make each term  a potential root so connecting it to hypernym root


from utils_tree import *
if not args.debug:
        # HERE
        print('Not args',args.debug)
        trees = read_tree_file( TASK_DIR+
            "/datasets/wn-bo/wn-bo-trees-4-11-50-train533-lower.ptb",
            given_root=args.given_root_train, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_val = read_tree_file( TASK_DIR+
            "/datasets/wn-bo/wn-bo-trees-4-11-50-dev114-lower.ptb",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_test = read_tree_file(
            TASK_DIR+"/datasets/wn-bo/wn-bo-trees-4-11-50-test114-lower.ptb",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_semeval = read_edge_files(TASK_DIR+"/datasets/SemEval-2016/original/", given_root=True, filter_root=args.filter_root, allow_up=False)
else:
        trees = read_tree_file( TASK_DIR+
            "/datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_train, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_val = read_tree_file( TASK_DIR+
            "/datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_train, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_test = read_tree_file( TASK_DIR+
            "/datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)
        trees_semeval = read_tree_file( TASK_DIR+
            "/datasets/wn-bo/train_sample.ptb2",
            given_root=args.given_root_test, filter_root=args.filter_root, allow_up=args.allow_up)

print(len(trees))
print(len(trees_val))
print(len(trees_test))

from knowledge_resource import *

# Load the resource (processed corpus)
import sys
from importlib import reload
reload(sys)


# def get_hidden_states(encoded, model, layers):
#     """Push input IDs through model. Stack and sum `layers` (last four by default).
#        Select only those subword token outputs that belong to our word of interest
#        and average them."""
#     with torch.no_grad():
#         output = model(**encoded)
#
#     # Get all hidden states
#     states = output.hidden_states
#     # Stack and sum all requested layers
#     output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
#
#     return output.mean(dim=0)
#
#
# def get_word_vector(sent, tokenizer, model, layers):
#     """Get a word vector by first tokenizing the input sentence, getting all token idxs
#        that make up the word of interest, and then `get_hidden_states`."""
#     encoded = tokenizer.encode_plus(sent, return_tensors="pt")
#     return get_hidden_states(encoded, model, layers)
#
# def get_wv(sent, tokenizer, model, n_layers):
#     # layers = range(-n_layers, 0, -1)
#     layers = [-4, -3, -2, -1] #if layers is None else layers
#     word_embedding = get_word_vector(sent, tokenizer, model, layers)
#     return word_embedding

pickled_file=TASK_DIR+'/pickled_data/preload_data_{}_debug{}.pkl'.format(args.model_prefix_file, args.debug)
print( 'Data loaded from',pickled_file, 'make sure pkl is correct')
(word_vectors, word_index, word_set, dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, dir_inverted_index) = pickle.load(
    open(pickled_file, 'rb'),  encoding='latin1')

if args.use_bert:
    del word_vectors
    with open("../pickled_data/bert.pkl", 'rb') as file:
        word_vectors = pickle.load(file)
    print("Using BERT embedding instead.")
else:
    print("Using default word vectors.")


# terms = set()
# for trees_i in [trees, trees_test, trees_val]:
#     for t in trees_i:
#         for w in t.terms:
#             terms.add(w)
#
# # index_to_word = {i: n for n, i in word_index.items()}
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
# placeholder = torch.zeros((420120, 768), dtype=torch.float32)
#
# for term in terms:
#     wv = get_wv(term, tokenizer, model, n_layers=4)
#     placeholder[word_index[term]] = wv





print('Number of words %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(word_index), len(pos_index), len(dep_index), len(dir_index)))


X_train = dataset_instances[:len(train_set)]
X_val = dataset_instances[len(train_set):len(train_set) + len(val_set)]
X_test = dataset_instances[len(train_set) + len(val_set):]
print (len(X_train), len(X_val), len(X_test))

from model import *

# check_data(train_set, X_train, word_set)
    # check_data(val_set, X_val, word_set)
    # check_data(test_set, X_test, word_set)
    # save_path_info(dataset_keys, dataset_instances)
    # scores_save = []
    # scores_save_test = []
    # prob_save = []
    # prob_save_test = []

policy = Policy(dataset_keys, dataset_instances, num_lemmas=len(word_index), num_pos=len(pos_index),
                    num_dep=len(dep_index), num_directions=len(dir_index), opt=opt, num_relations=len(relations),
                    lemma_embeddings=word_vectors)
trainer = dy.AdamTrainer(policy.model, alpha=args.actor_lr)
c_trainer = dy.AdamTrainer(policy.critic_model, alpha=args.critic_lr)

if args.debug:
        n_epoch = 3
else:
        n_epoch = 2

best = [0] * 6
best_idx = [0] * 6
best_val = [0] * 6
best_val_idx = [0] * 6
best_test = [0] * 6
best_test_idx = [0] * 6
best_semeval = [0] * 6
best_semeval_idx = [0] * 6
policy_save_test = defaultdict(list)
wrong_total_l = []

import numpy as np
from collections import defaultdict
from itertools import count
import pickle
from tqdm import tqdm

from lstm_common import get_paths, vectorize_path, get_id
from features import get_all_features

from utils_tree import copy_tree


def load_paths_and_word_vectors(corpus, dataset_keys, lemma_index, keys, string_paths=None):
    # Define the dictionaries
    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    _ = pos_index['#UNKNOWN#']
    _ = dep_index['#UNKNOWN#']
    _ = dir_index['#UNKNOWN#']

    # Vectorize tha paths
    # check for valid utf8 GB
    # keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    # keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in tqdm(dataset_keys)]
    print ('Get paths...')
    if string_paths is None:
        string_paths = [get_paths(corpus, x_id, y_id).items() for (x_id, y_id) in tqdm(keys)]
        if not args.debug:
            print ('saving string_paths...')
            pickle.dump(string_paths, open('pickled_data/string_paths_{}.pkl'.format(args.model_prefix_file), 'wb'))

    # Limit number of paths
    if args.max_paths_per_pair > 0:
        string_paths = [sorted(curr_paths, key=lambda x: x[1], reverse=True)[:args.max_paths_per_pair] for curr_paths in
                        string_paths]

    paths_x_to_y = [{vectorize_path(path, lemma_index, pos_index, dep_index, dir_index): count
                     for path, count in curr_paths}
                    for curr_paths in string_paths]
    paths = [{p: c for p, c in paths_x_to_y[i].iteritems() if p is not None} for i in range(len(keys))]

    # Get the word embeddings for x and y (get a lemma index)
    print( 'Getting word vectors for the terms...')
    x_y_vectors = [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    print ('Getting features for x y...')
    hyper2hypo = pickle.load(open('../datasets/SemEval-2016/candidates_taxi/all_freq_twodatasets.pkl', 'rb'))
    hypo2hyper = defaultdict(lambda: defaultdict(int))
    for hyper in hyper2hypo:
        for hypo in hyper2hypo[hyper]:
            hypo2hyper[hypo][hyper] = hyper2hypo[hyper][hypo]
    lower2original = pickle.load(open('pickled_data/lower2original.pkl', 'rb'))
    features = [get_all_features(x, y, sub_feat=True, hyper2hypo=hyper2hypo, hypo2hyper=hypo2hyper,
                                 lower2original=lower2original) for (x, y) in tqdm(dataset_keys)]

    pos_inverted_index = {i: p for p, i in pos_index.iteritems()}
    dep_inverted_index = {i: p for p, i in dep_index.iteritems()}
    dir_inverted_index = {i: p for p, i in dir_index.iteritems()}

    dataset_instances = list(zip(paths, x_y_vectors, features))
    return dataset_instances, dict(pos_index), dict(dep_index), dict(dir_index), \
           pos_inverted_index, dep_inverted_index, dir_inverted_index


def check_error(name, v):
    if np.any(np.isnan(v.npvalue())) or np.any(np.isinf(v.npvalue())):
        print (name, v.npvalue())
    else:
        print( name, 'looks good [check_error]')


def check_error_np(name, v):
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        print (name, v)


def get_micro_f1(micro_total):
    if micro_total[0] == 0:
        return 0
    prec = micro_total[0] / micro_total[1]
    rec = micro_total[0] / micro_total[2]
    return round(2 * prec * rec / (prec + rec), 3)


def find_top_k(T_rollout, prob_per, pairs_per, k):
    prob_per = np.vstack(prob_per)
    pairs_per = np.vstack(pairs_per)
    # two-dim indices of those with higher prob
    indices_flat = np.argsort(prob_per.ravel())
    indices_pairs = indices_flat[-k:]
    indices = np.dstack(np.unravel_index(indices_flat, (len(prob_per), len(prob_per[0]))))[0][-k:]
    pair_from_tree_idx = [idx[0] for idx in indices]
    new_T_rollout = []
    idx_used = set()
    for idx in indices:
        # use original tree once
        if idx[0] not in idx_used:
            new_T_rollout.append(T_rollout[idx[0]])
            idx_used.add(idx[0])
        else:
            new_T_rollout.append(copy_tree(T_rollout[idx[0]], 1, nolist=True))
    prob_total = prob_per[indices[:, 0], indices[:, 1]]
    selected_pairs = pairs_per[indices_pairs]
    return prob_total, new_T_rollout, selected_pairs, pair_from_tree_idx


def test(epoch, trees_test, policy, policy_save_test, best_test, best_test_idx):
    if epoch % 10 == 0:
        pass
    metric_total = [0] * 6
    micro_total = [0.] * 3
    wrong_at_total = [0.] * 10
    policy.disable_dropout()
    for i_episode in tqdm(range(len(trees_test))):
        dy.renew_cg()
        policy.re_init()
        # prob_l = []
        T = trees_test[i_episode]
        T_rollout = copy_tree(T, min(args.n_rollout_test, (len(T.terms) - 1) * 2))  # a list of T's copy
        policy.init_history(args.n_rollout_test)

        for i in range(len(T.terms) - 1):
            if i == 0:
                prob, pairs = policy.selection_by_tree(T, mode='test')
                prob = dy.log(prob).npvalue()
                indices = np.argsort(prob)[-args.n_rollout_test:]
                prob_total = prob[indices]
                selected_pairs = [pairs[idx] for idx in indices]
                pair_from_tree_idx = [0] * len(T_rollout)
            else:
                prob_per = []
                pairs_per = []
                for T_idx in range(len(T_rollout)):
                    prob, pairs = policy.selection_by_tree(T_rollout[T_idx], mode='test', idx=T_idx)
                    prob = dy.log(prob) + prob_total[T_idx]
                    prob_per.append(prob.npvalue())
                    pairs_per.append(pairs)
                prob_total, T_rollout, selected_pairs, pair_from_tree_idx = find_top_k(T_rollout, prob_per, pairs_per,
                                                                                       args.n_rollout_test)
            for tree_idx, (tree_i, pair_i, from_idx) in enumerate(zip(T_rollout, selected_pairs, pair_from_tree_idx)):
                pair_i = tuple(pair_i)
                tree_i.update(pair_i)
                policy.update_history(pair_i, from_idx=from_idx, to_idx=tree_idx)
        # best candidate
        metric_total, micro_total, wrong_at_total, wrong_total = T.evaluate(metric_total, micro_total,
                                                                            wrong_at_total, reward_type='print')
        # if args.debug:
        #     for tmp_T in T_rollout:
        #         tmp_total = [0] * 6
        #         print tmp_T.evaluate(data=tmp_total, reward_type='print')
        # T.re_init()
    for m_idx in range(5):
        metric_total[m_idx] = round(metric_total[m_idx] / len(trees_test), 3)
    for w_idx in range(len(wrong_at_total)):
        wrong_at_total[w_idx] = round(wrong_at_total[w_idx] / len(trees_test), 3)
    best_test, best_test_idx = update_best(metric_total, best_test, best_test_idx, epoch)
    if epoch % 1 == 0:
        print( '[test]epoch {}:{} {} {} {}'.format(epoch, metric_total, micro_total, get_micro_f1(micro_total),
                                                  wrong_at_total),)
        print( 'best_test', best_test, best_test_idx)

    return policy_save_test, best_test, best_test_idx


def get_vocabulary(corpus, dataset_keys, path_lemmas=None):
    '''
    Get all the words in the dataset and paths
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :return: a set of distinct words appearing as x or y or in a path
    '''
    print ('   word -> id ...')
    keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in tqdm(dataset_keys)]
    print ('   path_lemmas ...')
    if path_lemmas is None:
        path_lemmas = set([edge.split('/')[0]
                           for (x_id, y_id) in tqdm(keys)
                           for path in get_paths(corpus, x_id, y_id).keys()
                           for edge in path.split('_')
                           if x_id > 0 and y_id > 0])
    print( '   x_y_words ...')
    x_y_words = set([x for (x, y) in dataset_keys]).union([y for (x, y) in tqdm(dataset_keys)])
    return path_lemmas, x_y_words, keys
    # return list(path_lemmas.union(x_y_words))


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")


def sample_check(trees_test):
    sample_T = np.random.choice(trees_test)
    print( sample_T.taxo)
    print (sample_T.taxo_test)
    sample_T.evaluate(reward_type=args.reward, output=True)


def update_best(metric, best, best_i, epoch):
    for i in range(len(metric)):
        if metric[i] > best[i]:
            best[i] = metric[i]
            best_i[i] = epoch
    return best, best_i


def check_data(train_set, X_train, word_set):
    # how many samples have entity/path embeddings
    ebd_flag_train = []
    for i, (x, y) in enumerate(train_set.keys()):
        ebd_flag_train.append([])
        if x in word_set:
            ebd_flag_train[-1].append(1)
        if y in word_set:
            ebd_flag_train[-1].append(2)
        if len(X_train[i][0]) != 0:
            ebd_flag_train[-1].append(4)
    print("xy*={}, **p={}, x*p={}, *yp={}, xyp={} / {}".format(sum([sum(i) == 3 for i in ebd_flag_train]),
                                                               sum([sum(i) == 4 for i in ebd_flag_train]),
                                                               sum([sum(i) == 5 for i in ebd_flag_train]),
                                                               sum([sum(i) == 6 for i in ebd_flag_train]),
                                                               sum([sum(i) == 7 for i in ebd_flag_train]),
                                                               len(train_set)))


def check_limit(trees, policy, unk):
    scores_hyper = []
    scores_edge = []
    for T in trees:
        for hypo, hyper in T.taxo.items():
            if hyper == 'root007':
                continue
            if (hypo in unk or hyper in unk) and len(policy.get_data(hypo, hyper)[0]) == 0:
                continue
            T.update((hypo, hyper), test=True)
        scores_hyper.append(T.evaluate(reward_type='hyper'))
        scores_edge.append(T.evaluate(reward_type='edge'))
        T.re_init()
    print (scores_hyper)
    print (np.mean(scores_hyper))
    print (scores_edge)
    print (np.mean(scores_edge))


def save_path_info(dataset_keys, dataset_instances):
    # assert len(dataset_keys) == len(dataset_instances)
    pair2nPath = {}
    for k, v in zip(dataset_keys, dataset_instances):
        pair2nPath[k] = len(v[0])
    print( 'saving num of paths with term-pairs as keys...')
    pickle.dump(pair2nPath, open('pair2nPath.pkl', 'wb'))


def select_action(tree, policy, choose_max=False, return_prob=False, mode='train'):
    prob, pairs, pr__, (input_layers, _, _, _, _), q, cs_in, idx_prop = policy.selection_by_tree(tree, mode)
    if pairs is None:
        if return_prob:
            return None, None, None, None
        else:
            return None, None, None
    with np.errstate(all='raise'):
        try:
            prob_v = prob.npvalue()
            if choose_max:
                idx = np.argmax(prob_v)
            else:
                # if np.random.random() < policy.epsilon:
                #     idx = np.random.randint(len(prob_v))
                #     while prob_v[idx] == 0:
                #         idx = np.random.randint(len(prob_v))
                # else:
                idx = np.random.choice(range(len(prob_v)), p=prob_v / np.sum(prob_v))
        except:
            print(f"probs={prob_v}, sum={np.sum(prob_v)}, pr={pr__.npvalue()}")
            for para in policy.model_parameters:
                check_error(para, dy.parameter(policy.model_parameters[para]))
            check_error('history', policy.history.output())
            check_error('pr', prob)
    action = prob[int(idx)] #NG: made int
    policy.saved_actions[-1].append(action)
    policy.update_history(pairs[idx])
    if return_prob:
        return pairs[idx], prob_v[idx], pairs, prob_v
    return pairs[idx], prob_v[idx], dy.mean_elems(dy.cmult(prob, dy.log(prob))), q, cs_in, idx, idx_prop, input_layers


def train(epoch, trees, policy, trainer, c_trainer, best, best_idx, wrong_total_l, target_update_metronome: Metronome):
    # hyper edge fragment hyper-prec hyper-recall root
    f1 = []
    metric_total = [0] * 6
    micro_total = [0.] * 3
    wrong_at_total = [0.] * 10
    np.random.shuffle(trees)
    loss = 0
    policy.set_dropout(args.path_dropout_rate)
    critic_gamma = 0.9
    for i_episode in tqdm(range(len(trees))):
        T = trees[i_episode]
        # print(f"CLEANINF LSS")
        lss = []
        entropy_l = []
        dy.renew_cg()
        policy.re_init()
        for _ in range(args.n_rollout):
            previous_reward = None
            previous_q = None
            # prob_l = []
            policy.init_history()
            policy.rewards.append([])
            policy.saved_actions.append([])
            policy.next_q_estimate.append([])
            policy.current_q_value.append([])
            policy.cs.append([])
            counter = 0
            crap =[]
            while len(T.V) > 0:
                counter += 1
                pair, pr, entropy, q_, (c1, c2), action_idx, idx, input_layers = select_action(T, policy, choose_max=False, mode='train')
                if pair is None:
                    break
                entropy_l.append(entropy)
                # prob_l.append(pr)
                T.update(pair)
                if args.reward_form != 'last' or len(T.V) == 0:
                    reward = T.eval(reward_type=args.reward, reward_form=args.reward_form)
                else:
                    reward = 0
                policy.rewards[-1].append(reward)
                policy.cs[-1].append((c1, c2, input_layers, action_idx))
                current_q_estimate = policy.target_forward(c1, c2, idx=idx)[int(action_idx)].npvalue().item()
                if len(policy.next_q_estimate[-1]) > 0:
                    policy.next_q_estimate[-1][-1] = current_q_estimate
                policy.next_q_estimate[-1].append(0.)

                q = q_[int(action_idx)]
                policy.current_q_value[-1].append(q)

            metric_total, micro_total, wrong_at_total, wrong_total = T.evaluate(metric_total, micro_total,
                                                                                wrong_at_total, reward_type='print')
            wrong_total_l.append(wrong_total)
            # scores_save.append(T.evaluate(reward_type=REWARD, return_all=True))
            # prob_save.append(prob_l)
            f1.append(T.eval('edge', 'last'))
            T.re_init()
        loss += finish_episode(policy, trainer, entropy_l, lss, c_trainer)
        # print(f"episode finished")
    # del lss
    for m_idx in range(5):
        metric_total[m_idx] = round(metric_total[m_idx] / len(trees) / args.n_rollout, 3)
    metric_total[0] = T.f1_calc(metric_total[3], metric_total[4])
    for w_idx in range(len(wrong_at_total)):
        wrong_at_total[w_idx] = round(wrong_at_total[w_idx] / len(trees) / args.n_rollout, 3)
    metric_total[5] /= args.n_rollout
    best, best_idx = update_best(metric_total, best, best_idx, epoch)
    if epoch % 1 == 0:
        print (' (In train function) [train]epoch {}:{} {} {} {}'.format(epoch, metric_total, micro_total, get_micro_f1(micro_total),
                                                   wrong_at_total),)
        print ('(In train function) total_loss', loss, 'best', best, best_idx)
    f1 = np.array(f1).mean()
    print(f"Training F1 score: {f1}")
    return best, best_idx, f1


def helper(self, c1, c2, input_layers, action_idx):
    W1_rl = dy.parameter(self.model_parameters['W1_rl'])
    b1_rl = dy.parameter(self.model_parameters['b1_rl'])
    if not self.opt['one_layer']:
        W2_rl = dy.parameter(self.model_parameters['W2_rl'])
        b2_rl = dy.parameter(self.model_parameters['b2_rl'])

    # pr = W2_rl * dy.rectify(W1_rl * dy.concatenate_to_batch(input_layers) + b1_rl) + b2_rl
    # (V x N)x160 160x50 50x60 60x1

    c1_w1 = dy.parameter(self.critic_params['c1_w1'])
    c1_w2 = dy.parameter(self.critic_params['c1_w2'])
    c1_b1 = dy.parameter(self.critic_params['c1_b1'])
    c1_b2 = dy.parameter(self.critic_params['c1_b2'])

    c2_w1 = dy.parameter(self.critic_params['c2_w1'])
    c2_w2 = dy.parameter(self.critic_params['c2_w2'])
    c2_b1 = dy.parameter(self.critic_params['c2_b1'])
    c2_b2 = dy.parameter(self.critic_params['c2_b2'])

    q_w1 = dy.parameter(self.critic_params['q_w1'])
    q_b1 = dy.parameter(self.critic_params['q_b1'])
    # q_w2 = dy.parameter(self.model_parameters['q_w2'])
    # q_b2 = dy.parameter(self.model_parameters['q_b2'])

    # Here do the critic forward pass.
    while True:
        if self.opt['use_history']:
            c1_out = c1 * dy.rectify(c1_w1 * dy.rectify(c1_w2 * self.history[0].output() + c1_b2) + c1_b1)
            c2_out = c2 * dy.rectify(c2_w1 * dy.rectify(c2_w2 * self.history[0].output() + c2_b2) + c2_b1)
        else:
            c1_out = dy.rectify(c1 * c1_w1 + c1_b1) * c1_w2 + c1_b2
            c2_out = dy.rectify(c2 * c2_w1 + c2_b1) * c2_w2 + c2_b2

        c_out = dy.concatenate_cols([c1_out, c2_out])

        q = c_out * q_w1 + q_b1
        if not np.isnan(q.npvalue()).any():
            break

    if not self.opt['one_layer']:
        if self.opt['use_history']:
            while True:
                pr = input_layers * dy.rectify(W2_rl * dy.rectify(W1_rl * self.history[0].output() + b1_rl) + b2_rl)
                if not np.isnan(pr.npvalue()).any():
                    break
        else:
            while True:
                pr = dy.rectify(input_layers * W2_rl + b2_rl) * W1_rl + b1_rl
                if not np.isnan(pr.npvalue()).any():
                    break

    else:
        if self.opt['use_history']:
            pr = input_layers * dy.rectify(W1_rl * self.history[0].output() + b1_rl)
        else:
            pr = input_layers * W1_rl + b1_rl
    # (#actions, )
    pr = dy.reshape(pr, (pr.npvalue().shape[0],))
    pr = dy.softmax(pr)[int(action_idx)]
    q = q[int(action_idx)]

    return pr, q


def finish_episode(policy, trainer, entropy_l, lss, c_trainer):
    loss = []
    critic_gamma = 0.95
    all_cum_rewards = []
    for ct, p_rewards in enumerate(policy.rewards):
        R = 0
        rewards = []
        for r in p_rewards[::-1]:
            R = r + policy.gamma * R
            rewards.insert(0, r)  # rewards is list of returns (cumulative rewards) for each time step.
        all_cum_rewards.append(rewards)  # Holds return sequences for each rollout.
        rewards = np.array(rewards) - policy.baseline_reward  # Subtract baseline from rewards.
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)  # Normalize rewards.
        for _, reward, _, next_q, (c1, c2, input_layers, action_idx) in zip(policy.saved_actions[ct], rewards, policy.current_q_value[ct], policy.next_q_estimate[ct], policy.cs[ct]):
            while True:
                action, q = helper(policy, c1, c2, input_layers, action_idx)
                critic_tgt = reward + critic_gamma * next_q
                critic_loss = (critic_tgt - q) ** 2
                actor_loss = -dy.log(action) * q.npvalue().item()
                combined_loss = (critic_loss + actor_loss) / 2
                if not np.isnan(combined_loss.npvalue()).any():
                    break
            combined_loss.backward()
            loss.append(combined_loss.scalar_value())
            try:
                trainer.update()
                c_trainer.update()
            except RuntimeError:
                print("oof v2")
            # loss.append(combined_loss)
    # loss = dy.average(loss) + policy.decaying_beta * dy.average(entropy_l)
    # while True:
    #     loss = dy.average(loss)
    #     if not np.isnan(loss.npvalue()).any():
    #         break
    # loss.backward()
    try:
    #     trainer.update()
    #     c_trainer.update()
        policy.update_baseline(np.mean(all_cum_rewards))
    except RuntimeError:
        # print(policy.rewards)
        # for actions in policy.saved_actions:
        #     for action in actions:
        #         print(action.npvalue())
        print(f"oof")
    policy.update_global_step()
    policy.update_eps()
    policy.refresh_target()
    return np.average(loss)


def test_single(epoch, trees_test, policy, policy_save_test, best_test, best_test_idx, wrong_total_l,
                reward_type='print'):
    metric_total = [0] * 6
    micro_total = [0.] * 3
    wrong_at_total = [0.] * 10
    # if args.debug and epoch % 100 == 0:
    #     for T in trees_test:
    #         policy_save_test[T.rootname].append([])
    # elif not args.debug and epoch % 1 == 0:
    #     for T in trees_test:
    #         policy_save_test[T.rootname].append([])

    policy.disable_dropout()
    f1 = []
    height_l = []
    for i_episode in range(len(trees_test)):
        dy.renew_cg()
        policy.re_init()
        # prob_l = []
        T = trees_test[i_episode]
        policy.init_history()
        policy.rewards.append([])
        policy.saved_actions.append([])
        if args.allow_up:
            n_time = len(T.terms) - 1
        else:
            n_time = len(T.terms)
        if reward_type == 'print_each':
            for _ in range(n_time):
                pair, pr, pairs, prob = select_action(T, policy, choose_max=args.choose_max, return_prob=True,
                                                      mode='test')
                if args.allow_partial and pair is None:
                    break
                T.update(pair)
                print ("in test single: if rewardtype=print each",pair, pr,)
                # T.evaluate(output=True)
        else:
            for _ in range(n_time):
                pair, pr, pairs, prob = select_action(T, policy, choose_max=args.choose_max, return_prob=True,
                                                      mode='test')
                if pair is None:
                    break
                T.update(pair)
                # print ("in test single: else",pair, pr,)   <- Uncomment to see pairs during testing. Prints way too much.
                # T.evaluate(output=True)
            T.permute_ancestor()
        metric_total, micro_total, wrong_at_total, wrong_total = T.evaluate(metric_total, micro_total,
                                                                            wrong_at_total, reward_type=reward_type)
        wrong_total_l.append(wrong_total)
        height_l.append(T.cur_height)
        # T.draw()
        # T.save_for_vis(i_episode)
        # pickle.dump(T, open('{}.tree.pkl'.format(T.filename), 'wb'))`
        f1.append(T.eval('edge', 'last'))
        T.re_init()
        # scores_save_test.append(T.evaluate(reward_type=REWARD, return_all=True))
        # prob_save_test.append(prob_l)
    # sample_check(trees_test)
    for m_idx in range(5):
        metric_total[m_idx] = round(metric_total[m_idx] / len(trees_test), 3)
    metric_total[0] = T.f1_calc(metric_total[3], metric_total[4])
    for w_idx in range(len(wrong_at_total)):
        wrong_at_total[w_idx] = round(wrong_at_total[w_idx] / len(trees_test), 3)
    if metric_total[0] > 0.56 and args.load_model_file is None:
        policy.save_model('(In test_single), model_{}_epoch{}_{}'.format(args.model_prefix_file, epoch, metric_total[0]))
    best_test, best_test_idx = update_best(metric_total, best_test, best_test_idx, epoch)
    if epoch % 1 == 0:
        print ('(In test_single) [test]epoch {}:{} {} {} {}'.format(epoch, metric_total, micro_total, get_micro_f1(micro_total),
                                                  wrong_at_total),)
        print ('(In test_single) best_test', best_test, best_test_idx, np.mean(height_l), np.max(height_l), np.min(height_l))

    # pickle.dump((scores_save, scores_save_test, [], [], policy_save_test),
    #             open(score_filename, 'wb'))
    f1 = np.array(f1).mean()
    print(f"Test f1: {f1}")
    return policy_save_test, best_test, best_test_idx, f1


# TRAIN / TEST START HERE
n_epoch = 10000
metronome = Metronome(32)
if args.load_model_file is None:
    for epoch in range(n_epoch):
        print('epoch', epoch)
        start_t = time.time()
        best, best_idx, train_f1 = train(epoch, trees, policy, trainer, c_trainer, best, best_idx, wrong_total_l, target_update_metronome=metronome)
        _, best_val, best_val_idx, test_f1 = test_single(epoch, trees_val, policy, [], best_val, best_val_idx, wrong_total_l)
        end_t = time.time()

        duration = end_t - start_t
        minutes = int(duration // 60)
        seconds = int(duration - minutes * 60)

        if args.log_file is not None:
            now = datetime.datetime.now()
            with open(args.log_file, 'a') as file:
                file.write(
                    f"epoch {epoch + 1} | {train_f1:.3f}, {test_f1:.3f} | {minutes}:{seconds} | {now.strftime('%m-%d %H:%M:%S')}\n")
        # writer.add_scalar('edge-f1', train_f1, epoch)
        # writer.add_scalar('validation-edge-f1', train_f1, epoch)

        if args.save_file is not None:
            policy.save_model(args.save_file)

        # policy_save_test, best_test, best_test_idx = test_single(epoch, trees_test, policy, policy_save_test,
        #                                                          best_test, best_test_idx, wrong_total_l)
        # print('Train/Test time (test_single)','policy_save_test',policy_save_test,'best_test',best_test,'best_test_idx',best_test_idx)

# else:
#     load_candidate_from_pickle(trees_semeval)
#     _, best_semeval, best_semeval_idx = test_single(0, trees_semeval, policy, [], best_semeval,
#                                                     best_semeval_idx, wrong_total_l,
#                                                     reward_type='print_each')