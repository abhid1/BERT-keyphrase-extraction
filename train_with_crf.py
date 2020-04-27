import argparse
import logging
import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_pretrained_bert import BertModel

from data_loader import DataLoader
import utils
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/task1', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='bert-base-uncased-pytorch', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")


'''
Code borrowed from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
and mixed with https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py#L804
then added to BERT code from https://github.com/pranav-ust/BERT-keyphrase-extraction
'''


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


class BERT_CRF(nn.Module):

    def __init__(self, bert_model, vocab_size, tag_to_idx):
        super(BERT_CRF, self).__init__()
        self.hidden_size = 768
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.num_labels = len(tag_to_idx)

        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)

        # Maps the output of BERT into tag space.
        self.hidden2tag = nn.Linear(self.hidden_size, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.constant_(self.hidden2tag.bias, 0.0)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(params.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # START TAG has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.tag_to_idx[START_TAG]] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, sentence, sentence_mask):
        '''
          sentances -> word embedding -> bert -> MLP -> feats
          '''
        bert_seq_out, _ = self.bert(sentence, token_type_ids=None, attention_mask=sentence_mask,
                                    output_all_encoded_layers=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2tag(bert_seq_out)
        return bert_feats


    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        start = torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long).cuda()
        tags = torch.cat([start, tags.flatten()])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(params.device)
        log_delta[:, 0, self.tag_to_idx[START_TAG]] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(params.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(params.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, sentence, tags, sentence_mask):
        feats = self._get_bert_features(sentence, sentence_mask)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, sentence_mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from BERT
        bert_feats = self._get_bert_features(sentence, sentence_mask)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(bert_feats)
        return score, tag_seq


def train(model, train_data, val_data, optimizer, scheduler, params):
    print("***** Running Training *****")
    for epoch in range(1, params.epoch_num + 1):
        print("Epoch: ", epoch)
        model.train()
        optimizer.zero_grad()

        # a running average object for loss
        loss_avg = utils.RunningAverage()

        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        for batch_data, batch_tags in train_data_iterator:
            # Run our forward pass.
            batch_masks = batch_data.gt(0)
            loss = model.neg_log_likelihood(batch_data, batch_tags, batch_masks).sum()

            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu

            model.zero_grad()
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

            # Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            optimizer.step()
            loss_avg.update(loss.item())

        scheduler.step()

        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        params.eval_steps = params.train_steps
        evaluate(model, train_data_iterator, params, mark='Train', isCRF=True)
        print()
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=True)
        params.eval_steps = params.val_steps
        evaluate(model, val_data_iterator, params, mark='Val', isCRF=True)
        print('--------------------------------------------------------------')
    print()



# def evaluate(model, data_iterator, dataset_name):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_data, batch_tags in data_iterator:
#             batch_masks = batch_data.gt(0)
#             _, predicted_label_seq_ids = model(batch_data, batch_masks)
#             valid_predicted = torch.masked_select(predicted_label_seq_ids, batch_masks)
#             valid_label_ids = torch.masked_select(batch_tags, batch_masks)
#             all_preds.extend(valid_predicted.tolist())
#             all_labels.extend(valid_label_ids.tolist())
#             total += len(valid_label_ids)
#             correct += valid_predicted.eq(valid_label_ids).sum().item()
#
#     test_acc = correct / total
#     precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
#     print(dataset_name)
#     print('Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f' \
#           % (100. * test_acc, 100. * precision, 100. * recall, 100. * f1))
#     return test_acc, f1
#
#
# def f1_score(y_true, y_pred):
#     '''
#     0,1,2,3 are I, O, [CLS], [SEP]
#     '''
#     ignore_id = len(params.tag2idx) - 3
#
#     num_proposed = len(y_pred[y_pred < ignore_id])
#     num_correct = (np.logical_and(y_true == y_pred, y_true < ignore_id)).sum()
#     num_gold = len(y_true[y_true < ignore_id])
#
#     try:
#         precision = num_correct / num_proposed
#     except ZeroDivisionError:
#         precision = 1.0
#
#     try:
#         recall = num_correct / num_gold
#     except ZeroDivisionError:
#         recall = 1.0
#
#     try:
#         f1 = 2 * precision * recall / (precision + recall)
#     except ZeroDivisionError:
#         if precision * recall == 0:
#             f1 = 1.0
#         else:
#             f1 = 0
#
#     return precision, recall, f1


def add_start_stop_idx(tag2idx, idx2tag):
    max_idx = max(list(tag2idx.values()))
    tag2idx[START_TAG] = max_idx + 1
    tag2idx[STOP_TAG] = max_idx + 2

    idx2tag[max_idx + 1] = START_TAG
    idx2tag[max_idx + 2] = STOP_TAG


START_TAG = "[CLS]"
STOP_TAG = "[SEP]"

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(params.device, params.n_gpu, args.fp16))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)

    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')
    test_data = data_loader.load_data('test')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    params.test_size = test_data['size']

    # Add start and stop tag mappings
    add_start_stop_idx(params.tag2idx, params.idx2tag)

    # Prepare model
    bert_model = BertModel.from_pretrained(args.bert_model_dir)
    bert_model.to(params.device)
    crf_model = BERT_CRF(bert_model, params.train_size, params.tag2idx)
    crf_model.to(params.device)

    params.train_steps = params.train_size // params.batch_size
    params.val_steps = params.val_size // params.batch_size
    params.test_steps = params.test_size // params.batch_size

    if args.fp16:
        crf_model.half()

    if params.n_gpu > 1 and args.multi_gpu:
        crf_model = torch.nn.DataParallel(crf_model)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(crf_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(crf_model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=params.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = Adam(optimizer_grouped_parameters, lr=params.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))

    train(crf_model, train_data, val_data, optimizer, scheduler, params)
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=True)
    print("***** Running prediction *****")
    params.eval_steps = params.test_steps
    evaluate(crf_model, test_data_iterator, params, mark='Test')
