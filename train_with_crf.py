import argparse
import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertModel

from data_loader import DataLoader
from evaluate import evaluate
import utils

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
and adapted to BERT code from https://github.com/pranav-ust/BERT-keyphrase-extraction
'''

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BERT_CRF(nn.Module):

    def __init__(self, bert_model, vocab_size, tag_to_idx):
        super(BERT_CRF, self).__init__()
        self.hidden_size = 768
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)

        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)

        # Maps the output of BERT into tag space.
        self.hidden2tag = nn.Linear(self.hidden_size, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(params.device)
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
        # ===================================================== convert bottom to top
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.bert(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long).cuda(), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

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


def add_start_stop_idx(tag2idx, idx2tag):
    max_idx = max(list(tag2idx.values()))
    tag2idx[START_TAG] = max_idx + 1
    tag2idx[STOP_TAG] = max_idx + 2

    idx2tag[max_idx + 1] = START_TAG
    idx2tag[max_idx + 2] = STOP_TAG


START_TAG = "<START>"
STOP_TAG = "<STOP>"

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
    '''Data Shape
    {
        data: [ [id-tensors] ],
        tags: [ [tags-tensors] ]
    }
    '''

    '''Pytorch shape
    [(sentence array, tag array)] 
    '''

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    add_start_stop_idx(params.tag2idx, params.idx2tag)

    # Prepare model
    bert_model = BertModel.from_pretrained(args.bert_model_dir)
    model = BERT_CRF(bert_model, params.train_size, params.tag2idx)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.to(params.device)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(6):
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)

        for batch_data, batch_tags in train_data_iterator:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            batch_masks = batch_data.gt(0)
            loss = model.neg_log_likelihood(batch_data, batch_tags, batch_masks)

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()