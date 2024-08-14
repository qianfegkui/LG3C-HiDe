import torch
import torch.nn as nn
import torch.nn.functional as F
import himallgg

log = himallgg.utils.get_logger()


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        self.nll_loss = nn.NLLLoss()
    
    def get_prob(self, h, text_len_tensor):
        # h_hat = self.emotion_att(h, text_len_tensor)
        # hidden = self.drop(F.relu(self.lin1(h_hat)))
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=-1)

        return log_prob

    def get_fea(self, h, text_len_tensor):
        # h_hat = self.emotion_att(h, text_len_tensor)
        # hidden = self.drop(F.relu(self.lin1(h_hat)))
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        return hidden

    def get_prob1(self, h, text_len_tensor):
        # h_hat = self.emotion_att(h, text_len_tensor)
        # hidden = self.drop(F.relu(self.lin1(h_hat)))
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=-1)
        return scores
    
    def forward(self, h, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)

        return loss
