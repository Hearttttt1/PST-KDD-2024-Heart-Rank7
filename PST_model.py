import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.nn.parameter import Parameter
import numpy as np 
import json
import torch.nn.init as init
from types import SimpleNamespace
##################################################################################

################################    Loss    ##################################

##################################################################################

# criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor, weights=None):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )


def compute_kl_loss(self, p, q, pad_mask=None):
    '''
    使用 r_dropout 
    参考微信文章: Pytorch框架少样本情况下效果增强方法实现
    '''

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2

    return loss



##################################################################################

################################    Poolings    ##################################

##################################################################################

class MeanPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MeanPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings



class MeanMaxPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MeanMaxPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask 

        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        return mean_max_embeddings


class GeMText(nn.Module):
    def __init__(self, dim=1, cfg=None, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1
        # x seeems last hidden state

    def forward(self, x, attention_mask, input_ids, cfg):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, features):
        ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({'token_embeddings': weighted_average})
        return features



class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()



    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q.to("cuda"), h.transpose(-2, -1).to("cuda")).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0).to("cuda"), v_temp.to("cuda")).squeeze(2)
        return v

##################################################################################

################################    Model    ##################################

##################################################################################

class NLPPoolings:
    _poolings = {
        # "All [CLS] token": NLPAllclsTokenPooling,
        "GeM": GeMText,
        "Mean": MeanPooling,
        # "Max": MaxPooling,
        # "Min": MinPooling,
        "MeanMax": MeanMaxPooling,
        "WLP": WeightedLayerPooling,
        "ConcatPool": MeanPooling,
        "AP": AttentionPooling
    }

    @classmethod
    def get(cls, name):
        return cls._poolings.get(name)



class Net(nn.Module):
    def __init__(self, cfg, model_path=None, num_classes=2, pretrained=True):
        super(Net, self).__init__()
        self.cfg = cfg

        with open(f'{model_path}/config.json', 'r') as file:
            self.config = json.load(file)
        self.config = SimpleNamespace(**self.config)
        self.model = AutoModel.from_pretrained(model_path)
        self.pooling = NLPPoolings.get(self.cfg.pool)
        
        if self.cfg.pool == "WLP":
            self.pooling = self.pooling(self.config.num_hidden_layers, layer_start=cfg.layer_start)
        elif self.cfg.pool == "AP":
            hiddendim_fc = 128
            self.pooling = AttentionPooling(self.config.num_hidden_layers, self.config.hidden_size, hiddendim_fc)
        else:
            self.pooling = self.pooling(dim=1, cfg=cfg)

        if self.cfg.pool == "MeanMax":
            self.head = nn.Linear(self.config.hidden_size*2, num_classes)
        elif self.cfg.pool == "ConcatPool":
            if 'distilbert-base' in self.cfg.model_name:
                self.head = nn.Linear(self.config.hidden_size*4, num_classes)
            else:
                self.head = nn.Linear(self.config.hidden_size*4, num_classes)
        elif self.cfg.pool == "AP":
            self.head = nn.Linear(hiddendim_fc, num_classes) # regression head
        else:
            self.head = nn.Linear(self.config.hidden_size, num_classes)
        
        self.initializer_range = self.config.initializer_range
        print(self.model)
        print(self.head)
        self._init_weights(self.head)

        if hasattr(self.cfg, "reinit_n_layers") and cfg.reinit_n_layers > 0:
            self.reinit_n_layers = cfg.reinit_n_layers
            self._do_reinit()


        if 'deberta-base' in cfg.model_name or 'roberta-base' in cfg.model_name or 'scibert' in cfg.model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:9].requires_grad_(False) # exp15 45009
            
        elif 'deberta-large' in cfg.model_name or 'deberta-xlarge' in cfg.model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:18].requires_grad_(False) # exp15 45009
        
        if self.cfg.loss_function == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.cfg.loss_function == "CrossEntropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=cfg.class_weight)
        elif self.cfg.loss_function == "FocalLoss":
            self.loss_fn = FocalLoss(weight=cfg.class_weight)



    def _do_reinit(self):
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            # print(self.model)
            if 'facebook/bart' in self.cfg.model_name or 'distilbart' in self.cfg.model_name:
                self.model.encoder.layers[-(n+1)].apply(self._init_weights)
            elif 'funnel' in self.cfg.model_name:
                self.model.decoder.layers[-(n+1)].apply(self._init_weights)
            elif "distilbert" in self.cfg.model_name:
                self.model.transformer.layer[-(n+1)].apply(self._init_weights)
            else:
                self.model.encoder.layer[-(n+1)].apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        

    def feature(self, inputs):
        attention_mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]
        if self.cfg.pool == "WLP":
            if 'facebook/bart' in self.cfg.model_name or 'distilbart' in self.cfg.model_name:
                x = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                tmp = {'all_layer_embeddings': x.encoder_hidden_states}
            else:
                x = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                tmp = {'all_layer_embeddings': x.hidden_states}
                
            x = self.pooling(tmp)['token_embeddings'][:, 0]

        elif self.cfg.pool == "ConcatPool":
            if 'facebook/bart' in self.cfg.model_name or 'distilbart' in self.cfg.model_name:
                x = torch.stack(self.model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           output_hidden_states=True).decoder_hidden_states)
            else:
                x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states)

            p1 = self.pooling(x[-1], attention_mask, input_ids, self.cfg)
            p2 = self.pooling(x[-2], attention_mask, input_ids, self.cfg)
            p3 = self.pooling(x[-3], attention_mask, input_ids, self.cfg)
            p4 = self.pooling(x[-4], attention_mask, input_ids, self.cfg)

            x = torch.cat(
                (p1, p2, p3, p4), -1
            )
            

        elif self.cfg.pool == "AP":
            x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states)
            x = self.pooling(x)
        else:
            if 'facebook/bart' in self.cfg.model_name or 'distilbart' in self.cfg.model_name:
                x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            else:
                x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            x = self.pooling(x, attention_mask, input_ids, self.cfg)

        return x, inputs

    def forward(self, inputs, calculate_loss=True):
        x, input = self.feature(inputs)

        if (
            hasattr(self.cfg, "wide_dropout")
            and self.cfg.wide_dropout > 0.0
            and self.training
        ):
            x1 = self.head(F.dropout(x, p=self.cfg.wide_dropout, training=self.training))
            x2 = self.head(F.dropout(x, p=self.cfg.wide_dropout, training=self.training))
            x3 = self.head(F.dropout(x, p=self.cfg.wide_dropout, training=self.training))
            x4 = self.head(F.dropout(x, p=self.cfg.wide_dropout, training=self.training))
            x5 = self.head(F.dropout(x, p=self.cfg.wide_dropout, training=self.training))
            logits = (x1 + x2 + x3 + x4 + x5) / 5
        else:
            logits = self.head(x)

        outputs = {}
        outputs["logits"] = logits

        if "target" in input:
            outputs["target"] = input["target"]

        if calculate_loss:
            targets = input["target"]

            outputs["loss"] = self.loss_fn(logits, targets)
            # outputs["loss"] = 0.7* self.loss_fn(logits, targets) + 0.3*self.loss_fn(logits, targets)

        return outputs

