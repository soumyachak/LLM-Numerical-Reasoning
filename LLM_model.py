import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd

torch.cuda.is_available()

import gc
torch.cuda.empty_cache()
gc.collect()

'''LLM config'''
config = {
    "embed_dim" : 16,
    "num_heads" : 4,
    "block_size" : 2,
    "attn_pdrop" : 0.1,
    "resid_pdrop" : 0.1,
    "vocab_size" : 100,
    "embd_pdrop" : 0.1,
    "n_layer" : 6,
    "device" : 'cuda',
    "weight_decay" : 0.1,
    "learning_rate" : 3e-4,
    "betas" : (0.9, 0.95),
    "grad_norm_clip" : 1.0,
    "batch_size" : 4
}

'''Define rotary embedding'''
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    # Rotates half the hidden dims of the input.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Applies Rotary Position Embedding to the query and key tensors.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

'''Define Masked Multihead self attantion layer'''
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads in a batch
        self.q_attn = nn.Linear(config["embed_dim"], config["embed_dim"])
        self.k_attn = nn.Linear(config["embed_dim"], config["embed_dim"])
        self.v_attn = nn.Linear(config["embed_dim"], config["embed_dim"])

        # output projection
        self.c_proj = nn.Linear(config["embed_dim"], config["embed_dim"])
        # regularization
        self.resid_dropout = nn.Dropout(config["resid_pdrop"])

        # multi head attention
        self.multihead_attn = nn.MultiheadAttention(config["embed_dim"], config["num_heads"], batch_first=True, dropout=config["attn_pdrop"])

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("attn_mask", torch.zeros((config["block_size"], config["block_size"]), dtype=torch.bool) \
                                    .masked_fill(torch.tril(torch.ones(config["block_size"], config["block_size"])) \
                                    .view(config["block_size"], config["block_size"]) \
                                    [:config["block_size"],:config["block_size"]] == 0, True))
        
        # rotarry embedding
        self.rotary_emb = LlamaRotaryEmbedding(
                config["embed_dim"],
                max_position_embeddings=config["block_size"],
                base=10000,
            )

    def forward(self, x):
        # calculate query, key, values for batch
        q, k ,v  = self.q_attn(x), self.k_attn(x), self.v_attn(x)

        # apply rotary embeddings
        device = x.device
        b, t, l = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        cos, sin = self.rotary_emb(v, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k = q[0], k[0]

        # calculate self attention
        attn_output = self.multihead_attn(query=q, key=k, value=v, attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]])[0]

        # output projection
        y = self.resid_dropout(self.c_proj(attn_output))
        return y
    
'''Define Decoder block'''
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["embed_dim"])
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["embed_dim"])
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config["embed_dim"], 4 * config["embed_dim"]),
            c_proj  = nn.Linear(4 * config["embed_dim"], config["embed_dim"]),
            act     = nn.GELU(),
            dropout = nn.Dropout(config["resid_pdrop"]),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    
'''Define LLM model'''
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config["block_size"]
        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config["vocab_size"], config["embed_dim"]), # the code is commented since we are using distribution embedding

            wte = nn.Embedding(10000, 1), # distribution embedding

            drop = nn.Dropout(config["embd_pdrop"]),
            h = nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
            ln_f = nn.LayerNorm(config["embed_dim"]),
        ))

        self.lm_head = nn.Linear(config["embed_dim"], 1, bias=False)
        
        # initialize all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["n_layer"]))

        n_params = sum(p.numel() for p in self.transformer.parameters())

        param = self.lm_head.weight.shape
        print("number of parameters: %.2fM" % ((n_params+param[0]*param[1])/1e6))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            """ initialize distribution embedding from normal distribution """
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        idx = idx.to(torch.long)
        """ the below code is commented since we are using distribution embedding """
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        """ generate embeddings from distribution"""
        dist_index = np.zeros((idx.shape[0],config["block_size"],config["embed_dim"]))
        for i in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                np.random.seed(int(idx[i,j].item()))
                k = np.random.randint(0, 10000, size=config["embed_dim"])
                dist_index[i,j,:] = k
        inp = torch.tensor(dist_index).to(device)
        inp = inp.to(torch.long)
        tok_emb = self.transformer.wte(inp)
        tok_emb = tok_emb.reshape(idx.shape[0],config["block_size"],config["embed_dim"]).contiguous()

        """ This was used to provide a value to the token embedding vector based on the input number. This didn't help much in reducing loss
         function. Don't clip the gradient if this is used. """
        # idx_expanded = idx.view(-1, config["block_size], 1).expand_as(tok_emb)
        # tok_emb = idx_expanded * tok_emb

        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        """ apply linear layer to the final embedding """
        x = x[:,-1,:]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.mse_loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the prediction
            logits, _ = self(idx_cond)

        return logits

    def configure_optimizers(self, config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == config["n_layer"], "parameters %s were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config["learning_rate"], betas=config["betas"])
        return optimizer


if __name__ == "__main__":
    '''Generate data'''
    np.random.seed(32)
    X1 = np.random.randint(1, 100, size=1000)
    X2 = np.random.randint(1, 100, size=1000)
    Y = np.log(X1) + np.sin(X2)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

    data = {}
    lst = df.values.tolist()
    for i in range(len(lst)):
        data[i] = {"input": lst[i][:-1],
                "output": [lst[i][-1]]}

    lst=[]
    for x in sorted(data.keys()):
        lst.append([data[x]['input'], data[x]['output']])
    lst_train = lst[:800]
    lst_test = lst[800:]
    print(lst_train[0])

    '''Check number of common rows bw train and eval dataset'''
    k = 0
    for i in range(len(lst_test)):
        for j in range(len(lst_train)):
            if lst_test[i][0] == lst_train[j][0]:
                k+=1
                break
    print("Common rows be train and eval data", k)

    '''Generate training and eval dataset'''
    class RegressDataset(Dataset):
        def __init__(self, train):
            if train=='train':
                self.data = lst_train
            else:
                self.data = lst_test

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x,y = torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])
            return x, y
        
    train_dataset = RegressDataset('train')
    test_dataset = RegressDataset('test')
    train_dataloader = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size=config["batch_size"]
                            )
    test_dataloader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=config["batch_size"]
                            )

    '''Define training function'''
    train_loss_per_epoch = []
    eval_loss_per_epoch= []
    def train_epocs(model, optimizer, train_dataloader, epochs):
        for i in range(epochs):
            start_time = time.time()
            model.train()
            idx=0
            sum_loss = 0
            for x, y in train_dataloader:
                x = x.cuda()
                y = y.cuda()
                logits, loss = model(x, y)
                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_norm_clip"])
                optimizer.step()
                idx+=len(x)
                sum_loss+= loss
            train_loss_per_epoch.append(sum_loss/idx)

            model.eval()
            with torch.no_grad():
                val_loss = 0
                idx_test=0
                sum_test_loss = 0
                for x, y in test_dataloader:
                    x = x.cuda()
                    y = y.cuda()
                    logits, loss = model(x, y)
                    idx_test += len(x)
                    sum_test_loss += loss
                eval_loss_per_epoch.append(sum_test_loss/idx_test)

            print("Total time taken for Epoch {} is {} secs and train loss = {} and test loss = {}".format(i, int(time.time()-start_time), 
                                                                                                        sum_loss/idx, sum_test_loss/idx_test))
        return

    '''Initialize model and optimizer'''
    model=GPT(config)
    model = model.cuda()
    optimizer = model.configure_optimizers(config)

    '''Model training'''
    train_epocs(model, optimizer, train_dataloader, epochs=40)

    '''Save model'''
    PATH = "./regress_GPT.pt"
    torch.save(model.state_dict(), PATH)

    '''Load model'''
    model=GPT(config)
    model = model.cuda()
    PATH = "./regress_GPT.pt"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    '''Model prediction'''
    print(lst_test)
    inp = torch.tensor([[81, 59]], dtype=torch.long).to('cuda')
    with torch.no_grad():
        cat = model.generate(inp, 1)
    print(inp)
    print(cat[:,:])