'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from datetime import datetime

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2
    
class LUT():
    def __init__(self):
        self.lookupTable = [] # main lookup table
        self.lookupTableMetaData = [] # idx 0 calls since last response, idx 1 number of calls
        self.CS_threshold = 0.5
        self.cost_scale = 10

    def train(self, xs, ys):
        for x, y in zip(xs, ys):
            print("adding rows...")
            row = torch.stack([x, y])
            self.lookupTable.append(row)
            self.lookupTableMetaData.append([1000, 0])

    
    def forward(self, x):
        x = x[-1, -1, :]
        x = x.squeeze()
        closest_row_output = None
        if len(self.lookupTable) == 0:
            return None, None
        lookup_vecs = torch.stack([row[0] for row in self.lookupTable])
        outputs = torch.stack([row[1] for row in self.lookupTable])

        for i in range(len(self.lookupTableMetaData)):
            meta = self.lookupTableMetaData[i]
            meta[0] = meta[0] + 1
            self.lookupTableMetaData[i] = meta

        ### adding a punishment/ cost for looking up memories which have come up just before
        costs = []
        for row in self.lookupTableMetaData:
            number_of_look_up_since_last_hit = row[0]

            cost = (1/(number_of_look_up_since_last_hit+1))
            costs.append(cost)

        costs = torch.tensor(costs, device=device)
        sims = F.cosine_similarity(lookup_vecs, x.unsqueeze(0), dim=1)# - (1/10)*torch.norm(lookup_vecs- x.unsqueeze(0), dim=1)  # dim one since N, d
        sims = sims - self.cost_scale*costs
        
        max_sim_idx = torch.argmax(sims)
        highest_sim = sims[max_sim_idx].item()
        print("highest sim: ", highest_sim)


        closest_row_output = outputs[max_sim_idx]
        if highest_sim < self.CS_threshold:  # arbitrary threshold- This must be fixed as it is a very temp workaround. The best fix would be to somehow have an active threshold based 
            print("Low similarity, cs threshold: ", self.CS_threshold)
            return torch.zeros_like(closest_row_output), 1
        row_meta_data = self.lookupTableMetaData[max_sim_idx]
        self.lookupTableMetaData[max_sim_idx]= [0, row_meta_data[1]+1]
        # Note to self
        # output of forward could take into account more rows by adjusting the outputs in a sort of
        #  weighted average effected by the relative cosine distance 
        highest_sim= 1  ## Temp line for testing to be deleted asap
        return closest_row_output, highest_sim
    
    def resetLUT(self):
        self.lookupTable = [] # main lookup table
        self.lookupTableMetaData = []

    def reset_costs(self):
        for row in self.lookupTableMetaData:
            row[0] = 1000
            

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, wnn_block=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.LUT = LUT()
        self.wnn_block = wnn_block # we only activate the wnn block in the last n layers as specified by num_wnn_blocks
        if self.wnn_block:
            self.use_wnn = True
        else:
            self.use_wnn = False
        self.pre_wnn_x = None  
        self.residual_scale = 15
        

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        self.pre_wnn_x = x + self.mlp(self.ln_2(x))
        if self.wnn_block:
            if self.use_wnn and len(self.LUT.lookupTable)>0: # not using wnn for training only for inference 
                self.pre_wnn_x.requires_grad_()
                
                #x = self.pre_wnn_x + residual_scale*self.LUT.forward(self.pre_wnn_x) # It may not actually be useful to scale the residual since it is trained based on its residual effect being one- may require more thought though
                wnn_residual, highest_sim = self.LUT.forward(self.pre_wnn_x)
                wnn_residual = wnn_residual.unsqueeze(0) # add back the batch dim(1)
                res_tensor = torch.zeros_like(self.pre_wnn_x)
                res_tensor[:, -1, :] = wnn_residual
                x = self.pre_wnn_x +self.residual_scale*(res_tensor) # The idea here is that if highest sim is low then the model doest look as much to the lu
                return x, present
        return self.pre_wnn_x, present
    

def _ensure_pad_token_and_embeddings(tokenizer, model_self, lm_head, device):
    # ensure pad token exists in tokenizer
    if "<|pad|>" not in tokenizer.encoder:
        pad_id = max(tokenizer.encoder.values()) + 1
        tokenizer.encoder["<|pad|>"] = pad_id
        tokenizer.decoder[pad_id] = "<|pad|>"
    else:
        pad_id = tokenizer.encoder["<|pad|>"]

    vocab_size = len(tokenizer.encoder)
    old_emb = model_self.wte.weight.data
    old_vocab, emb_dim = old_emb.shape

    if vocab_size > old_vocab:
        new_emb = torch.nn.Embedding(vocab_size, emb_dim)
        new_emb.weight.data[:old_vocab] = old_emb
        new_emb.weight.data[old_vocab:] = old_emb.mean(dim=0)
        model_self.wte = new_emb.to(device)
        lm_head.weight = model_self.wte.weight  # tie weights

    return pad_id

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        self.n_ctx= config.n_ctx
        self.residual_scale = 15

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents
    
    def trainLUT(self, tokenizer, lm_head, label, label_context=None, sparsity_level=None):

        # Turning of lut during training- may change this in future
        for block in self.h:
            block.use_wnn = False

        # Encode label and optional label_context 
        encoded_label = tokenizer.encode(label)
        if label_context is not None:
            label_context_encoded = tokenizer.encode(label_context)
        else:
            label_context_encoded = []

        if len(encoded_label) == 0:
            return  # nothing to train on

        for i, block in enumerate(self.h):
            if not block.wnn_block:
                continue

            print(f"Collecting data for WNN training on block {i}")
            now_block = datetime.now()

            for k in range(len(encoded_label)):
                if sparsity_level is not None and sparsity_level < 1.0:
                    if torch.rand(()) > sparsity_level:
                        continue

                context = encoded_label[max(0, k - self.n_ctx):k]
                if label_context_encoded:
                    context = label_context_encoded + context

                # Crop on the left if longer than n_ctx
                context = context[-self.n_ctx:]

                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
                target_tensor = torch.tensor([encoded_label[k]], dtype=torch.long, device=device)     # [1]

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)         # [1, T]

                x = self.wte(context_tensor) + self.wpe(position_ids)

                print("Doing inference (pre-WNN blocks)")
                for block_idx in range(i):
                    print("  Block:", block_idx)
                    with torch.no_grad():
                        x, _ = self.h[block_idx](x)

                print("Doing inference on current block:", i)
                # run current block once to  block.pre_wnn_x
                _, _ = block(x)
                pre_wnn_x = block.pre_wnn_x  # [1, T, d]

                pre_wnn_x.retain_grad()
                x = pre_wnn_x

      
                for block_idx in range(i + 1, len(self.h)):
                    print("  Doing inference on block:", block_idx)
                    x, _ = self.h[block_idx](x)

                logits = lm_head(x)  # [1, T, vocab]
                print("Completed inference. Time:", datetime.now() - now_block)

                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                self.zero_grad()
                print("Start backward")
                now_back = datetime.now()
                loss.backward()
                print("Finish backward. Time:", datetime.now() - now_back)

                # Target residual = negative gradient at pre_wnn_x
                wnn_target_residual = self.residual_scale * (-pre_wnn_x.grad)  # [1, T, d]

                # Take only last time step for LUT
                pre_wnn_x_last = pre_wnn_x.detach()[:, -1, :]           # [1, d]
                target_residual_last = wnn_target_residual.detach()[:, -1, :]  # [1, d]

                # (Optional) normalize to avoid giant residuals on rare tokens
                # target_residual_last = target_residual_last / (
                #     torch.linalg.vector_norm(target_residual_last, ord=2, dim=-1, keepdim=True) + 1e-6
                # )

                print("Training LUT")
                now_lut = datetime.now()
                block.LUT.train(pre_wnn_x_last, target_residual_last)
                print("Time for training LUT:", datetime.now() - now_lut)

            print("Finished block", i, "total time:", datetime.now() - now_block)

        # Re-enable WNN usage for inference
        for block in self.h:
            block.use_wnn = True




class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents