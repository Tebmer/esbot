#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging
from torch import Tensor
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def eval_model_loss(model, eval_dataloader, epoch_id, infer, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_sample = []
    tot_strat_loss = []
    pointwise_loss = []
    pointwise_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            # batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            for k, v in batch.items():
                if isinstance(v, Tensor): 
                    batch[k] = v.to(args.device)
                # For the key "strat_seq_input", its `v` is a `dict.
                elif isinstance(v, dict):
                    for k_, v_ in v.items():
                        batch[k][k_] = v_.to(args.device)
                else:
                    batch[k] = v
                    
            loss_sample, n_sample, strat_loss = model(
                validation=True,
                **batch
            )
            if torch.isnan(loss_sample).sum().cpu().long().numpy() > 0:
                print(loss_sample)
                exit()
            tot_loss.append(loss_sample.sum().cpu().float().numpy())
            tot_sample.append(n_sample.sum().cpu().float().numpy())
            if strat_loss is not None: 
                tot_strat_loss.append(strat_loss.cpu().float().numpy())
            if infer:
                pointwise_loss.extend(loss_sample.sum(dim=-1).cpu().tolist())
                pointwise_sample.extend(n_sample.cpu().tolist())
    #exit()
    tot_loss = np.sum(tot_loss)
    tot_sample = np.sum(tot_sample)
    mean_loss = tot_loss / tot_sample
    mean_ppl = np.exp(mean_loss)
    mean_strat_loss = np.mean(tot_strat_loss)
    
    print(f"\n Epoch {epoch_id}: Val loss {mean_loss} Val ppl {mean_ppl} Val strategy loss {mean_strat_loss}")
    return mean_loss, mean_ppl, mean_strat_loss, tot_sample, pointwise_loss, pointwise_sample
