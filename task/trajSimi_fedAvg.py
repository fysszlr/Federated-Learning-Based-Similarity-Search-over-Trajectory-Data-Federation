import torch
import torch.nn as nn
import torch.utils.data
import math
import time
from tqdm import tqdm
from task.fed_trajsimi import *


class Client(BaseClient):
    def __init__(self, id, encoder, dataset):
        super().__init__(id, encoder, dataset)

    def local_train(self):
        self.model.train()
        self.encoder.train()
        _time_ep = time.time()
        train_losses = []
        train_gpus = []
        train_rams = []
        self.omega = self.model.parameters_to_tensor().clone().detach()
        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_pairs_batchi()):
            _time_batch = time.time()
            self.optimizer.zero_grad()

            trajs_emb, trajs_emb_p, trajs_len, sub_simi = batch
            embs = self.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
            outs = self.model(embs)

            pred_l1_simi = torch.cdist(outs, outs, 1)
            pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1]
            truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1]

            omega_loss = torch.norm(
                self.model.parameters_to_tensor() - self.omega
            ) ** 2
            loss_train = self.criterion(pred_l1_simi, truth_l1_simi)

            loss_train.backward()
            self.optimizer.step()
            train_losses.append(loss_train.item())
            train_gpus.append(tool_funcs.GPUInfo.mem()[0])
            train_rams.append(tool_funcs.RAMInfo.mem())

            if i_batch % 200 == 0 and i_batch:
                logging.debug("training. id={}, ep-batch={}, train_loss={:.4f}, @={:.3f}, gpu={}, ram={}" \
                              .format(self.id, i_batch, loss_train.item(),
                                      time.time() - _time_batch, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

        # i_ep
        logging.info("training. i_id={}, loss={:.4f}, @={:.3f}" \
                     .format(self.id, tool_funcs.mean(train_losses), time.time() - _time_ep))

    # pair-wise batchy data generator - for training
    def trajsimi_dataset_generator_pairs_batchi(self):
        datasets_simi, max_distance = self.dataset['trains_simi'], \
            self.dataset['max_distance']
        datasets = self.dataset['trains_traj']
        len_datasets = len(datasets)
        datasets_simi = torch.tensor(datasets_simi, device=Config.device, dtype=torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance

        count_i = 0
        batch_size = len_datasets if len_datasets < Config.trajsimi_batch_size else Config.trajsimi_batch_size
        counts = math.ceil((len_datasets / batch_size) ** 2)

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k=batch_size)
            # dataset_idxs_sample.sort(key = lambda idx: len(datasets[idx][1]), reverse = True) # len descending order
            sub_simi = datasets_simi[dataset_idxs_sample][:, dataset_idxs_sample]

            trajs = [datasets[d_idx] for d_idx in dataset_idxs_sample]

            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first=False).to(Config.device)

            trajs_emb_cell = [self.cellembs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first=False).to(
                Config.device)  # [seq_len, batch_size, emb_dim]

            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype=torch.long, device=Config.device)

            yield trajs_emb_cell, trajs_emb_p, trajs_len, sub_simi
            count_i += 1


class Server(BaseServer):
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)
        return

    def train(self):
        # random clients
        clients = self.sample_client()

        for client in clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate

        for client in clients:
            # local train
            client.local_train()

        # aggregate params
        self.aggregate_model(clients)
        self.step_count +=1
        if (self.step_count == 10):
            self.learning_rate *= 0.5

        return clients
