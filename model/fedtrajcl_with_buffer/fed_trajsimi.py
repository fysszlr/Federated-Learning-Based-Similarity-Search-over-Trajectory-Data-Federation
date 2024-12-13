
import logging
import random
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from config import Config as Config
from utils import tool_funcs
from utils.data_loader import read_trajsimi_simi_dataset, read_trajsimi_traj_dataset
from utils.traj import merc2cell2, generate_spatial_features


class BaseClient():
    def __init__(self, id, encoder, dataset):
        self.dataset = dataset
        self.encoder = encoder
        self.id = id

        _seq_embedding_dim = Config.seq_embedding_dim
        self.model = TrajSimiRegression(_seq_embedding_dim)
        self.model.to(Config.device)
        self.criterion = nn.MSELoss()
        self.criterion.to(Config.device)

        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.cellembs = pickle.load(open(Config.dataset_embs_file, 'rb')).to(Config.device)  # tensor
        self.batch_size = Config.trajsimi_batch_size
        self.E = Config.E
        self.device = Config.device

        self.optimizer = torch.optim.Adam([ \
            {'params': self.model.parameters(), \
             'lr': Config.trajsimi_learning_rate, \
             'weight_decay': Config.trajsimi_learning_weight_decay}, \
            {'params': self.encoder.clmodel.encoder_q.parameters(), \
             'lr': Config.trajsimi_learning_rate}])

    def local_train(self):
        raise NotImplementedError()

    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return

    def test_accuracy(self):
        if self.testset == None: return -1
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def get_features_and_labels(self, train=True, batch=-1):
        dataloader = None
        if train:
            dataloader = self.trainloader
        else:
            dataloader = self.testloader
        features_batch = []
        labels_batch = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i == batch: break
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                _, f_s = self.model(inputs, features=True)
                features_batch.append(f_s)
                labels_batch.append(labels)
        features = torch.cat(features_batch)
        labels = torch.cat(labels_batch)
        return features, labels

    def save_features_and_labels(self, fn, train=True, batch=-1):
        features, labels = self.get_features_and_labels(train, batch)
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        np.save('%s_features.npy' % fn, features)
        np.save('%s_labels.npy' % fn, labels)
        return


class BaseServer(BaseClient):
    def __init__(self, id, encoder, dataset):
        super().__init__(id,  encoder, dataset)
        self.step_count = 0
        self.n_clients = Config.cls_num
        self.n_clients_per_round = round(Config.C * self.n_clients)
        self.learning_rate = Config.trajsimi_learning_rate

    def aggregate_model(self):
        raise NotImplementedError()

    def train(self):
        # finish 1 comm round
        raise NotImplementedError()

    def sample_client(self):
        return random.sample(
            self.clients,
            self.n_clients_per_round,
        )


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def parameters_to_tensor(self):
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params, 0)
        return params

    def tensor_to_parameters(self, tensor):
        p = 0
        for param in self.parameters():
            shape = param.shape
            delta = 1
            for item in shape: delta *= item
            param.data = tensor[p: p + delta].view(shape).detach().clone()
            p += delta

class TrajSimiRegression(BaseModule):
    def __init__(self, nin):
        super(TrajSimiRegression, self).__init__()
        self.enc = nn.Sequential(nn.Linear(nin, nin),
                                nn.ReLU(),
                                nn.Linear(nin, nin))

    def forward(self, trajs):
        # trajs: [batch_size, emb_size]
        return F.normalize(self.enc(trajs), dim=1) #[batch_size, emb_size]

class TrajSimi():
    def __init__(self, encoder):
        super(TrajSimi, self).__init__()
        self.dic_datasets = TrajSimi.load_trajsimi_dataset()
        self.encoder = encoder
        dic_datasets_split = tool_funcs.split_simi(self.dic_datasets, Config.cls_num)

        self.checkpoint_filepath = '{}/{}_trajsimi_{}_{}_best{}.pt' \
            .format(Config.checkpoint_dir, Config.dataset_prefix, \
                    Config.trajsimi_encoder_name, \
                    Config.trajsimi_measure_fn_name, \
                    Config.dumpfile_uniqueid)

        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.cellembs = pickle.load(open(Config.dataset_embs_file, 'rb')).to(Config.device)  # tensor

        import model.fedtrajcl.trajSimi_fedAvg as fed
        self.clients = []
        for i in range(Config.cls_num):
            id = i + 1
            client = fed.Client(id, encoder, dic_datasets_split[i])
            self.clients.append(client)
        # init server
        self.server = fed.Server(0, encoder, self.dic_datasets)
        self.server.clients = self.clients
        self.criterion = nn.MSELoss()

    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_trajsimi start.@={:.3f}".format(training_starttime))

        _seq_embedding_dim = Config.seq_embedding_dim

        self.trajsimiregression = TrajSimiRegression(_seq_embedding_dim)
        self.trajsimiregression.to(Config.device)
        self.criterion.to(Config.device)

        best_epoch = 0
        best_hr_eval = 0
        bad_counter = 0
        bad_patience = Config.trajsimi_training_bad_patience

        pat = 0

        for i_ep in range(Config.trajsimi_epoch):
            _time_ep = time.time()
            train_losses = []
            train_gpus = []
            train_rams = []

            self.server.train()

            eval_metrics = self.test(dataset_type='eval')
            logging.info(
                "eval.     i_ep={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}, gpu={}, ram={}".format(i_ep, *eval_metrics))

            hr_eval_ep = eval_metrics[1]
            training_gpu_usage = tool_funcs.mean(train_gpus)
            training_ram_usage = tool_funcs.mean(train_rams)

            # early stopping
            if hr_eval_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = hr_eval_ep
                bad_counter = 0

                torch.save({"encoder_q": self.server.encoder.clmodel.encoder_q.state_dict(),
                            "trajsimi": self.server.model.state_dict()},
                           self.checkpoint_filepath)
            else:
                bad_counter += 1

            pat += 1
            if pat == 4 :
                training_endtime = time.time()
                logging.info("training end. @={:.3f}, best_epoch={}, best_hr_eval={:.4f}" \
                             .format(training_endtime - training_starttime, best_epoch, best_hr_eval))
                break

            if bad_counter == bad_patience or i_ep + 1 == Config.trajsimi_epoch:
                training_endtime = time.time()
                logging.info("training end. @={:.3f}, best_epoch={}, best_hr_eval={:.4f}" \
                             .format(training_endtime - training_starttime, best_epoch, best_hr_eval))
                break

        # test
        checkpoint = torch.load(self.checkpoint_filepath)
        self.trajsimiregression.load_state_dict(checkpoint['trajsimi'])
        self.trajsimiregression.to(Config.device)
        self.trajsimiregression.eval()
        self.encoder.clmodel.encoder_q.load_state_dict(checkpoint['encoder_q'])

        test_starttime = time.time()
        test_metrics = self.test(dataset_type='test')
        test_endtime = time.time()
        logging.info("test. @={:.3f}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}, gpu={}, ram={}".format( \
            test_endtime - test_starttime, *test_metrics))

        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': test_metrics[4], \
                'task_test_ram': test_metrics[5], \
                'hr5': test_metrics[1], 'hr20': test_metrics[2], 'hr20in5': test_metrics[3]}

    @torch.no_grad()
    def test(self, dataset_type):
        # prepare dataset
        if dataset_type == 'eval':
            datasets_simi, max_distance = self.dic_datasets['evals_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['evals_traj']

        elif dataset_type == 'test':
            datasets_simi, max_distance = self.dic_datasets['tests_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['tests_traj']

        self.server.model.eval()
        self.server.encoder.eval()

        datasets_simi = torch.tensor(datasets_simi, device=Config.device, dtype=torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        traj_outs = []

        # get traj embeddings
        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_single_batchi(datasets)):
            trajs_emb, trajs_emb_p, trajs_len = batch
            embs = self.server.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
            outs = self.server.model(embs)
            traj_outs.append(outs)

        # calculate similarity
        traj_outs = torch.cat(traj_outs)
        pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal=1) == 1]

        # metrics
        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)
        hrA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hrB = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
        hrBinA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)

        gpu = tool_funcs.GPUInfo.mem()[0]
        ram = tool_funcs.RAMInfo.mem()

        return loss.item(), hrA, hrB, hrBinA, gpu, ram

    @torch.no_grad()
    def all_test(self, dataset_type):
        # prepare dataset
        if dataset_type == 'eval':
            datasets_simi, max_distance = self.dic_datasets['evals_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['evals_traj']

        elif dataset_type == 'test':
            datasets_simi, max_distance = self.dic_datasets['tests_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['tests_traj']

        self.server.model.eval()
        self.server.encoder.eval()

        datasets_simi = torch.tensor(datasets_simi, device=Config.device, dtype=torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        traj_outs = []

        # get traj embeddings
        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_single_batchi(datasets)):
            trajs_emb, trajs_emb_p, trajs_len = batch
            embs = self.server.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
            outs = self.server.model(embs)
            traj_outs.append(outs)

        # calculate similarity
        traj_outs = torch.cat(traj_outs)
        pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal=1) == 1]

        # metrics
        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)

        hr5 = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hr10 = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
        hr20 = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
        hr50 = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
        r10_50 = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10)

        gpu = tool_funcs.GPUInfo.mem()[0]
        ram = tool_funcs.RAMInfo.mem()

        logging.info("hr5={}, hr10={}, hr20={}, hr50={}, r10_50={}".format(hr5, hr10, hr20, hr50, r10_50))

    @torch.no_grad()
    def trajsimi_dataset_generator_single_batchi(self, datasets):
        cur_index = 0
        len_datasets = len(datasets)

        while cur_index < len_datasets:
            end_index = cur_index + Config.trajsimi_batch_size \
                if cur_index + Config.trajsimi_batch_size < len_datasets \
                else len_datasets

            trajs = [datasets[d_idx] for d_idx in range(cur_index, end_index)]

            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first=False).to(Config.device)

            trajs_emb_cell = [self.cellembs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first=False).to(
                Config.device)  # [seq_len, batch_size, emb_dim]

            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype=torch.long, device=Config.device)

            yield trajs_emb_cell, trajs_emb_p, trajs_len
            cur_index = end_index

    @staticmethod
    def hitting_ratio(preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int):
        # hitting ratio and recall metrics. see NeuTraj paper
        # the overlap percentage of the topk predicted results and the topk ground truth
        # overlap(overlap(preds@pred_topk, truths@truth_topk), truths@truth_topk) / truth_topk

        # preds = [batch_size, class_num], tensor, element indicates the probability
        # truths = [batch_size, class_num], tensor, element indicates the probability
        assert preds.shape == truths.shape and pred_topk < preds.shape[1] and truth_topk < preds.shape[1]

        _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim=1, largest=False)
        _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim=1, largest=False)

        preds_k_idx = preds_k_idx.cpu()
        truths_k_idx = truths_k_idx.cpu()

        tp = sum([np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size for i in range(preds_k_idx.shape[0])])

        return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])

    @staticmethod
    def load_trajsimi_dataset():
        # read (1) traj dataset for trajsimi, (2) simi matrix dataset for trajsimi
        trajsimi_traj_dataset_file = Config.dataset_file
        trajsimi_simi_dataset_file = '{}_traj_simi_dict_{}.pkl'.format( \
            Config.dataset_file, Config.trajsimi_measure_fn_name)

        trains_traj, evals_traj, tests_traj = read_trajsimi_traj_dataset(trajsimi_traj_dataset_file)
        trains_traj, evals_traj, tests_traj = trains_traj.merc_seq.values, evals_traj.merc_seq.values, tests_traj.merc_seq.values
        trains_simi, evals_simi, tests_simi, max_distance = read_trajsimi_simi_dataset(trajsimi_simi_dataset_file)

        # trains_traj : [[[lon, lat_in_merc], [], ..], [], ...]
        # trains_simi : list of list
        return {'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj, \
                'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
                'max_distance': max_distance}

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_filepath)
        self.server.model.load_state_dict(checkpoint['trajsimi'])
        self.server.model.to(Config.device)
        self.server.model.eval()
        self.encoder.clmodel.encoder_q.load_state_dict(checkpoint['encoder_q'])


        return

