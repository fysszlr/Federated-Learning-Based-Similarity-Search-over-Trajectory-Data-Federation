import heapq

import time
import logging
import pickle

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from model.fedtrajcl_with_buffer.moco import MoCo
from layers.dual_attention import DualSTB
from utils.data_loader import read_traj_dataset
from utils.traj import *
from utils import tool_funcs

sys.path.append('../..')


class BaseClient():
    def __init__(self, id, str_aug1, str_aug2, dataset):
        self.batch_size = Config.trajcl_batch_size
        self.trainset = dataset
        self.testset = None
        self.id = id

        self.E = Config.E
        self.aug1 = get_aug_fn(str_aug1)
        self.aug2 = get_aug_fn(str_aug2)

        self.embs = pickle.load(open(Config.dataset_embs_file, 'rb')).to('cpu').detach()  # tensor
        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.temperature = 0.07

        # if self.trainset != None:
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=Config.trajcl_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            collate_fn=partial(collate_and_augment, cellspace=self.cellspace,
                               embs=self.embs, augfn1=self.aug1, augfn2=self.aug2)
        )
        # if self.testset != None:
        #     self.testloader = DataLoader(
        #         self.testset,
        #         batch_size=Config.trajcl_batch_size,
        #         shuffle=False,
        #         num_workers=0,
        #         drop_last=True,
        #         collate_fn=partial(collate_and_augment, cellspace=self.cellspace,
        #                            embs=self.embs, augfn1=self.aug1, augfn2=self.aug2)
        #     )

        self.E = 1
        self.device = Config.device
        self.model = TrajCL().to(Config.device)
        self.last_model = TrajCL().to(Config.device)
        self.global_model = TrajCL().to(Config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.trajcl_training_lr, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=Config.trajcl_training_lr_degrade_step,
                                                         gamma=Config.trajcl_training_lr_degrade_gamma)

    def local_train(self):
        raise NotImplementedError()

    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return


class BaseServer(BaseClient):
    def __init__(self, id, str_aug1, str_aug2, dataset):
        super().__init__(id, str_aug1, str_aug2, dataset)
        self.n_clients = Config.cls_num
        self.n_clients_per_round = round(self.n_clients * Config.C)
        self.learning_rate = Config.trajcl_training_lr
        self.step = 0

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

    def load_parameters(self, new_params):
        with torch.no_grad():
            for param, new_param in zip(self.parameters(), new_params):
                new_param = np.array(new_param) if not isinstance(new_param, np.ndarray) else new_param
                param.copy_(torch.from_numpy(new_param))


class TrajCL(BaseModule):

    def __init__(self):
        super(TrajCL, self).__init__()

        encoder_q = DualSTB(Config.seq_embedding_dim,
                            Config.trans_hidden_dim,
                            Config.trans_attention_head,
                            Config.trans_attention_layer,
                            Config.trans_attention_dropout,
                            Config.trans_pos_encoder_dropout)
        encoder_k = DualSTB(Config.seq_embedding_dim,
                            Config.trans_hidden_dim,
                            Config.trans_attention_head,
                            Config.trans_attention_layer,
                            Config.trans_attention_dropout,
                            Config.trans_pos_encoder_dropout)

        self.clmodel = MoCo(encoder_q, encoder_k,
                            Config.seq_embedding_dim,
                            Config.moco_proj_dim,
                            Config.moco_nqueue,
                            temperature=Config.moco_temperature)

    def forward(self, trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len):
        # create kwargs inputs for TransformerEncoder

        max_trajs1_len = trajs1_len.max().item()  # in essense -- trajs1_len[0]
        max_trajs2_len = trajs2_len.max().item()  # in essense -- trajs2_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device=Config.device)[None, :] >= trajs1_len[:, None]
        src_padding_mask2 = torch.arange(max_trajs2_len, device=Config.device)[None, :] >= trajs2_len[:, None]

        logits, targets = self.clmodel(
            {'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len,
             'srcspatial': trajs1_emb_p},
            {'src': trajs2_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len,
             'srcspatial': trajs2_emb_p})
        return logits, targets

    def interpret(self, trajs1_emb, trajs1_emb_p, trajs1_len):
        max_trajs1_len = trajs1_len.max().item()  # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len, device=Config.device)[None, :] >= trajs1_len[:, None]
        traj_embs = self.clmodel.encoder_q(
            **{'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len,
               'srcspatial': trajs1_emb_p})
        return traj_embs

    def loss(self, logits, targets):
        return self.clmodel.loss(logits, targets)

    def load_checkpoint(self):
        checkpoint_file = '{}/{}_TrajCL_best{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
                                                          Config.dumpfile_uniqueid)
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self


def collate_and_augment(trajs, cellspace, embs, augfn1, augfn2):
    # trajs: list of [[lon, lat], [,], ...]

    # 1. augment the input traj in order to form 2 augmented traj views
    # 2. convert augmented trajs to the trajs based on mercator space by cells
    # 3. read cell embeddings and form batch tensors (sort, pad)

    trajs1 = [augfn1(t) for t in trajs]
    trajs2 = [augfn2(t) for t in trajs]

    trajs1_cell, trajs1_p = zip(*[merc2cell2(t, cellspace) for t in trajs1])
    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs2])

    trajs1_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs1_p]
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p]

    trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first=False).to(Config.device)
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first=False).to(Config.device)

    trajs1_emb_cell = [embs[list(t)] for t in trajs1_cell]
    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]

    trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first=False).to(
        Config.device)  # [seq_len, batch_size, emb_dim]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first=False).to(
        Config.device)  # [seq_len, batch_size, emb_dim]

    trajs1_len = torch.tensor(list(map(len, trajs1_cell)), dtype=torch.long, device=Config.device)
    trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype=torch.long, device=Config.device)

    # return: two padded tensors and their lengths
    return trajs1_emb_cell, trajs1_emb_p, trajs1_len, trajs2_emb_cell, trajs2_emb_p, trajs2_len


def collate_for_test(trajs, cellspace, embs):
    # trajs: list of [[lon, lat], [,], ...]

    # behavior is similar to collate_and_augment, but no augmentation

    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs])
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p]
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first=False).to(Config.device)

    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first=False).to(
        Config.device)  # [seq_len, batch_size, emb_dim]

    trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype=torch.long, device=Config.device)
    # return: padded tensor and their length
    return trajs2_emb_cell, trajs2_emb_p, trajs2_len


class TrajCLTrainer:

    def __init__(self, str_aug1, str_aug2):
        # import model.fedavg_baselines as Fed
        import model.fedtrajcl_with_buffer.fed_trajcl as Fed
        super(TrajCLTrainer, self).__init__()

        self.aug1 = get_aug_fn(str_aug1)
        self.aug2 = get_aug_fn(str_aug2)

        self.embs = pickle.load(open(Config.dataset_embs_file, 'rb')).to('cpu').detach()  # tensor
        # self.embs, _ = torch.sort(self.embs)
        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))

        train_dataset, _, test_dataset = read_traj_dataset(Config.dataset_file)
        if Config.ldp == 1:
            train_dataset_split = tool_funcs.split(get_ldp_dataset(train_dataset), Config.cls_num)
            print(train_dataset_split)
        else:
            train_dataset_split = tool_funcs.split(train_dataset, Config.cls_num)

        # init clients
        self.clients = []
        for i in range(Config.cls_num):
            id = i + 1
            client = Fed.Client(
                id,
                str_aug1,
                str_aug2,
                train_dataset_split[i]
            )
            self.clients.append(client)
        # init server
        self.server = Fed.Server(0, str_aug1, str_aug2, test_dataset)
        self.server.clients = self.clients

        self.checkpoint_file = '{}/{}_TrajCL_best_{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
                                                               Config.dumpfile_uniqueid)

    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.trajcl_training_bad_patience

        for i_ep in range(Config.trajcl_training_epochs):
            _time_ep = time.time()

            Config.fedproc_loss_weight = 1 - (i_ep + 1) / Config.trajcl_training_epochs / 10

            _, loss_ep_avg, delta = self.server.train()

            logging.info("[Training] ep={}: avg_loss={:.3f}, delta={:.3f}, @={:.3f}/{:.3f}, time={}" \
                         .format(i_ep, loss_ep_avg, delta, time.time() - _time_ep, time.time() - training_starttime,
                                 time.time() - training_starttime))

            # early stopping
            if abs(loss_ep_avg - best_loss_train) > 1e-2:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == Config.trajcl_training_epochs:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                             .format(time.time() - training_starttime, best_epoch, best_loss_train))
                break

        return {'enc_train_time': time.time() - training_starttime, \
                'enc_train_gpu': training_gpu_usage, \
                'enc_train_ram': training_ram_usage}

    @torch.no_grad()
    def test(self):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying downsampling rates, and distort rates

        logging.info('[Test]start.')
        self.load_checkpoint()
        self.server.model.eval()

        # varying downsampling; varying distort
        vt = Config.test_type
        results = []
        for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
                q_lst, db_lst = pickle.load(fh)
                querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
                dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
                targets = torch.diag(dists)  # [1000]
                result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                results.append(result)
        logging.info(
            '[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))
        return

    @torch.no_grad()
    def test_merc_seq_to_embs(self, q_lst, db_lst):
        querys = []
        databases = []
        num_query = len(q_lst)  # 1000
        num_database = len(db_lst)  # 100000
        batch_size = num_query

        for i in range(num_database // batch_size):
            if i == 0:
                trajs1_emb, trajs1_emb_p, trajs1_len \
                    = collate_for_test(q_lst, self.cellspace, self.embs)
                trajs1_emb = self.server.model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                querys.append(trajs1_emb)

            trajs2_emb, trajs2_emb_p, trajs2_len \
                = collate_for_test(db_lst[i * batch_size: (i + 1) * batch_size], self.cellspace, self.embs)
            trajs2_emb = self.server.model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)
            databases.append(trajs2_emb)

        querys = torch.cat(querys)  # tensor; traj embeddings
        databases = torch.cat(databases)
        return querys, databases

    def dump_embeddings(self):
        return

    def save_checkpoint(self):
        torch.save({'model_state_dict': self.server.model.state_dict(),
                    'aug1': self.aug1.__name__,
                    'aug2': self.aug2.__name__},
                   self.checkpoint_file)
        return

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.server.model.load_state_dict(checkpoint['model_state_dict'])
        self.server.model.to(Config.device)

        return

    def knn_test(self, fn_name="edwp"):
        # 1. Read beat model
        # 2. Get 1000 querys and 10000 database
        # 3. Collate for knn test
        # 4. Calculate tradition knn using traj_dist
        # 5. count hit ratio and recall ratio

        # 1
        logging.info('[Knn-Test]start.')
        start_time = time.time()
        self.load_checkpoint()
        self.server.model.eval()
        fn = tool_funcs.get_simi_fn(fn_name)

        # 2
        with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as fh:
            _, db_lst = pickle.load(fh)
            q_lst, db_lst = db_lst[1000:2000], db_lst[2000:12000]

            # 3
            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
            ret_lstses = tool_funcs.normalization([q_lst, db_lst])
            normal_q, normal_db = ret_lstses[0], ret_lstses[1]
            # normal_q, normal_db = q_lst, db_lst
            points = [5, 10, 20, 50]
            hr = [[] for _ in range(4)]
            for i, query in enumerate(querys):
                normal_i = normal_q[i]
                print(i, time.time() - start_time)

                heap_b = []
                for j, database in enumerate(databases):
                    dist = -torch.norm(query - database).item()
                    if len(heap_b) < 50 or dist > heap_b[0][0]:
                        heapq.heappush(heap_b, (dist, j))
                    if len(heap_b) > 50:
                        heapq.heappop(heap_b)
                heap_b = sorted(heap_b, reverse=True)

                heap_a = []
                for j, normal_j in enumerate(normal_db):
                    simi = -fn(normal_i, normal_j)
                    if len(heap_a) < 50 or simi > heap_a[0][0]:
                        heapq.heappush(heap_a, (simi, j))
                    if len(heap_a) > 50:
                        heapq.heappop(heap_a)
                heap_a = sorted(heap_a, reverse=True)

                if i == 0:
                    for value, id in heap_b:
                        print(value, id)
                    print("=======================")
                    for simi, id in heap_a:
                        print(simi, id)

                for idx, point in enumerate(points):
                    cnt = 0
                    for now_i in range(point):
                        for now_j in range(point):
                            if heap_b[now_i][1] == heap_a[now_j][1]:
                                cnt += 1
                    hr[idx].append(cnt)
                    print(i, point, cnt)

        for idx in range(4):
            cnt = 0
            for num in hr[idx]:
                cnt += num
            print(cnt / points[idx])

        return

    def personal_test(self):
        trajs2_emb_cell, trajs2_emb_p, trajs2_len = collate_for_test(
            [[[-968000, 5030500], [-968050, 5030550], [-968100, 5030600], [-968150, 5030650], [-968200, 5030700]]],
            self.cellspace, self.embs)
        traj_embs1 = self.server.model.interpret(trajs2_emb_cell, trajs2_emb_p, trajs2_len)
        trajs2_emb_cell, trajs2_emb_p, trajs2_len = collate_for_test(
            [[[-968200, 5030700], [-968150, 5030650], [-968100, 5030600], [-968050, 5030550], [-968000, 5030500]]],
            self.cellspace, self.embs)
        traj_embs2 = self.server.model.interpret(trajs2_emb_cell, trajs2_emb_p, trajs2_len)

        print(traj_embs1.shape)
        print(traj_embs2.shape)
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cos_sim(traj_embs1[0], traj_embs2[0])
        print(sim)


@torch.no_grad()
def lcss_test():
    logging.info('[Lcss_Test]start.')
    import traj_dist.distance as tdist
    from tqdm import tqdm

    # varying downsampling; varying distort
    vt = Config.test_type
    results = []
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
            q_lst, db_lst = pickle.load(fh)
            querys, databases = q_lst, db_lst
            n_querys = len(querys)
            n_databases = len(databases)
            dists = np.zeros((n_querys, n_databases))
            for i, query in tqdm(enumerate(querys), desc=f'start {vt} lcss count', unit='trajs'):
                for j, database in enumerate(databases):
                    dists[i, j] = tdist.lcss(np.array(query), np.array(database))
            targets = np.diag(dists)  # [1000]
            rank = np.sum(np.sum(dists[:, :n_databases] <= targets[:, np.newaxis], axis=1)) / len(q_lst)
            results.append(rank)
    logging.info(
        '[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
            .format(vt, *results))