import torch.utils.data

from model.fedtrajcl_with_buffer.fed_trainer import *


class Client(BaseClient):
    def __init__(self, id, str_aug1, str_aug2, dataset):
        super().__init__(id, str_aug1, str_aug2, dataset)

    def local_train(self):
        param = self.model.parameters_to_tensor()
        self.global_model.tensor_to_parameters(param)
        loss_ep_avg = 0
        for epoch in range(self.E):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            _time_batch_start = time.time()
            for i_batch, batch in enumerate(self.trainloader):
                _time_batch = time.time()
                self.optimizer.zero_grad()

                trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len = batch

                model_rtn = self.model(trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len)
                loss = self.model.loss(*model_rtn)

                loss *= (np.random.randn() / 100 + 1)

                loss.backward()
                self.optimizer.step()
                loss_ep.append(loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                if i_batch % 100 == 0 and i_batch:
                    logging.debug("[Training] client{} ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                                  .format(self.id, epoch, i_batch, loss.item(), time.time() - _time_batch_start,
                                          tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            # self.scheduler.step()

            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] client{} ep={}: avg_loss={:.3f}, gpu={}, ram={}" \
                         .format(self.id, epoch, loss_ep_avg,
                                 tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)
        param = self.model.parameters_to_tensor()
        self.last_model.tensor_to_parameters(param)
        return loss_ep_avg


class Server(BaseServer):
    def __init__(self, id, str_aug1, str_aug2, dataset):
        super().__init__(id, str_aug1, str_aug2, dataset)
        self.queue = 0
        self.sigma = Config.sigma
        self.weights = [np.expand_dims(p.cpu().detach().numpy(), -1) for p in self.model.parameters()]
        self.num_weights = len(self.weights)
        self.updates = []
        self.median = []
        self.norms = []
        self.clipped_updates = []
        self.m = 0.0
        self.global_model = None

    def compute_updates(self):
        self.updates = [self.weights[i] - np.expand_dims(self.global_model[i], -1) for i in range(self.num_weights)]
        self.weights = [np.expand_dims(p.cpu().detach().numpy(), -1) for p in self.model.parameters()]
        # print("updates:", self.updates)

    def compute_norms(self):
        self.norms = [np.sqrt(np.sum(
            np.square(self.updates[i]), axis=tuple(range(self.updates[i].ndim)[:-1]), keepdims=True)) for i in
            range(self.num_weights)]
        # print("norms:", self.norms)

    def clip_updates(self):
        self.compute_updates()
        self.compute_norms()

        self.median = [np.median(self.norms[i], axis=-1, keepdims=True) for i in range(self.num_weights)]
        # print("median:", self.median)

        factor = [self.norms[i] / self.median[i] for i in range(self.num_weights)]
        # print("old_factor:", factor)

        for i in range(self.num_weights):
            factor[i][factor[i] < 1.0] = 1.0
        # print("new_factor:", factor)

        self.clipped_updates = [self.updates[i] / factor[i] for i in range(self.num_weights)]
        # print("clipped_updates:", self.clipped_updates)

    def aggregate_model(self, clients):
        for client in clients:
            self.weights = [
                np.concatenate((self.weights[i], np.expand_dims(p.cpu().detach().numpy(), -1)), -1)
                for i, p in enumerate(client.model.parameters())]
        self.global_model = [p.cpu().detach().numpy() for p in self.model.parameters()]

        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)

    def aggregate_queue(self, clients):
        self.queue = torch.randn(Config.moco_proj_dim, Config.cls_num * Config.moco_nqueue)
        for i, client in enumerate(clients):
            self.queue[:, i * Config.moco_nqueue: (i + 1) * Config.moco_nqueue] = client.model.clmodel.queue
        for client in clients:
            client.model.clmodel.global_queue = self.queue

    def train(self):
        # random clients
        clients = self.sample_client()

        for client in clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p['lr'] = self.learning_rate

        loss_ep_sum = 0.0
        for client in clients:
            # local train
            loss_ep_sum += client.local_train()

        # aggregate params
        self.aggregate_model(clients)
        self.aggregate_queue(clients)

        self.step += 1

        if self.step % Config.trajcl_training_lr_degrade_step == 0:
            self.learning_rate *= Config.trajcl_training_lr_degrade_gamma

        return clients, loss_ep_sum / self.n_clients_per_round, 0
