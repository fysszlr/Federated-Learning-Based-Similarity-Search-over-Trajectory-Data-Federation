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

    def fedprox_local_train(self):
        param = self.model.parameters_to_tensor()
        self.global_model.tensor_to_parameters(param)
        loss_ep_avg = 0
        self.omega = self.model.parameters_to_tensor().clone().detach()
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

                omega_loss = torch.norm(
                    self.model.parameters_to_tensor() - self.omega
                ) ** 2
                loss += omega_loss * 0.25

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

    def moon_local_train(self):
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
                loss_sup = self.model.loss(*model_rtn)

                z1 = self.model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                z1_global = self.global_model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                z1_last = self.last_model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                z2 = self.model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)
                z2_global = self.global_model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)
                z2_last = self.last_model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)

                l1_pos = torch.einsum('nc,nc->n', [z1, z1_global]).unsqueeze(-1)
                l1_neg = torch.einsum('nc,nc->n', [z1, z1_last]).unsqueeze(-1)
                l2_pos = torch.einsum('nc,nc->n', [z2, z2_global]).unsqueeze(-1)
                l2_neg = torch.einsum('nc,nc->n', [z2, z2_last]).unsqueeze(-1)

                logits1 = torch.cat([l1_pos, l1_neg], dim=1)
                logits1 /= self.temperature
                logits2 = torch.cat([l2_pos, l2_neg], dim=1)
                logits2 /= self.temperature

                labels1 = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()
                labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()

                loss1_con = self.criterion(logits1, labels1)
                loss2_con = self.criterion(logits2, labels2)

                loss = loss_sup + (loss1_con + loss2_con) / 2 * Config.moon_loss_weight
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

    def fedproc_local_train(self):
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
                loss_sup = self.model.loss(*model_rtn)

                z1 = self.model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                z1_global = self.global_model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                z1_last = self.last_model.interpret(trajs1_emb, trajs1_emb_p, trajs1_len)
                z2 = self.model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)
                z2_global = self.global_model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)
                z2_last = self.last_model.interpret(trajs2_emb, trajs2_emb_p, trajs2_len)

                l1_pos = torch.einsum('nc,nc->n', [z1, z1_global]).unsqueeze(-1)
                l1_neg = torch.einsum('nc,nc->n', [z1, z1_last]).unsqueeze(-1)
                l2_pos = torch.einsum('nc,nc->n', [z2, z2_global]).unsqueeze(-1)
                l2_neg = torch.einsum('nc,nc->n', [z2, z2_last]).unsqueeze(-1)

                logits1 = torch.cat([l1_pos, l1_neg], dim=1)
                logits1 /= self.temperature
                logits2 = torch.cat([l2_pos, l2_neg], dim=1)
                logits2 /= self.temperature

                labels1 = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()
                labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()

                loss1_con = self.criterion(logits1, labels1)
                loss2_con = self.criterion(logits2, labels2)

                loss = loss_sup * Config.fedproc_loss_weight + (loss1_con + loss2_con) / 2 * (
                            1 - Config.fedproc_loss_weight)

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
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)

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

        self.step += 1

        if self.step % Config.trajcl_training_lr_degrade_step == 0:
            self.learning_rate *= Config.trajcl_training_lr_degrade_gamma

        return clients, loss_ep_sum / self.n_clients_per_round, 0
