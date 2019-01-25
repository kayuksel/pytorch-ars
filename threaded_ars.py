from past.builtins import execfile

from copy import deepcopy
from numpy.random import RandomState

from threading import Lock
lock = Lock()

batch_size = 256
epochs = 100
lr_rate = 1e-2

no_cpus = 4

execfile('start.py')
execfile('archs.py')

best_model = None
best_loss = 2.0
best_prev = None

curr_scores = np.ones(32) * 2.0

class RandomSearcher(threading.Thread):

    def __init__(self, train_loader, test_loader, thread_id):
        threading.Thread.__init__(self)

        self.cid = thread_id % no_cpus

        self.model = Arch3(in_channels = No_Channels, out_channels = No_Channels+3, gap_size = Input_Size).cuda(self.cid)

        self.backup = Arch3(in_channels = No_Channels, out_channels = No_Channels+3, gap_size = Input_Size).cuda(self.cid)

        self.checkpoint = deepcopy(self.model.state_dict())

        self.prev = np.ones(len(train_loader))

        self.curr_epoch = 0
        self.thread_id = thread_id

        self.t_loader = train_loader
        self.v_loader = test_loader

        self.train_loss = 1.0
        self.valid_loss = 1.0

        self.train_pb = None
        
    def run(self):
        global best_model
        global best_loss
        global best_prev
        global curr_scores

        for epoch in range(epochs):

            self.curr_epoch = epoch

            self.train_pb = tq(self.t_loader, position=self.thread_id)
            self.train_pb.set_description("%i Training %i" % (self.thread_id, self.curr_epoch))
            self._train_iter()
            self.train_pb.close()
            
            
            self.train_pb = tq(self.t_loader, position=self.thread_id)
            self.train_pb.set_description("%i Analysis %i" % (self.thread_id, self.curr_epoch))
            self._final_train()
            self.train_pb.close()

            BestOne = False

            curr_scores[self.thread_id] = self.train_loss
        
            #print(curr_scores)
            #print(np.mean(curr_scores))

            lock.acquire()

            if self.train_loss < best_loss:
                best_model = deepcopy(self.model.state_dict())
                best_loss = self.train_loss
                best_prev = self.prev
                #print('========== %i ===========' % self.thread_id)
                #print('best loss %f' % best_loss)

                BestOne = True
                
            elif self.train_loss > np.percentile(curr_scores, 50):
                # Code for breeding with the best model instead of just cloning it
                '''
                tau = best_loss / (best_loss + self.train_loss)
                
                if best_model is not None: self.backup.load_state_dict(best_model)

                for model_param, target_param in zip(self.model.parameters(), self.backup.parameters()):
                    model_param.data.copy_(tau * model_param.data + (1 - tau) * target_param.data)
                self.train_loss = 2.0
                '''

                # Code for cloning from the best model if the solution is not good
                self.model.load_state_dict(best_model)
                self.train_loss = best_loss

                self.prev = best_prev
                #print('killed %i' % self.thread_id)

            lock.release()

            if BestOne:
                self.train_pb = tq(self.v_loader, position=self.thread_id)
                self.train_pb.set_description("%i Test/Valid %i" % (self.thread_id, self.curr_epoch))
                self._valid()
                self.train_pb.close()

    def _train_iter(self):

        global best_model

        with torch.no_grad():

            selected_batches = np.random.multinomial(int(len(train_loader)*0.5), self.prev/np.sum(self.prev))
            if self.curr_epoch == 0: selected_batches = self.prev

            tsum_loss = 0.0

            self.model.train(True)

            for i, (batch_features, batch_targets) in enumerate(self.train_pb):

                #if (selected_batches[i] == 0): continue

                features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda(self.cid, non_blocking=True)
                targets = batch_targets.float().view(len(batch_targets), -1).cuda(self.cid, non_blocking=True)

                self.checkpoint = deepcopy(self.model.state_dict())
                if best_model is not None: self.backup.load_state_dict(best_model) 

                for model_param, noise_param in zip(self.model.parameters(), self.backup.parameters()):
                    # calculate difference vector to move away
                    diff = model_param.data - noise_param.data
                    #if (diff**2).sum() != 0.0: diff /= diff.norm()

                    noise = torch.randn(model_param.size()).cuda(self.cid)

                    # randomly move away from the best model
                    if best_model is not None:
                        noise *= ((noise.sign() * diff.sign()) >= 0.0).float()
                    noise_norm = noise.norm()
                    if noise_norm != 0.0: noise /= noise.norm()

                    velo = lr_rate * noise
                    #velo = (lr_rate * (noise + diff * 0.2))

                    noise_param.data.copy_(velo)

                for model_param, noise_param in zip(self.model.parameters(), self.backup.parameters()):
                    model_param.add_(noise_param.data)

                add_loss = calculate_loss(self.model, features, targets)


                for model_param, noise_param in zip(self.model.parameters(), self.backup.parameters()):
                    model_param.sub_(noise_param.data * 2.0)
                
                sub_loss = calculate_loss(self.model, features, targets)

                for model_param, noise_param in zip(self.model.parameters(), self.backup.parameters()):
                    model_param.add_(noise_param.data * (1.0 + sub_loss - add_loss))

                loss = calculate_loss(self.model, features, targets)

                if self.prev[i] < loss:
                    # accept the worsening move with e.g. 5% probability
                    accept_prob = torch.exp((self.prev[i] - loss) / loss)

                    # with 95% probability revert back to the checkpoint
                    if random.random() < accept_prob: 
                        self.model.load_state_dict(self.checkpoint)
                        loss = self.prev[i]

                self.prev[i] = loss

                tsum_loss = tsum_loss + loss.item()

                self.train_pb.set_postfix({'Loss': '{:.3f}'.format(np.mean(self.prev))})

            torch.cuda.empty_cache()

            iter_size = float(np.count_nonzero(selected_batches))

            self.train_loss = tsum_loss / iter_size
            #print(self.train_loss)

    def _final_train(self):

        with torch.no_grad():

            self.model.eval()

            for i, (batch_features, batch_targets) in enumerate(self.train_pb):

                features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda(self.cid, non_blocking=True)
                targets = batch_targets.float().view(len(batch_targets), -1).cuda(self.cid, non_blocking=True)
                loss = calculate_loss(self.model, features, targets)

                self.prev[i] = loss

                self.train_pb.set_postfix({'Loss': '{:.3f}'.format(np.mean(self.prev))})

            torch.cuda.empty_cache()

            self.train_loss = np.mean(self.prev)
            #print(self.train_loss)

    def _valid(self):

        with torch.no_grad():

            self.model.eval()

            tsum_loss = 0.0

            for i, (batch_features, batch_targets) in enumerate(self.train_pb):

                features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda(self.cid, non_blocking=True)
                targets = batch_targets.float().view(len(batch_targets), -1).cuda(self.cid, non_blocking=True)

                loss = calculate_loss(self.model, features, targets)
                tsum_loss = tsum_loss + loss.item()

                self.train_pb.set_postfix({'Loss': '{:.3f}'.format(tsum_loss/(i+1))})

            torch.cuda.empty_cache()

            self.valid_loss = tsum_loss / (i+1)
            #print(self.valid_loss)

for i in range(32):
    RandomSearcher(train_loader, test_loader, thread_id = i).start()

while(True):
    pass
