from past.builtins import execfile

from copy import deepcopy
from numpy.random import RandomState

from threading import Lock
lock = Lock()

batch_size = 256
epochs = 100
lr_rate = 1e-3

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

            self.train_pb = tq(self.t_loader)
            self.train_pb.set_description("%i Training %i" % (self.thread_id, self.curr_epoch))

            self._train_iter()
            #self.train_pb.close()

            #self.train_pb = tq(self.t_loader)
            #self.train_pb.set_description("%i Analysis %i" % (self.thread_id, self.curr_epoch))

            self._final_train()
            #self.train_pb.close()
        
            lock.acquire()

            curr_scores[self.thread_id] = self.train_loss

            print(curr_scores)
            print(np.mean(curr_scores))

            if self.train_loss < best_loss:
                best_model = deepcopy(self.model.state_dict())
                best_loss = self.train_loss
                best_prev = self.prev
                print('========== %i ===========' % self.thread_id)
                print('best loss %f' % best_loss)
                '''
                self.train_pb = tq(self.v_loader)
                self.train_pb.set_description("%i Test/Valid %i" % (self.thread_id, self.curr_epoch))
                self._valid()
                self.train_pb.close()
                '''
            elif self.train_loss > np.percentile(curr_scores, 50):
                self.model.load_state_dict(best_model)
                self.train_loss = best_loss
                self.prev = best_prev
                print('killed %i' % self.thread_id)

            lock.release()

    def _train_iter(self):

        global best_model

        with torch.no_grad():

            selected_batches = np.random.multinomial(int(len(train_loader)*0.5), self.prev/np.sum(self.prev))
            if self.curr_epoch == 0: selected_batches = self.prev

            tsum_loss = 0.0

            self.model.train(True)

            for i, (batch_features, batch_targets) in enumerate(self.train_pb):

                if (selected_batches[i] == 0): continue

                features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda(self.cid, non_blocking=True)
                targets = batch_targets.float().view(len(batch_targets), -1).cuda(self.cid, non_blocking=True)

                self.checkpoint = deepcopy(self.model.state_dict())

                rseed = random.randint(0, 4294967295)

                rstate = np.random.RandomState(rseed)
                #torch.manual_seed(rseed)
                #torch.cuda.manual_seed(rseed)

                for param in self.model.parameters():
                    #if not param.requires_grad: continue
                    #noise = torch.randn(param.size()) * lr_rate
                    noise = rstate.randn(*param.size()) * lr_rate
                    noise = torch.from_numpy(noise).float()
                    param.add_(noise.cuda(self.cid))

                add_loss = calculate_loss(self.model, features, targets)

                rstate = np.random.RandomState(rseed)
                #torch.manual_seed(rseed)
                #torch.cuda.manual_seed(rseed)

                for param in self.model.parameters():
                    #if not param.requires_grad: continue
                    #noise = torch.randn(param.size()) * lr_rate
                    noise = rstate.randn(*param.size()) * lr_rate
                    noise = torch.from_numpy(noise).float()
                    param.sub_(noise.cuda(self.cid) * 2.0)

                sub_loss = calculate_loss(self.model, features, targets)

                rstate = np.random.RandomState(rseed)
                #torch.manual_seed(rseed)
                #torch.cuda.manual_seed(rseed)

                #for param, best in zip(self.model.parameters(), best_model):
                for param in self.model.parameters():
                    #if not param.requires_grad: continue
                    #noise = torch.randn(param.size()) * lr_rate
                    noise = rstate.randn(*param.size()) * lr_rate
                    noise = torch.from_numpy(noise).float().cuda(self.cid)

                    #explore = (param - best.cuda(self.cid))
                    #explore /= explore.max() * lr_rate

                    if sub_loss < add_loss:
                        potential = self.prev[i] / sub_loss.item()
                    else:
                        potential = self.prev[i] / add_loss.item()

                    # if sub_loss is higher than add noise, vica versa
                    param.add_(noise * (1.0 + (sub_loss - add_loss) * potential))

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

            for i, (batch_features, batch_targets) in enumerate(self.t_loader):

                features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda(self.cid, non_blocking=True)
                targets = batch_targets.float().view(len(batch_targets), -1).cuda(self.cid, non_blocking=True)
                loss = calculate_loss(self.model, features, targets)

                self.prev[i] = loss

                #self.train_pb.set_postfix({'Loss': '{:.3f}'.format(np.mean(self.prev))})

            torch.cuda.empty_cache()

            self.train_loss = np.mean(self.prev)
            #print(self.train_loss)

    def _valid(self):

        with torch.no_grad():

            self.model.eval()

            tsum_loss = 0.0

            for i, (batch_features, batch_targets) in enumerate(self.v_loader):

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
