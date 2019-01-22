from past.builtins import execfile
from copy import deepcopy

batch_size = 512
epochs = 200
lr_rate = 1e-3

execfile('start.py')
execfile('archs.py')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model = Arch3(in_channels = No_Channels, out_channels = No_Channels+3, gap_size = Input_Size).cuda()

bw = np.ones(len(train_loader))

with torch.no_grad():

    for epoch in range(epochs):

        selected_batches = np.random.multinomial(int(len(train_loader)*0.5), bw/np.sum(bw))
        if epoch == 0: selected_batches = bw

        tsum_loss = 0.0

        model.train(True)

        progress_bar = tq(train_loader)
        progress_bar.set_description("Training %i" % epoch)

        noises = []

        for i, (batch_features, batch_targets) in enumerate(progress_bar):

            if (selected_batches[i] == 0): continue

            features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda()
            targets = batch_targets.float().view(len(batch_targets), -1).cuda()

            checkpoint = deepcopy(model.state_dict())

            rseed = np.random.randint(low=0, high=9223372036854775807)

            torch.manual_seed(rseed)
            torch.cuda.manual_seed(rseed)

            for param in model.parameters():
                #if not param.requires_grad: continue
                noise = torch.randn(param.size()) * lr_rate
                param.add_(noise.cuda())

            add_loss = calculate_loss(model, features, targets)

            torch.manual_seed(rseed)
            torch.cuda.manual_seed(rseed)

            for param in model.parameters():
                #if not param.requires_grad: continue
                noise = torch.randn(param.size()) * lr_rate
                param.sub_(noise.cuda() * 2.0)

            sub_loss = calculate_loss(model, features, targets)

            torch.manual_seed(rseed)
            torch.cuda.manual_seed(rseed)

            for param in model.parameters():
                #if not param.requires_grad: continue
                noise = torch.randn(param.size()) * lr_rate

                if sub_loss < add_loss:
                    potential = bw[i] / sub_loss.item()
                else:
                    potential = bw[i] / add_loss.item()

                # if sub_loss is higher than add noise, vica versa
                param.add_(noise.cuda() * (1.0 + (sub_loss - add_loss) * potential))

            loss = calculate_loss(model, features, targets)

            if bw[i] < loss:
                accept_prob = torch.exp(-1.0 * (loss - bw[i]) / loss)

                if random.random() > accept_prob: 
                    model.load_state_dict(checkpoint)
                    loss = bw[i]
            bw[i] = loss

            tsum_loss = tsum_loss + loss.item()

            progress_bar.set_postfix({'Loss': '{:.3f}'.format(np.mean(bw))})

            torch.cuda.empty_cache()

        iter_size = float(np.count_nonzero(selected_batches))

        print(tsum_loss / iter_size)

        model.eval()

        tsum_loss = 0.0

        progress_bar = tq(test_loader)
        progress_bar.set_description("Test/Valid %i" % epoch)

        for i, (batch_features, batch_targets) in enumerate(progress_bar):
            features = batch_features.float().view(len(batch_features), -1, Input_Size).cuda()
            targets = batch_targets.float().view(len(batch_targets), -1).cuda()
            loss = calculate_loss(model, features, targets)
            tsum_loss = tsum_loss + loss.item()
            progress_bar.set_postfix({'Loss': '{:.3f}'.format(tsum_loss / (i+1))})
            torch.cuda.empty_cache()
        print(tsum_loss / (i+1))
