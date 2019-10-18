import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.distributions.gumbel import Gumbel
from torch.distributions.categorical import Categorical
import time
import datetime
from torch.nn import Parameter
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical, ExpRelaxedCategorical

## 8/17/19 - also models transitions between contexts.


class ContextNet(nn.Module):
    def __init__(self, K=4, N=10, Kc=3, Nc=10, Nhid=10, savesuffix=""):
        super(ContextNet, self).__init__()
        self.outdim = K
        self.indim = N
        self.Wfix = Parameter(torch.rand((N, K)), requires_grad=False) # this is fixed. (from y back to x)
        self.Wfix_context = Parameter(torch.rand((Nc, Kc)), requires_grad=False)
        self.hiddendim = Nhid

        self.K = K # states
        self.N = N # size of state
        self.Kc = Kc # contexts
        self.Nc = Nc # size of contexts`
        self.Nhid = Nhid # size of hidden layer.

        # savdir
        import os
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%y-%H%M%S')
        savedir = "saved_models/model_{}st_{}size_{}ctxt_{}hid_{}".format(K, N, Kc, Nhid, tstamp) + "_{}".format(savesuffix)
        os.mkdir(savedir)
        self.savedir = savedir


        # ===== initialize layers
        self.fc1 = nn.Linear(N + Nc + Nhid, Nhid, bias=True) # context + state --> mixed # TODO: modify so that feedback weights are all 1?
        self.fc2 = nn.Linear(Nhid, K, bias=True) # mixed --> state(onehot)
        self.fc3 = nn.Linear(Nhid, Kc, bias=True) # mixed --> context(onehot)
        # self.fc4 = nn.Linear(Nhid, Nhid, bias=True) # mixed --> mixed


    def initiate(self):
        # get a random y to initiate
        y = torch.zeros(self.K)
        y[0]=1
        return y

    def initiateContext(self):
        # get a random c to initiate
        c = torch.zeros(self.Kc)
        c[torch.randint(0, self.Kc, (1,))]=1
        return c

    def initiate_mixed(self):
        m = torch.randn((self.Nhid, ))
        # c[torch.randint(0, self.Kc, (1,))] = 1
        return m

    def idx_to_onehot(self, idx, dim):
        one_hot = torch.FloatTensor(dim)
        one_hot.zero_()
        one_hot.scatter_(0, idx, 1)
        return one_hot


    def forward(self, y, c, m, useGumbel=True, temp=0.5): # TODO: add m to all things calling this.
        # y is the one-hot input (goal is to predict next output)
        # c is one-hot representing context
        # useGumbel=True, then uses Gumbel instead of softmax

        # embed in N-dim
        x_context = torch.matmul(self.Wfix_context, c)
        x_state = torch.matmul(self.Wfix, y)

        # concatenate token and context
        x = torch.cat([x_context, x_state, m]) # TODO: confirm works

        # hidden node
        b = torch.tanh(self.fc1(x))

        # ---- STATE
        if useGumbel: # will not get gradients for fc3 if do this.
            h = torch.tanh(self.fc2(b)) # linear layer + nonlinearity
            z = Gumbel(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size((h.shape[0],))) # add gumbel noise
            yind = (h.view((-1,)) + z.view((-1,))).argmax() # take argmax
            yout = self.idx_to_onehot(yind, self.outdim) # convert to onehot

            # ---- CONTEXT
            h_context = torch.tanh(self.fc3(b))
            z_context = Gumbel(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size((h_context.shape[0],))) # add gumbel noise
            c_ind = (h_context.view((-1,)) + z_context.view((-1,))).argmax() # take argmax
            c_out = self.idx_to_onehot(c_ind, self.Kc)
        else:
            h = torch.tanh(self.fc2(b)) # linear layer + nonlinearity
            yout = RelaxedOneHotCategorical(temp, logits=h).rsample()
            # yout = self.idx_to_onehot(yind, self.outdim) # convert to onehot
            yind = []

            # ---- CONTEXT
            h_context = torch.tanh(self.fc3(b))
            c_out = RelaxedOneHotCategorical(temp, logits=h_context).rsample()
            c_ind = []

        m = b # rename as m
        return yout, h, yind, c_out, h_context, c_ind, m


def makeTransMat(v):
    # v is sequence
    # Mji is from i to j.
    n = len(set(v))
    M = np.zeros((n,n))

    for (i, j) in zip(v, v[1:]):
        M[j][i] += 1
    M = M/M.sum(axis=0)
    return M

def sampleSequence(M, ind1, ntrain):
    v = [torch.tensor([ind1])]
    for i in range(ntrain):
        v.append(Categorical(M[:, v[i]].view([-1,])).sample())
    v = torch.tensor(v).long()
    return v

def sampleSequenceHeir(net, Tcontext, Tstates_all, ind1, ntrain):
    # first sample a sequence of contexts. Then sample states conditioned on those contexts
    # give an ind 1 to initiate sampling. will not include first ind.

    # - context
    cc = [torch.randint(0, net.Kc, (1,))]
    for i in range(ntrain):
        cc.append(Categorical(Tcontext[:, cc[i]].view([-1,])).sample())

    # - samples
    # note: context at t influences transition from state at t to t+1
    # v = [torch.tensor([ind1])]
    v = [torch.randint(0, net.K, (1,))]
    for i in range(ntrain):
        # - which transition matrix to use? [pick out the previous state and context]
        cthis = cc[i]
        statethis = v[i]
        Tthis = Tstates_all[cthis]

        tmp = Tthis[:, statethis]
        tmp = tmp.view([-1,])
        v.append(Categorical(tmp).sample())

    # --- delete the first, since they were random.
    del cc[0]
    cc = torch.tensor(cc).long()
    del v[0]
    v = torch.tensor(v).long()

    return v, cc





def generateTrainDat(net, makedifferent=False):
    # Kc = ncontext
    # K = nstate
    # outputs one context dynaimcs matrix and one trans matrix for each context
    # context should be slower - i.e., larger on diagonal.
    if makedifferent:
        assert(net.Kc==2)
        assert(net.K == 2 | net.K==3)

    a = 0.6
    t1 = torch.rand((net.Kc,))*(1-a)*torch.eye(net.Kc) + a*torch.eye(net.Kc)
    t2 = torch.rand(net.Kc, net.Kc)
    t2 = (torch.eye(net.Kc) - t1).sum(0)*t2/t2.sum(0)
    Tcontext = t1 + t2
    Tcontext = Tcontext/Tcontext.sum(0)


    # *********** generate training data
    Tactual_cont = []
    for _ in range(net.Kc):
        # generate transition matrix and data
        T = torch.rand(net.K, net.K)
        T = T / T.sum(0)

        # save
        Tactual_cont.append(T)

    # ======== FOR DEBUGGING, ENTER INFORMATION BY HAND
    if makedifferent:
        Tcontext = torch.Tensor([[0.9, 0.1], [0.1, 0.9]])
        if net.K==2:
            Tactual_cont = [torch.Tensor([[0.7, 0.3], [0.3, 0.7]]), torch.Tensor([[0.3, 0.7], [0.7, 0.3]])]
        elif net.K==3:
            Tactual_cont = [torch.Tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), \
                            torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])]

    return Tactual_cont, Tcontext



def train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights=False, doBPTT=True,
          trainUsingContext=False, temp=0.5, doanneal=False, learncontextslower=False,
          feedInputStates=True, mintemp = 5e-1, hack=False, clampcontext=(False, 0), onlyModifyContext=False): #
    # trainUsingContext, if True, then will supervise with both state and context signals.
    # if doanneal, then temp is the max temp. will go down from that.
    # ------ Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    lossall = []
    net = net.train()

    if clampcontext[0]:
        # then clamp context - do this by acting like training using context.
        trainUsingContext=True


    if freezeWeights:
        for p in net.fc1.parameters():
            p.requires_grad=False
        # - update optimizer
        if learncontextslower: # TODO: note, I think I should not do this - I think this leads to poor learning of state (i.e. same state for all context)
            optimizer = torch.optim.Adam([
                {'params': net.fc3.parameters()},
                {'params': net.fc2.parameters(), 'lr': lr/50}
            ], lr=lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
            if learncontextslower:  # TODO: note, I think I should not do this - I think this leads to poor learning of state (i.e. same state for all context)
                optimizer = torch.optim.Adam([
                    {'params': net.fc1.parameters()},
                    {'params': net.fc3.parameters()},
                    {'params': net.fc2.parameters(), 'lr': lr / 100}
                ], lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    [print(p) for p in net.fc2.parameters()]

    if onlyModifyContext:
        optimizer = torch.optim.Adam([
            {'params': net.fc3.parameters()}], lr=lr)


    if doanneal:
        # --- precalcualte the scaling factor: temp = temp_0 * (scale)^(epoch)
        lograte = np.log((mintemp)/temp)/nepochs # 1e-4 is the final temp that should converge to. temp is start
        annealrate = np.exp(lograte) #

    # ====== SAMPLE
    for epoch in range(nepochs):
        if epoch % 100 == 0:
            print("epoch {}".format(epoch))

        # === for each epoch, do one round of backprop, using one sample.
        if trainUsingContext:
            v, v_context = sampleSequenceHeir(net, Tcontext, Tactual_cont, 0, ntrain)
        else:
            v, _ = sampleSequenceHeir(net, Tcontext, Tactual_cont, 0, ntrain)

        c = net.initiateContext() # just iniitate at seomthing random
        y = net.initiate()
        m = net.initiate_mixed() # TODO: write this

        if doanneal:
            # then decrease temperature over training
            tempthis = temp * (np.power(annealrate, epoch))
        else:
            # then just use the given temp
            tempthis = temp

        # ************ TRAINING
        optimizer.zero_grad()
        loss = torch.Tensor([0.0])
        for i in range(len(v) - 1):

            if trainUsingContext:
                # --- feed inputs and prepare outputs
                y = net.idx_to_onehot(v[i], net.outdim)
                target = v[i + 1].long()
                c = net.idx_to_onehot(v_context[i], net.Kc)
                target_c = v_context[i+1].long()

                # ---
                _, h_state, _, _, h_context, _, m = net(y, c, m)

                # -- loss function is due to both state and context
                loss += criterion(h_state.unsqueeze(0), target.unsqueeze(0)) + \
                       criterion(h_context.unsqueeze(0), target_c.unsqueeze(0))
            else:
                target = v[i + 1].long()
                if feedInputStates:
                    y = net.idx_to_onehot(v[i], net.outdim)

                if doBPTT:
                    # then have output be gumbel-softmax approxiamtion of categorical variable
                    # so that can take grdients over entire sequence
                    y, h_state, _, c, _, _, m = net(y, c, m, useGumbel=False, temp=tempthis)
                else:
                    # then don't care about c being differentiable.
                    y, h_state, _, c, _, _, m = net(y, c, m)

                # --- loss function is entirely due to state (context must be inferred)
                if i>0: # TODO: set breakpoint here and then test the outputs of above and how depend on whether usegumbel
                    # since the first iteration will not have gradients for the wieght from mixture to output
                    loss += criterion(h_state.unsqueeze(0), target.unsqueeze(0))

            if hack and i>0:
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()  # this would only keep gradients due to prediction on this round.

            lossall.append(loss.detach())
            # print(loss)
            if not hack:
                if not doBPTT:
                    # then do not do backprop thru time
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()  # this would only keep gradients due to prediction on this round.
        if not hack:
            if doBPTT:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  # this would only keep gradients due to prediction on this

    lossall = torch.tensor(lossall)

    if freezeWeights:
        [print(p) for p in net.fc1.parameters()]
    [print(p) for p in net.fc2.parameters()]

    return net, lossall


def evaluate(net, ntest):
    net = net.eval()
    # ==== 1) fix context, extract transition matrices
    Tdat_cont = []
    for nc in range(net.Kc):
        y = net.initiate()
        c = net.idx_to_onehot(torch.tensor(nc), net.Kc)
        m = net.initiate_mixed()
        yall = []
        for i in range(ntest):
            y, _, yind, _, _, _, m = net(y, c, m)
            yall.append(yind)

        # get transition matrix
        yall = np.array(yall)
        Tdat = makeTransMat(yall)
        Tdat_cont.append(Tdat)

    # ==== 2) sample with no constraints - what is context dynamics?
    y = net.initiate()
    c = net.initiateContext()
    m = net.initiate_mixed()
    c_all = []
    y_all = []
    for i in range(ntest):
        y, _, yind, c, _, cind, m = net(y, c, m)
        c_all.append(cind)
        y_all.append(yind)
    c_all = np.array(c_all)
    y_all = np.array(y_all)

    # get transition matric for c
    Tcontext_empirical = makeTransMat(c_all)
    dat_freerun = (c_all, y_all, Tcontext_empirical)

    return Tdat_cont, dat_freerun

def save(net):

    # data = {"Kc": Kc,
    #           "Nc": Nc,
    #           "Nhid": Nhid,
    #           "Tactual_cont": Tactual_cont,
    #           "Tdat_cont": Tdat_cont,
    #           "K": K,
    #           "N": N,
    #           "ntrain": ntrain,
    #           "nepochs": nepochs,
    #           "lr": lr,
    #           "ntest": ntest}

    # # save dir
    # # -- params
    # savename = savedir + "/data.txt"
    # with open(savename, 'w') as f:
    #     print(data, file=f)

    savepath = net.savedir + "/model"
    torch.save(net.state_dict(), savepath)

def plot(net, Tactual_cont, Tcontext, Tdat_cont, dat_freerun, lossall):
    import matplotlib.pyplot as plt

    # ===== original
    plt.figure()
    # - one subplot for each context
    for nc in range(net.Kc):
        plt.subplot(3,4,2*nc+1)
        plt.title("[actual T] context {}".format(nc))
        t = Tactual_cont[nc]
        plt.imshow(t, vmin=0, vmax=1, cmap="plasma")

        plt.subplot(3, 4, 2 * nc+2)
        plt.title("[empirical T] context {}".format(nc))
        t = Tdat_cont[nc]
        plt.imshow(t, vmin=0, vmax=1, cmap="plasma")
    savename = net.savedir + "/transmat.pdf"
    plt.savefig(savename)
    plt.close()

    # ===== plot difference between contexts
    if len(Tdat_cont)==2:
        tdiff_emp = Tdat_cont[1] - Tdat_cont[0]
        tdiff_actual = Tactual_cont[1] - Tactual_cont[0]
        plt.figure()

        plt.subplot(2,3,1)
        plt.title('diff of T (c1-c0), actual')
        plt.imshow(tdiff_actual, vmin=0, vmax=1, cmap="plasma")

        plt.subplot(2,3,2)
        plt.title('empirical')
        plt.imshow(tdiff_emp, vmin=0, vmax=1, cmap="plasma")

        savename = net.savedir + "/transmat_diffacrosscont.pdf"
        plt.savefig(savename)
        plt.close()


    # ---- plot context transitions
    plt.figure()
    plt.subplot(2,3, 1)
    plt.title('actual context dynamics')
    plt.imshow(Tcontext, vmin=0, vmax=1, cmap="plasma")

    plt.subplot(2,3,2)
    plt.title('empirical context dynamics')
    plt.imshow(dat_freerun[2], vmin=0, vmax=1, cmap="plasma")

    savename = net.savedir + "/transmat_context.pdf"
    plt.savefig(savename)
    plt.close()


    # ==== plot loss function
    plt.figure()
    # x = np.arange(0,100)
    plt.plot(lossall, '.k')
    N = 100
    lossall_sm = np.convolve(lossall, np.ones((N,)) / N, mode='same')
    plt.plot(lossall_sm, '.r')
    plt.title('loss')
    plt.xlabel("iteration")
    plt.savefig(net.savedir + "/loss.pdf")
    plt.close()

    plt.figure()
    # x = np.arange(0,100)
    N = 100
    lossall_sm = np.convolve(lossall, np.ones((N,)) / N, mode='valid')
    plt.plot(lossall_sm, '.r')
    plt.title('loss (smoothed), n={}'.format(N))
    plt.xlabel("iteration")
    plt.savefig(net.savedir + "/loss_smoothed.pdf")
    plt.close()


if __name__ == '__main__':
    ## ************************* INITIATE PARAMETERS
    K = 4  # number of states
    N = 10  # size of embedding
    ntrain = 5  # length of sequence

    # TRAINING
    nepochs = 100
    lr = 0.001

    # TESTING
    ntest = 1000

    # CONTEXT params
    Kc = 3
    Nc = N  # state size
    Nhid = N  # hidden layer size.

    # ************ TRAIN - pass sequentially through each context
    # (i.e., not interleaved)
    net = ContextNet(K, N, Kc, Nc, Nhid)

    ## GENERATE training data
    Tactual_cont = generateTrainDat(net)

    if False:
        print(net)
        params = list(net.parameters())
        print("# params: {}".format(len(params)))


    ## TRAIN
    train(net, nepochs, Tactual_cont)

    # ************ evaluate network (test generation)
    # GET EMPIRICAL TRANSITION MATRIX
    Tdat_cont = evaluate(net, ntest)

    # ===== PLOT transition matrices
    save(net)

    # f = open(savename, "w")
    # f.write(params)
    # f.close()
    # -- model
    # savename = savedir + "/model.txt"
    # f = open(savename, "w")
    # f.write(str(savename))
    # f.close()

    # --- make some figures
    plot(net, Tactual_cont, Tdat_cont, lossall, savedir)

    print({"loss: {}".format(lossall)})







