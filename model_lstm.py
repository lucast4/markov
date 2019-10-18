import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.distributions.gumbel import Gumbel
from torch.distributions.categorical import Categorical
import time
import datetime
from torch.nn import Parameter
import model_heirarchical as MH
import matplotlib.pyplot as plt

from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical, ExpRelaxedCategorical

## 8/17/19 - also models transitions between contexts.
class LSTM(nn.Module):
    def __init__(self, K=4, N=10, Nhid=10, savesuffix=""):
        super(LSTM, self).__init__()
        self.Wfix = Parameter(torch.rand((N, K))*torch.tensor(np.sqrt(1/K)), requires_grad=False) # this is fixed. (from y back to x)

        self.K = K # states
        self.N = N # size of state
        self.Nhid = Nhid # size of hidden layer.

        # savedir
        import os
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%y-%H%M%S')
        savedir = "saved_models/model_{}st_{}size_{}hid_{}".format(K, N, Nhid, tstamp) + "_{}".format(savesuffix)
        os.mkdir(savedir)
        self.savedir = savedir

        # ===== initialize layers
        self.lstm = nn.LSTMCell(N, Nhid)
        self.fc1 = nn.Linear(Nhid, K, bias=True) # state --> mixed

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        hx = torch.randn((1, self.Nhid))
        cx = torch.randn((1, self.Nhid)) # TODO: these correct?
        return (hx, cx)

    def initiate(self):
        # get a random y to initiate
        y = torch.zeros(self.K)
        y[torch.randint(0, self.K, (1,))] = 1

        return y

    def forward(self, y, hx, cx):
        # y is the one-hot input (goal is to predict next output)
        # hx, cx are hidden and cell state from previous timepoint.

        # --- forward pass
        x_state = torch.matmul(self.Wfix, y)
        hx, cx = self.lstm(x_state[None, :], (hx, cx)) # TODO: check shapes of hx input and output

        # --- compress into a one-hot layer
        h = self.fc1(hx.reshape(-1,))

        # ==== get one-hot output
        z = Gumbel(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size((h.shape[0],)))  # add gumbel noise
        yind = (h.view((-1,)) + z.view((-1,))).argmax()  # take argmax
        yonehot = self.idx_to_onehot(yind, self.K)  # convert to onehot

        return yonehot, yind, h, hx, cx

    def idx_to_onehot(self, idx, dim):
        one_hot = torch.FloatTensor(dim)
        one_hot.zero_()
        one_hot.scatter_(0, idx, 1)
        return one_hot

def train(net, nepochs, v, lr): #
    # trainUsingContext, if True, then will supervise with both state and context signals.
    # v is training data
    # ------ Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    lossall = []
    net = net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # ====== TRAIN
    lossall = []
    for epoch in range(nepochs):
        # if epoch % 100 == 0:
        #     print("epoch {}".format(epoch))

        # y = net.initiate()
        hx, cx = net.init_hidden() # TODO: generate input hidden vector states

        # ************ TRAINING
        optimizer.zero_grad()
        loss = torch.Tensor([0.0])
        for i in range(len(v) - 1):
            target = v[i + 1].long()
            y = net.idx_to_onehot(v[i], net.K) # TODO: pull in the correct package

            # then don't care about c being differentiable.
            _, _, h, hx, cx = net(y, hx, cx) # TODO : check this.

            loss += criterion(h.unsqueeze(0), target.unsqueeze(0))

        lossall.append(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # this would only keep gradients due to prediction on this

    return net, lossall

def evaluate(net, ntest):
    net = net.eval()
    # # ==== 1) fix context, extract transition matrices
    # Tdat_cont = []
    # for nc in range(net.Kc):
    #     y = net.initiate()
    #     c = net.idx_to_onehot(torch.tensor(nc), net.Kc)
    #     yall = []
    #     for i in range(ntest):
    #         y, _, yind, _, _, _ = net(y, c)
    #         yall.append(yind)
    #
    #     # get transition matrix
    #     yall = np.array(yall)
    #     Tdat = makeTransMat(yall)
    #     Tdat_cont.append(Tdat)

    ## GENERATE SEQUENCE, starting from random input
    hx, cx = net.init_hidden()
    y = net.initiate()
    y_all = []
    for i in range(ntest):

        # then don't care about c being differentiable.
        y, yind, _, hx, cx = net(y, hx, cx)

        y_all.append(yind)
    y_all = np.array(y_all)


    #
    #
    # # ==== 2) sample with no constraints - what is context dynamics?
    # y = net.initiate()
    # # c = net.initiateContext()
    # # c_all = []
    # y_all = []
    # for i in range(ntest):
    #     y, _, yind, c, _, cind = net(y, c)
    #     c_all.append(cind)
    #     y_all.append(yind)
    # c_all = np.array(c_all)
    # y_all = np.array(y_all)
    #
    # # get transition matric for c
    # Tcontext_empirical = makeTransMat(c_all)
    # dat_freerun = (c_all, y_all, Tcontext_empirical)

    return y_all

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

def plot(net, Tcontext, Tstates_all, lossall, v, vsampled, vorig_context=np.empty(0)):
    import matplotlib.pyplot as plt

    v = np.array(v)
    vsampled = np.array(vsampled)
    vorig_context = np.array(vorig_context)

    # ===== original
    plt.figure()
    # - one subplot for each context
    for nc in range(len(Tstates_all)):
        plt.subplot(4,5,2*nc+1)
        plt.title("[actual T] context {}".format(nc))
        t = Tstates_all[nc]
        plt.imshow(t, vmin=0, vmax=1, cmap="plasma")

        # plt.subplot(3, 4, 2 * nc+2)
        # plt.title("[empirical T] context {}".format(nc))
        # t = Tdat_cont[nc]
        # plt.imshow(t, vmin=0, vmax=1, cmap="plasma")
    savename = net.savedir + "/transmat.pdf"
    plt.savefig(savename)
    plt.close()

    # ---- plot context transitions
    plt.figure()
    plt.subplot(2,3,1)
    plt.title('actual context dynamics')
    plt.imshow(Tcontext, vmin=0, vmax=1, cmap="plasma")

    # plt.subplot(2,3,2)
    # plt.title('empirical context dynamics')
    # plt.imshow(dat_freerun[2], vmin=0, vmax=1, cmap="plasma")

    savename = net.savedir + "/transmat_context.pdf"
    plt.savefig(savename)
    plt.close()

    # ==== plot loss function
    plt.figure()
    # x = np.arange(0,100)
    plt.plot(lossall, '.k')
    N = 200
    lossall_sm = np.convolve(lossall, np.ones((N,)) / N, mode='same')
    plt.plot(lossall_sm, '.r')
    plt.title('loss')
    plt.xlabel("iteration")
    plt.savefig(net.savedir + "/loss.pdf")
    plt.close()

    plt.figure()
    # x = np.arange(0,100)
    N = 200
    lossall_sm = np.convolve(lossall, np.ones((N,)) / N, mode='valid')
    plt.plot(lossall_sm, '.r')
    plt.title('loss (smoothed), n={}'.format(N))
    plt.xlabel("iteration")
    plt.savefig(net.savedir + "/loss_smoothed.pdf")
    plt.close()

    # ========= Plot state transitions
    nmax = min(len(v), 100)
    plt.figure()
    if vorig_context.any():
        plt.subplot(3,1,1)
        plt.title('ground truth [context]')
        plt.plot(vorig_context[:nmax], 'o-b')

    plt.subplot(3, 1, 2)
    plt.title('ground truth, one sample')
    plt.plot(v[:nmax], 'o-r')

    plt.subplot(3, 1, 3)
    plt.title('sampled from model, trained')
    plt.plot(vsampled[:nmax], 'o-k')

    # plt.subplot(3, 1, 3)
    # plt.title('sampled from model, trained')
    # plt.plot(vsampled[:len(v)], 'o-k')

    plt.savefig(net.savedir + "/sample_sequences.pdf")
    plt.close()

    # ======== GET SOME STATS
    # --- 1) mean transition matrix
    # -- origianl data
    t_actual = MH.makeTransMat(v)
    t_dat = MH.makeTransMat(vsampled)

    plt.figure()

    plt.subplot(2,3,1)
    plt.title('mean T, ignore context [truth]')
    plt.ylabel('to')
    plt.imshow(t_actual, vmin=0, vmax=1, cmap="plasma")

    plt.subplot(2,3,2)
    plt.title('mean T, ignore context [sampled]')
    plt.ylabel('to')
    plt.imshow(t_dat, vmin=0, vmax=1, cmap="plasma")

    plt.subplot(2, 3, 3)
    plt.title('truth - sampled')
    plt.ylabel('to')
    plt.imshow(t_actual-t_dat, vmin=0, vmax=1, cmap="plasma")

    plt.subplot(2,3,4)
    plt.colorbar()

    plt.savefig(net.savedir + "/hmm_meantransmat.pdf")
    plt.close()

    # ---- 2) transition matrices (t --> t+1), conditioned on t-1
    S = set(v)
    plt.figure()
    ncols = Tstates_all[0].shape[0]
    for i, s in enumerate(S):
        ta = MH.makeTransMat(v, s)
        td = MH.makeTransMat(vsampled, s)

        plt.subplot(ncols,2, 2*i+1)
        plt.title("actual, s_t-1={}".format(s))
        plt.xlabel("t")
        plt.ylabel("t+1")
        plt.imshow(ta, vmin=0, vmax=1, cmap="plasma")

        plt.subplot(ncols,2, 2*i+2)
        plt.title("trained model, s_t-1={}".format(s))
        plt.xlabel("t")
        plt.ylabel("t+1")
        plt.imshow(td, vmin=0, vmax=1, cmap="plasma")
    plt.savefig(net.savedir + "/hmm_meantrans_conditioned.pdf")
    plt.close()

    # ---- 3) correlation structure
    # --- get signal (0 0 1 0 1 ...) for each state

    # SAMPLED
    vthis = v
    plotxcorr(vthis, net, savename="truth")

    vthis = vsampled
    plotxcorr(vthis, net, savename="trainedmodel")


def plotxcorr(vthis, net, savename, plotON=True, xboundlist=[10]):
    from scipy.signal import correlate
    # vthis is the sequence of tokens you want to plot
    vboolean, S = sequenceToBoolean(vthis)
    ncols = len(S)
    corrall = []
    for xbound in xboundlist:
        counter = 1
        if plotON:
            plt.figure()
        for vv, ss in zip(vboolean, S):
            for vvv, sss in zip(vboolean, S):
                #  get correlation
                corr = correlate(vv, vvv, mode="same")
                corrall.append(corr)
                # x values to plot
                xmid = len(corr) / 2
                x = np.arange(xmid - xbound, xmid + xbound)
                # plot
                if plotON:
                    plt.subplot(ncols, ncols, counter)
                    plt.xlabel("{} -- {}".format(ss, sss))
                    counter += 1
                    plt.plot(x, corr[x.astype(int)], '-k')
                # -- set ylim
                    mad = np.diff(np.percentile(corr[x.astype(int)], (5, 95)))
                    plt.ylim(np.median(corr[x.astype(int)]) - 1.5 * mad, np.median(corr[x.astype(int)]) + 1.5 * mad)
                    ylim = plt.ylim()
                    plt.plot(np.array([xmid, xmid]), np.array(ylim), '--b')
        if plotON:
            plt.savefig(net.savedir + "/hmm_correlations_xbound{}_{}.pdf".format(xbound, savename))
            plt.close()
    return corrall


def sequenceToBoolean(v):
    "given sequence of state np.array([1, 2, 0, 0, 1, ...]), convert to N x D array of 0 and 1, where N is nmber of states, D is length"
    S = set(v)
    vboolean = [[(v==s).astype(int)] for s in S]
    vboolean = np.array(vboolean)
    vboolean = vboolean.reshape((vboolean.shape[0], vboolean.shape[2]))

    return vboolean, S





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







