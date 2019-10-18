import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.distributions.gumbel import Gumbel
from torch.distributions.categorical import Categorical
import time
import datetime

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("K", type=int)
# parser.add_argument("N", type=int)
# parser.add_argument("N", type=int)
#
# args = parser.parse_args()



# input = Variable(torch.Tensor([[sos]*batch_size])
# hidden, cell = RNN.init_hidden_cell(batch_size)
# for di in range(MAX_LENGTH):
#     output, (hidden,cell) = RNN(input, (hidden,cell))
#     loss += criterion(output, targets[di,:])
#     # for ex of a NLP output:
#     # contains the probabilities for each word IDs.
#     # The index of the probability is the word ID.
#     _, topi = torch.topk(output,1)
#     # topi = (batch*1)
#     input = torch.transpose(topi,0,1)

class ContextNet(nn.Module):
    def __init__(self, K=4, N=10, Kc=3, Nc=10, Nhid=10, savesuffix=""):
        super(ContextNet, self).__init__()
        self.outdim = K
        self.indim = N
        self.Wfix = torch.rand((N, K)) # this is fixed. (from y back to x)
        self.Wfix_context = torch.rand((Nc, Kc))
        self.hiddendim = Nhid

        self.K = K
        self.N = N
        self.Kc = Kc
        self.Nc = Nc
        self.Nhid = Nhid

        # savdir
        import os
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%y-%H%M%S')
        savedir = "saved_models/model_{}st_{}size_{}ctxt_{}hid_{}".format(K, N, Kc, Nhid, tstamp) + "_{}".format(savesuffix)
        os.mkdir(savedir)
        self.savedir = savedir


        # ===== initialize layers
        self.fc1 = nn.Linear(N + Nc, Nhid, bias=True)
        self.fc2 = nn.Linear(Nhid, K, bias=True)


    def initiate(self):
        # get a random y to initiate

        y = torch.zeros(self.K)
        y[0]=1
        return y


    def idx_to_onehot(self, idx, dim):
        one_hot = torch.FloatTensor(dim)
        one_hot.zero_()
        one_hot.scatter_(0, idx, 1)
        return one_hot

    def forward(self, y, c):
        # y is the one-hot input (goal is to predict next output)
        # c is one-hot representing context

        # embed in N-dim
        x_context = torch.matmul(self.Wfix_context, c)
        x = torch.matmul(self.Wfix, y)

        # concatenate token and context
        x = torch.cat([x_context, x])

        # hidden node
        b = torch.tanh(self.fc1(x))

        # linear layer + nonlinearity
        h = torch.tanh(self.fc2(b))

        # add gumbel noise
        z = Gumbel(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size((h.shape[0],)))

        # take argmax
        yind = (h.view((-1,)) + z.view((-1,))).argmax()

        # convert to onehot
        yout = self.idx_to_onehot(yind, self.outdim)

        return yout, h, yind


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


def generateTrainDat(net):
    # Kc = ncontext
    # K = nstate

    # *********** generate training data
    Tactual_cont = []
    for _ in range(net.Kc):
        # generate transition matrix and data
        T = torch.rand(net.K, net.K)
        T = T / T.sum(0)

        # save
        Tactual_cont.append(T)

    return Tactual_cont

def train(net, nepochs, Tactual_cont, lr, ntrain, freezeWeights=False):
    # ------ Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    lossall = []
    net = net.train()
    if freezeWeights:
        for p in net.fc1.parameters():
            p.requires_grad=False
        # - update optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    [print(p) for p in net.fc2.parameters()]


    for epoch in range(nepochs):
        if epoch % 100 == 0:
            print("epoch {}".format(epoch))
        for nc in range(net.Kc):

            # ************ SAMPLE - from ground truth transition matrix
            v = sampleSequence(Tactual_cont[nc], 0, ntrain)
            # ************ TRAINING

            # ------
            # y = net.initiate()
            for i in range(len(v) - 1):
                optimizer.zero_grad()

                y = net.idx_to_onehot(v[i], net.outdim)
                c = net.idx_to_onehot(torch.tensor(nc), net.Kc)
                target = v[i + 1].long()

                y, h, _ = net(y, c)[:3]
                # print(y)
                # print(h)

                loss = criterion(h.unsqueeze(0), target.unsqueeze(0))
                lossall.append(loss.detach())
                # print(loss)
                loss.backward()
                optimizer.step()
    lossall = torch.tensor(lossall)

    if freezeWeights:
        [print(p) for p in net.fc1.parameters()]
    [print(p) for p in net.fc2.parameters()]

    return net, lossall


def evaluate(net, ntest, Tactual_cont):
    net = net.eval()
    Tdat_cont = []
    for nc in range(net.Kc):
        y = net.initiate()
        c = net.idx_to_onehot(torch.tensor(nc), net.Kc)
        yall = []
        for i in range(ntest):
            y, _, yind = net(y, c)
            yall.append(yind)

        # get transition matrix
        yall = np.array(yall)
        Tdat = makeTransMat(yall)
        Tdat_cont.append(Tdat)
        print("actual transition matrix: {}".format(Tactual_cont[nc]))
        print("empirical transition matrix: {}".format(Tdat))
        print("sample: {}".format(yall))
    return Tdat_cont

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

def plot(net, Tactual_cont, Tdat_cont, lossall):
    import matplotlib.pyplot as plt

    # ===== original
    plt.figure()
    # - one subplot for each context
    for nc in range(net.Kc):
        plt.subplot(3,4,2*nc+1)
        plt.title("[actual T] context {}".format(nc))
        t = Tactual_cont[nc]
        plt.imshow(t, vmin=0, vmax=1)

        plt.subplot(3, 4, 2 * nc+2)
        plt.title("[empirical T] context {}".format(nc))
        t = Tdat_cont[nc]
        plt.imshow(t, vmin=0, vmax=1)
    savename = net.savedir + "/transmat.pdf"
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







