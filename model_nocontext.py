import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.distributions.gumbel import Gumbel
from torch.distributions.categorical import Categorical


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("K", type=int)
# parser.add_argument("N", type=int)
# parser.add_argument("N", type=int)
#
# args = parser.parse_args()



## INITIATE PARAMETERS
K = 4 # number of states
N = 10 # size of embedding

# Tactual = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
Tactual = torch.rand(K,K)
Tactual = Tactual/Tactual.sum(0)

ntrain = 5 # length of sequence

# TRAINING
nepochs = 100
lr = 0.001

# TESTING
ntest = 1000

# CONTEXT params
Kc = 3
Nc = N # state size
Nhid = N # hidden layer size.


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
    def __init__(self, K, N, Kc, Nc, Nhid):
        super(ContextNet, self).__init__()
        self.outdim = K
        self.indim = N
        self.Wfix = torch.rand((N, K)) # this is fixed. (from y back to x)

        self.Wfix_context = torch.rand((Nc, Kc))

        self.hiddendim = Nhid

        # ===== initialize layers
        self.fc1 = nn.Linear(N + Nc, Nhid, bias=True)
        self.fc2 = nn.Linear(Nhid, K, bias=True)


    def initiate(self):
        # get a random y to initiate

        y = torch.zeros(K)
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



class Net(nn.Module):

    def __init__(self, K, N):
        super(Net, self).__init__()
        self.outdim = K
        self.indim = N
        self.Wfix = torch.rand((N, K)) # this is fixed.

        # ===== initialize layers
        self.fc = nn.Linear(N, K, bias=False)

    def initiate(self):
        # get a random y to initiate

        y = torch.zeros(K)
        y[0]=1
        return y

    def idx_to_onehot(self, idx, dim):
        one_hot = torch.FloatTensor(dim)
        one_hot.zero_()
        one_hot.scatter_(0, idx, 1)
        return one_hot

    def forward(self, y):
        # y is the one-hot input (goal is to predict next output)

        # embed in N-dim
        x = torch.matmul(self.Wfix, y)

        # linear layer
        h = self.fc(x)

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

def sampleSequence(M, ind1):
    v = [torch.tensor([ind1])]
    for i in range(ntrain):
        v.append(Categorical(M[:, v[i]].view([-1,])).sample())
    v = torch.tensor(v).long()
    return v



if __name__ == '__main__':

    if True:
        # NO CONTEXT
        # ************ SAMPLE - from ground truth transition matrix
        # v = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        #                  dtype=torch.long)  # 3 X 1 tensor
        v = sampleSequence(Tactual, 0)
        print("actual transition matrix: {}".format(Tactual))
        print("sample: {}".format(v))

        # ************ TRAINING
        net = Net(K, N)
        print(net)

        params = list(net.parameters())
        print("# params: {}".format(len(params)))


        # ------ Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # ------
        y = net.initiate()
        lossall = []
        for epoch in range(nepochs):
            for i in range(len(v)-1):
                optimizer.zero_grad()

                y = net.idx_to_onehot(v[i], net.outdim)
                # target = net.idx_to_onehot(v[i+1], net.outdim).long()
                target = v[i+1].long()

                y, h, _ = net(y)
                # print(y)
                # print(h)

                loss = criterion(h.unsqueeze(0), target.unsqueeze(0))
                lossall.append(loss.detach())
                # print(loss)
                loss.backward()
                optimizer.step()
        lossall = torch.tensor(lossall)

        # ************ evaluate network (test generation)
        # initiate first token
        y = net.initiate()
        yall = []
        for i in range(ntest):
            y, _, yind = net(y)
            yall.append(yind)


        # get transition matrix
        yall = np.array(yall)
        Tdat = makeTransMat(yall)

        print("empirical transition matrix: {}".format(Tdat))
        print("sample: {}".format(yall))

        print({"loss: {}".format(lossall)})

    else:
        # With context
        # *********** generate training data
        Tactual_cont = []
        for _ in range(Kc):
            # generate transition matrix and data
            T = torch.rand(K, K)
            T = T / T.sum(0)

            # save
            Tactual_cont.append(T)

        # ************ TRAIN - pass sequentially through each context
        # (i.e., not interleaved)
        net = ContextNet(K, N, Kc, Nc, Nhid)
        print(net)

        params = list(net.parameters())
        print("# params: {}".format(len(params)))

        # ------ Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        lossall = []

        for epoch in range(nepochs):
            if epoch % 100 == 0:
                print("epoch {}".format(epoch))
            for nc in range(Kc):

                # ************ SAMPLE - from ground truth transition matrix
                v = sampleSequence(Tactual_cont[nc], 0)
                # ************ TRAINING

                # ------
                # y = net.initiate()
                for i in range(len(v) - 1):
                    optimizer.zero_grad()

                    y = net.idx_to_onehot(v[i], net.outdim)
                    c = net.idx_to_onehot(torch.tensor(nc), Kc)
                    target = v[i + 1].long()

                    y, h, _ = net(y, c)
                    # print(y)
                    # print(h)

                    loss = criterion(h.unsqueeze(0), target.unsqueeze(0))
                    lossall.append(loss.detach())
                    # print(loss)
                    loss.backward()
                    optimizer.step()
        lossall = torch.tensor(lossall)



        # ************ evaluate network (test generation)
        # initiate first token
        Tdat_cont = []
        for nc in range(Kc):
            y = net.initiate()
            c = net.idx_to_onehot(torch.tensor(nc), Kc)
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

        # ===== PLOT transition matrices
        data = {"Kc": Kc,
                  "Nc": Nc,
                  "Nhid": Nhid,
                  "Tactual_cont": Tactual_cont,
                  "Tdat_cont": Tdat_cont,
                  "K": K,
                  "N": N,
                  "ntrain": ntrain,
                  "nepochs": nepochs,
                  "lr": lr,
                  "ntest": ntest}

        # save dir
        import time
        import os
        savedir = "saved_models/model_{}st_{}size_{}ctxt_{}".format(K, N, Kc, time.time())
        os.mkdir(savedir)
        # -- params
        savename = savedir + "/data.txt"
        with open(savename, 'w') as f:
            print(data, file=f)

        # f = open(savename, "w")
        # f.write(params)
        # f.close()
        # -- model
        # savename = savedir + "/model.txt"
        # f = open(savename, "w")
        # f.write(str(savename))
        # f.close()

        # --- make some figures
        import matplotlib.pyplot as plt

        # ===== original
        plt.figure()
        # - one subplot for each context
        for nc in range(Kc):
            plt.subplot(2,3,2*nc+1)
            plt.title("[actual T] context {}".format(nc))
            t = Tactual_cont[nc]
            plt.imshow(t, vmin=0, vmax=1)

            plt.subplot(2, 3, 2 * nc+2)
            plt.title("[empirical T] context {}".format(nc))
            t = Tdat_cont[nc]
            plt.imshow(t, vmin=0, vmax=1)
        savename = savedir + "/transmat.pdf"
        plt.savefig(savename)

        # ==== plot loss function
        plt.figure()
        # x = np.arange(0,100)
        plt.plot(lossall[0:100], '.k')
        plt.title('loss')
        plt.xlabel("iteration")
        plt.savefig(savedir + "/loss.pdf")

        print({"loss: {}".format(lossall)})







