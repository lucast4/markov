## SCRIPT to train context models
import model as M
from model import ContextNet
import torch

## PARAMS
K_list = [3, 3, 3, 3]  # number of states
N_list = [3, 5, 10, 25]  # size of states
Nhid_list = [20, 20, 20, 20] # size of hidden layer
Kc_list = [3, 3, 3, 3] # N contexts
ntrain = 5  # length of sequence
nepochs = 5000 # training epochs
lr = 0.001 # learning rate
ntest = 2000 # testing, to calculate transition matrix

savesuffix_list = ["context", "contextfixfc1"] # can be empty
freezeWeights_list = [False, True]
nrepeats = 2

for K, N, Kc, Nhid in zip(K_list, N_list, Kc_list, Nhid_list):
        Nc = N  # size of context layer
        # Nhid = N  # hidden layer size.
        for _ in range(nrepeats):
                for savesuffix, freezeWeights in zip(savesuffix_list, freezeWeights_list):

                        ## LOAD MODEL
                        if True:
                            net = ContextNet(K, N, Kc, Nc, Nhid, savesuffix)
                        else:
                            model.load_state_dict(state['state_dict'])
                            optimizer.load_state_dict(state['optimizer'])


                        ## RUN
                        Tactual_cont = M.generateTrainDat(net)
                        net, lossall = M.train(net, nepochs, Tactual_cont, lr, ntrain, freezeWeights)
                        Tdat_cont = M.evaluate(net, ntest, Tactual_cont)

                        M.plot(net, Tactual_cont, Tdat_cont, lossall)
                        print(net.savedir)


                        # === save
                        # save model
                        # M.save(net)

                        # ######### SAVE
                        state = {
                                "Kc": Kc,
                                "Nc": Nc,
                                "Nhid": Nhid,
                                "Tactual_cont": Tactual_cont,
                                "Tdat_cont": Tdat_cont,
                                "K": K,
                                "N": N,
                                "ntrain": ntrain,
                                "nepochs": nepochs,
                                "lr": lr,
                                "ntest": ntest,
                                "state_dict": net.state_dict(),
                                "loss": lossall}

                        savename = net.savedir + "/state"
                        torch.save(state, savename)
