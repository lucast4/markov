## SCRIPT to train context models
# from model_heirarchical import ContextNet
import model_heirarchical as Mheir
import model_lstm as M
from model_lstm import LSTM
from model_heirarchical import ContextNet
import matplotlib.pyplot as plt
import model as Morig
import torch

## PARAMS
# K_list = [2, 2, 2, 2]  # number of states
# N_list = [10, 10, 10, 10]  # size of states
# Nhid_list = [50, 50, 50, 50] # size of hidden layer
# Kc_list = [2, 2, 2, 2] # N contexts
# ntrain_list = [2, 5, 10, 20] # length of each sample sequence
# temp_list = [1, 1, 1, 1]
K_list = [4, 4, 3, 8, 8, 12, 15]  # number of states
N_list = [10, 10, 10, 10, 10, 15, 20]  # size of states
Nhid_list = [50, 50, 100, 50, 50, 100, 100] # size of hidden layer
Kc_list = [2, 5, 8, 3, 4, 5, 5] # N contexts
ntrain_list = [100, 150, 150, 100, 100, 150, 200] # length of each sample sequence
# temp_list = [1]

nepochs = 10000 # training epochs
lr = 0.001 # learning rate
ntest = 2000 # testing, to calculate transition matrix

# savesuffix_list       = ["context", "contextfixfc1"] # can be empty
# freezeWeights_list = [False, True]
savesuffix_list = ["lstmtough"] # can be empty
freezeWeights_list = [True]
nrepeats = 2
doBPTT = True
trainUsingContext = False # false means only observe states. True means observes context + state.
doanneal = True # if true, goes from temp to 1e-4. if False then uses just temp throughout.
makedifferent = False # if false, then random matrices. If true, then makes them different [only if 2 contexgts]
feedStateInputs = True # does nothing if training using context. if training from state only, then put this False to not reset input state on each step.
dohack = False # leave False, if True, then does backward() on each timestep.
learncontextslower = False # default False; if true then lowers learning rate for the mixed --> state weights

# ==== do curriculum
docurriculum = False
doFirstTrainWithContext = False

def plotCrossCorr(savename, groundtruth, M, net, ntoaverage=20):
        import numpy as np

        # get mean and sem over many samples
        corrall_all = []
        for n in range(ntoaverage):
                # -- sampel new sequence
                if groundtruth:
                        v, _ = Mheir.sampleSequenceHeir(net_datagenerator, Tcontext,
                                                            Tstates_all, 0, 10000)
                else:
                        v = M.evaluate(net, 10000)

                v = np.array(v)
                corrall = M.plotxcorr(v, net, savename="test", plotON=False, xboundlist=[10])
                corrall_all.append(corrall)
        corrall_all = np.array(corrall_all)
        # --- take average
        corrmeans = corrall_all.mean(axis=0)
        corrsem = corrall_all.std(axis=0) / np.sqrt(corrall_all.shape[0] - 1)

        # -- plot
        nrows = len(set(v))
        ncols = len(set(v))
        for xbound in [10, 40, 80]:
                counter = 1
                plt.figure()

                for i in range(len(corrmeans)):
                        cmean = corrmeans[i]
                        csem = corrsem[i]
                        plt.subplot(nrows, ncols, counter)
                        xmid = len(cmean) / 2
                        x = np.arange(xmid - xbound, xmid + xbound).astype(int)
                        # plt.xlabel("{} -- {}".format(ss, sss))
                        # plt.plot(x, cmean[x], '-k')
                        plt.errorbar(x, cmean[x], yerr=csem[x])
                        # -- set ylim
                        mad = np.diff(np.percentile(cmean[x.astype(int)], (5, 95)))
                        plt.ylim(np.median(cmean[x.astype(int)]) - 1.0 * mad,
                                 np.median(cmean[x.astype(int)]) + 1.0 * mad)
                        ylim = plt.ylim()
                        plt.plot(np.array([xmid, xmid]), np.array(ylim), '--b')
                        counter += 1
                plt.savefig(net.savedir + "/hmm_correlations_xbound{}_{}_trialmean{}.pdf".format(xbound, savename, ntoaverage))
                plt.close()

if __name__ == "__main__":
        for K, N, Kc, Nhid, ntrain in zip(K_list, N_list, Kc_list, Nhid_list, ntrain_list):
                Nc = N  # size of context layer
                # Nhid = N  # hidden layer size.
                for _ in range(nrepeats):
                        for savesuffix, freezeWeights in zip(savesuffix_list, freezeWeights_list):

                                ## LOAD MODEL
                                net = LSTM(K, N, Nhid, savesuffix)
                                net_datagenerator = ContextNet(K, N, Kc, Nc, Nhid, savesuffix, saveoff=True)

                                ## RUN
                                Tstates_all, Tcontext = Mheir.generateTrainDat(net_datagenerator, makedifferent)

                                for epoch in range(nepochs):
                                        if epoch % 100 == 0:
                                                print("epoch {}".format(epoch))

                                        # -- sample new sequence in this epoch
                                        v, _ = Mheir.sampleSequenceHeir(net_datagenerator, Tcontext, Tstates_all, 0, ntrain)

                                        # -- train on this sequence
                                        net, lossall = M.train(net, 1, v, lr)

                                ## EVALUATE
                                vsampled = M.evaluate(net, 10000)
                                vorig, vorig_context = Mheir.sampleSequenceHeir(net_datagenerator, Tcontext, Tstates_all, 0, 10000)
                                M.plot(net, Tcontext, Tstates_all, lossall, vorig, vsampled, vorig_context)

                                # ======= PLOT MANY CROSSCORRELATION
                                plotCrossCorr("truth", True, M, net, ntoaverage=30)
                                plotCrossCorr("modeltrained", False, M, net, ntoaverage=30)



                                # Tdat_cont, dat_freerun = M.evaluate(net, ntest)
                                # for nc in range(Kc):
                                #         print("actual transition matrix: {}".format(Tactual_cont[nc]))
                                #         print("empirical transition matrix: {}".format(Tdat_cont[nc]))
                                # print("actual transition matrix (context): {}".format(Tcontext))
                                # print("empirical transition matrix (context): {}".format(dat_freerun[2]))
                                #
                                # M.plot(net, Tactual_cont, Tcontext, Tdat_cont, dat_freerun, lossall)
                                # print(net.savedir)


                                # === save
                                # save model
                                # M.save(net)

                                # ######### SAVE
                                state = {
                                        "Nhid": Nhid,
                                        "Tstates_all": Tstates_all,
                                        "K": K,
                                        "N": N,
                                        "Kc": Kc,
                                        "Nc": Nc,
                                        "ntrain": ntrain,
                                        "nepochs": nepochs,
                                        "lr": lr,
                                        "ntest": ntest,
                                        "state_dict": net.state_dict(),
                                        "loss": lossall,
                                        "Tcontext": Tcontext}

                                savename = net.savedir + "/state"
                                torch.save(state, savename)

