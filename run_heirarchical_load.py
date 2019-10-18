## SCRIPT, extending from run_heirarchical, but loading specific model to do analysis on
import model_heirarchical as M
from model_heirarchical import ContextNet
import model as Morig
import torch

# modelpath = "/Users/mbl/Google Drive/SCIENCE/Professional/Workshops/MCN 2019/DURING/Project/markov/saved_models/contextheir_fullyobserved/model_3st_10size_2ctxt_40hid_180819-124410_contextheir_trainseecontext"
modelpath = "/Users/mbl/Google Drive/SCIENCE/Professional/Workshops/MCN 2019/DURING/Project/markov/saved_models/contextheir_hmm/model_3st_10size_2ctxt_40hid_180819-111727_contextheir"

## PARAMS
# K_list = [2, 2, 2, 2]  # number of states
# N_list = [10, 10, 10, 10]  # size of states
# Nhid_list = [50, 50, 50, 50] # size of hidden layer
# Kc_list = [2, 2, 2, 2] # N contexts
# ntrain_list = [2, 5, 10, 20] # length of each sample sequence
# temp_list = [1, 1, 1, 1]
K_list = [3]  # number of states
N_list = [10]  # size of states
Nhid_list = [40] # size of hidden layer
Kc_list = [2] # N contexts
ntrain_list = [10] # length of each sample sequence
temp_list = [1]
nepochs = 1000 # training epochs
lr = 0.001 # learning rate
ntest = 2000 # testing, to calculate transition matrix

# savesuffix_list       = ["context", "contextfixfc1"] # can be empty
# freezeWeights_list = [False, True]
savesuffix_list = ["test"] # can be empty
freezeWeights_list = [True]
nrepeats = 1
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

for K, N, Kc, Nhid, ntrain, temp in zip(K_list, N_list, Kc_list, Nhid_list, ntrain_list, temp_list):
        Nc = N  # size of context layer
        # Nhid = N  # hidden layer size.
        for _ in range(nrepeats):
                for savesuffix, freezeWeights in zip(savesuffix_list, freezeWeights_list):

                        ## LOAD MODEL
                        net = ContextNet(K, N, Kc, Nc, Nhid, savesuffix)
                        state = torch.load(modelpath + "/state")
                        net.load_state_dict(state['state_dict'])

                        ## print things about state dict
                        for p in state:
                                print(p)

                        ## generate a sequence - record neural activity during
                        Tactual_cont = state["Tactual_cont"]
                        Tcontext = state["Tcontext_actual"]
                        Tdat_cont, dat_freerun = M.evaluate(net, ntest)
                        for nc in range(Kc):
                                print("actual transition matrix: {}".format(Tactual_cont[nc]))
                                print("empirical transition matrix: {}".format(Tdat_cont[nc]))
                        print("actual transition matrix (context): {}".format(Tcontext))
                        print("empirical transition matrix (context): {}".format(dat_freerun[2]))

                        ## dimensionality reduction of mixed layer acgtivty
                        # - PCA on mixed activity
                        from sklearn.decomposition import PCA
                        import numpy as np
                        mixed_activity = np.array([d[0] for d in dat_freerun[3]])
                        pca = PCA()
                        pca.fit(mixed_activity)
                        mixed_activity_reduced = pca.transform(mixed_activity)

                        # - plot in PCA space for each context and state
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.subplot(2,3,1)
                        plt.title('variance explained')
                        plt.plot(pca.explained_variance_ratio_, '-ok')
                        plt.xlabel('pc #')

                        plt.subplot(2,3,2)
                        for i in range(5):
                                plt.plot(pca.components_[i], '-o')

                        savename = modelpath + "/pcaresults.pdf"
                        plt.savefig(savename)
                        plt.close()




                        # ===== for each context.state pair plot position in pc space
                        v_state = np.array([d for d in dat_freerun[1]])
                        v_context = np.array([d for d in dat_freerun[0]])

                        context_list = set(v_context)
                        state_list = set(v_state)
                        dim2keep = 2
                        X = []
                        C = []
                        S = []
                        for c in context_list:
                                for s in state_list:
                                        X.append(np.array([mixed_activity_reduced[i][:dim2keep] for i in
                                                      range(mixed_activity_reduced.shape[0]) if (v_context[i]==c and v_state[i]==s)]))
                                        C.append(c)
                                        S.append(s)

                        # -- plot
                        plt.figure()
                        plt.xlabel('pc1')
                        plt.ylabel('pc2')
                        plt.title("mixed layer activations")

                        pcols = ['r', 'b']
                        for c, s, x in zip(C, S, X):
                                xmean = x.mean(axis=0)
                                xstd = x.std(axis=0)
                                # plt.plot(x[:,0], x[:,1], '.{}'.format(pcols[c]))
                                # plt.plot(x[:,0], x[:,1], '.', color=pcols[c])
                                plt.plot(xmean[0], xmean[1], "o{}".format(pcols[c]))
                                plt.text(xmean[0], xmean[1], "{}".format(s), color='k')
                        savename = modelpath + "/mixedactivations.pdf"
                        plt.savefig(savename)
                        plt.close()



                        ##
                        # ## RUN
                        # Tactual_cont, Tcontext = M.generateTrainDat(net, makedifferent)
                        # if docurriculum:
                        #         # --- do this for each context
                        #         # first train while clamping context. then unclamp and train on actual dynamics
                        #         # net, lossall = M.train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights,
                        #         #                        doBPTT, trainUsingContext, temp, doanneal,
                        #         #                        feedInputStates=feedStateInputs,
                        #         #                        hack=dohack)
                        #         ntrain_clamp = 5
                        #         nepochs_clamp = nepochs
                        #         net, lossall = Morig.train(net, nepochs, Tactual_cont, lr, ntrain_clamp, freezeWeights)
                        #         onlyModifyContext = True
                        # else:
                        #         onlyModifyContext = False
                        #
                        # if doFirstTrainWithContext:
                        #         net, lossall = M.train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights,
                        #                                doBPTT, True, temp, doanneal,
                        #                                learncontextslower=learncontextslower,
                        #                                feedInputStates=feedStateInputs, hack=dohack,
                        #                                onlyModifyContext=onlyModifyContext)
                        #
                        # net, lossall = M.train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights,
                        #                        doBPTT, trainUsingContext, temp, doanneal, learncontextslower=learncontextslower,
                        #                        feedInputStates=feedStateInputs, hack=dohack, onlyModifyContext=onlyModifyContext)
                        #
                        # Tdat_cont, dat_freerun = M.evaluate(net, ntest)
                        # for nc in range(Kc):
                        #         print("actual transition matrix: {}".format(Tactual_cont[nc]))
                        #         print("empirical transition matrix: {}".format(Tdat_cont[nc]))
                        # print("actual transition matrix (context): {}".format(Tcontext))
                        # print("empirical transition matrix (context): {}".format(dat_freerun[2]))
                        #
                        # M.plot(net, Tactual_cont, Tcontext, Tdat_cont, dat_freerun, lossall)
                        # print(net.savedir)
                        #
                        #
                        # # === save
                        # # save model
                        # # M.save(net)
                        #
                        # # ######### SAVE
                        # state = {
                        #         "Kc": Kc,
                        #         "Nc": Nc,
                        #         "Nhid": Nhid,
                        #         "Tactual_cont": Tactual_cont,
                        #         "Tdat_cont": Tdat_cont,
                        #         "K": K,
                        #         "N": N,
                        #         "ntrain": ntrain,
                        #         "nepochs": nepochs,
                        #         "lr": lr,
                        #         "ntest": ntest,
                        #         "state_dict": net.state_dict(),
                        #         "loss": lossall,
                        #         "Tcontext_actual": Tcontext,
                        #         "Tcontext_empirical": dat_freerun[2]}
                        #
                        # savename = net.savedir + "/state"
                        # torch.save(state, savename)
