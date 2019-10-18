## SCRIPT to train context models
import model_heirarchical_recurrent as M
from model_heirarchical_recurrent import ContextNet
import model as Morig
import torch

## PARAMS
# K_list = [2, 2, 2, 2]  # number of states
# N_list = [10, 10, 10, 10]  # size of states
# Nhid_list = [50, 50, 50, 50] # size of hidden layer
# Kc_list = [2, 2, 2, 2] # N contexts
# ntrain_list = [2, 5, 10, 20] # length of each sample sequence
# temp_list = [1, 1, 1, 1]
K_list = [3]  # number of states
N_list = [10]  # size of states
Nhid_list = [100] # size of hidden layer
Kc_list = [2] # N contexts
ntrain_list = [20] # length of each sample sequence
temp_list = [1]
nepochs = 10000 # training epochs
lr = 0.001 # learning rate
ntest = 2000 # testing, to calculate transition matrix

# savesuffix_list       = ["context", "contextfixfc1"] # can be empty
# freezeWeights_list = [False, True]
savesuffix_list = ["recurrent"] # can be empty
freezeWeights_list = [True]
nrepeats = 2
doBPTT = True
trainUsingContext = False # false means only observe states. True means observes context + state.
doanneal = True # if true, goes from temp to 1e-4. if False then uses just temp throughout.
makedifferent = True # default: False; if false, then random matrices. If true, then makes them different [only if 2 contexgts]
feedStateInputs = True # default: True; does nothing if training using context. if training from state only, then put this False to not reset input state on each step.
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
                        if True:
                            net = ContextNet(K, N, Kc, Nc, Nhid, savesuffix)
                        else:
                            model.load_state_dict(state['state_dict'])
                            optimizer.load_state_dict(state['optimizer'])


                        ## RUN
                        Tactual_cont, Tcontext = M.generateTrainDat(net, makedifferent)
                        if docurriculum:
                                # --- do this for each context
                                # first train while clamping context. then unclamp and train on actual dynamics
                                # net, lossall = M.train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights,
                                #                        doBPTT, trainUsingContext, temp, doanneal,
                                #                        feedInputStates=feedStateInputs,
                                #                        hack=dohack)
                                ntrain_clamp = 5
                                nepochs_clamp = nepochs
                                net, lossall = Morig.train(net, nepochs, Tactual_cont, lr, ntrain_clamp, freezeWeights)
                                onlyModifyContext = True
                        else:
                                onlyModifyContext = False

                        if doFirstTrainWithContext:
                                net, lossall = M.train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights,
                                                       doBPTT, True, temp, doanneal,
                                                       learncontextslower=learncontextslower,
                                                       feedInputStates=feedStateInputs, hack=dohack,
                                                       onlyModifyContext=onlyModifyContext)

                        net, lossall = M.train(net, nepochs, Tactual_cont, Tcontext, lr, ntrain, freezeWeights,
                                               doBPTT, trainUsingContext, temp, doanneal, learncontextslower=learncontextslower,
                                               feedInputStates=feedStateInputs, hack=dohack, onlyModifyContext=onlyModifyContext)

                        Tdat_cont, dat_freerun = M.evaluate(net, ntest)
                        for nc in range(Kc):
                                print("actual transition matrix: {}".format(Tactual_cont[nc]))
                                print("empirical transition matrix: {}".format(Tdat_cont[nc]))
                        print("actual transition matrix (context): {}".format(Tcontext))
                        print("empirical transition matrix (context): {}".format(dat_freerun[2]))

                        M.plot(net, Tactual_cont, Tcontext, Tdat_cont, dat_freerun, lossall)
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
                                "loss": lossall,
                                "Tcontext_actual": Tcontext,
                                "Tcontext_empirical": dat_freerun[2]}

                        savename = net.savedir + "/state"
                        torch.save(state, savename)
