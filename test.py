# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

super(ContextNet, self).__init__()
# self.outdim = K
# self.indim = N
self.Wfix = Parameter(torch.rand((N, K)), requires_grad=False)  # this is fixed. (from y back to x)
# self.Wfix_context = Parameter(torch.rand((Nc, Kc)), requires_grad=False)
# self.hiddendim = Nhid

self.K = K  # states
self.N = N  # size of state
# self.Kc = Kc # contexts
# self.Nc = Nc # size of contexts`
self.Nhid = Nhid  # size of hidden layer.

# savdir
import os

tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%y-%H%M%S')
savedir = "saved_models/model_{}st_{}size_{}ctxt_{}hid_{}".format(K, N, Kc, Nhid, tstamp) + "_{}".format(savesuffix)
os.mkdir(savedir)
self.savedir = savedir

# ===== initialize layers
self.lstm = nn.LSTM(N, Nhid, 1)
self.fc1 = nn.Linear(Nhid, K, bias=True)  # state --> mixed
# self.fc2 = nn.Linear(Nhid, K, bias=True) # mixed --> state(onehot)
# self.fc3 = nn.Linear(Nhid, Kc, bias=True) # mixed --> context(onehot)
# self.fc4 = nn.Linear(N + Nc + Nhid, Nhid, bias=True) # context + state + mixed --> mixed

