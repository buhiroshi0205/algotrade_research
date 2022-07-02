import torch
from torch.nn import Module, Linear, GRU, Softmax, ReLU, Tanh, Dropout, BatchNorm1d, ModuleList, MSELoss, init

class DailyModel(Module):
    def __init__(self, dim_0, dim_1=256, dim_2=512, dropout=0.1, gru_layers=1) -> None:
        super(DailyModel, self).__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2

        self.linear = Linear(dim_0, dim_1)
        self.gru = GRU(dim_1, dim_2, batch_first=True, num_layers=gru_layers)
        self.linear_a = Linear(dim_2, dim_2)
        self.linear_out = Linear(dim_2, 1)

        self.dropout = Dropout(dropout)

        self.relu = ReLU()
        self.tanh = Tanh()
        self.softmax = Softmax(dim=1)

        self.bn_lin = BatchNorm1d(dim_1)
        self.bn_gru = BatchNorm1d(dim_2)
        self.bn_lin_a = BatchNorm1d(dim_2)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]

        # Linear + ReLU
        out = self.linear(x)
        out = self.bn_lin(out.view(-1, self.dim_1)).view(batch_size, -1, self.dim_1)
        out = self.dropout(out)
        out = self.relu(out)

        # GRU
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.bn_gru(out)

        a = self.linear_a(out)
        a = self.bn_lin_a(a)
        a = self.dropout(a)
        a = self.tanh(a)
        a = self.softmax(a)
        out = out * a
        
        # Linear + ReLU
        out = self.linear_out(out)
        return out.view(-1)
    
def weight_init(m) -> None:
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

def get_model(*args, **kwargs):
    model = DailyModel(*args, **kwargs)
    model.apply(weight_init)
    return model

def get_loss_fn():
    return MSELoss()