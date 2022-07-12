import torch
from torch import nn
from torch.nn import Module, Linear, GRU, Softmax, ReLU, Tanh, Dropout, BatchNorm1d, ModuleList, MSELoss, init
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

# Model Reconstructed:
# 
# Idea--> LAS model:
# The architectures I used is a LAS(attention) model includes two main modules: encoder and decoder. The encoder is based on 1 layer of bidirectional lstm(bottom) followed by 3 layers of pyramidal lstm and followed by value network and key network which are used for the attention mechanism, combined with the query obtained in the decoder. The decoder consists of two lstm cells and use the attention mechanism to output context and attentions.

# In[6]:


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        # the base/bottom lstm layer
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)


    def forward(self, x):
        # Pad input if it is packed
        x_unpacked, x_lens = pad_packed_sequence(x, batch_first=True)
        # x_unpacked: shape[B, T, dim]
        # Truncate the input length dimension by concatenating feature dimension
        # x_lens = x_lens.cuda()
        # chop off last element if input is odd
        chopped_length = torch.div(x_unpacked.shape[1],2, rounding_mode='floor')*2
        x_unpacked = x_unpacked[:,:chopped_length,:]
        # deal with input length array (x_lens) after truncating the input?
        # reshape to (B, T/2, dim*2)
        new_seq_len = torch.div(x_unpacked.shape[1],2, rounding_mode='floor')
        x_reshaped = x_unpacked.reshape(x_unpacked.shape[0], new_seq_len, x_unpacked.shape[2] * 2)
        x_lens = torch.div(x_lens, 2, rounding_mode='floor')

        # need to repack the seqences to serve as input to lstm layer(similar to HW3 part2)
        x_packed = pack_padded_sequence(x_reshaped, lengths=x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # pass into the lstm layer
        out, hiddens = self.blstm(x_packed)
        return out


# In[52]:


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        
        # First module: use 2 layers of mlp
        

        
        self.linear_in = nn.Sequential(nn.Linear(input_dim, encoder_hidden_dim//2),
                                       #nn.BatchNorm1d(encoder_hidden_dim//2, eps=1e-05, momentum=0.1),
                                       nn.Dropout(0.2),
                                       nn.LeakyReLU(),
                                       
                                       nn.Linear(encoder_hidden_dim//2, encoder_hidden_dim),
                                       #nn.BatchNorm1d(encoder_hidden_dim, eps=1e-05, momentum=0.1),
                                       nn.Dropout(0.2),
                                       nn.LeakyReLU())
        
        
        
        
        # 2 layers of bidirectional LSTM
        self.lstm = nn.LSTM(input_size=encoder_hidden_dim, hidden_size= encoder_hidden_dim*2, num_layers=2, bidirectional=True, batch_first=True)
                                 
        

        # Define the blocks of pBLSTMs
        # Dimensions should be chosen carefully
        # Hint: Bidirectionality, truncation-> x 4
        # Optional dropout
        """
        Design: 1 bottom BLSTM, 3 pBLSTMs on top, Reducing input length by factor of 8
        """
        self.pBLSTMs = nn.Sequential(pBLSTM(encoder_hidden_dim*4, encoder_hidden_dim),
            pBLSTM(encoder_hidden_dim*4, encoder_hidden_dim),
            pBLSTM(encoder_hidden_dim*4, encoder_hidden_dim),
        )
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? --> x 2
        self.key_network = nn.Linear(encoder_hidden_dim*4, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_dim*4, key_value_size)

    def forward(self, x):
        """
        Key and value: 
            Linear projections of the output from the last pBLSTM network.
        """
        B = x.shape[0]
        out = self.linear_in(x)
        #length = torch.zeros(B,dtype=torch.int64).fill_(30).cuda()
        #x_packed = pack_padded_sequence(out, lengths= length.cpu(), batch_first=True, enforce_sorted=False)
        out,_ = self.lstm(out)

        # Pass it through the pyramidal LSTM layer
        #out = self.pBLSTMs(out)

        # Pad your input back to (B, T, *) shape
        #linear_in, truncated_lens = pad_packed_sequence(out, batch_first=True)
        keys = self.key_network(out)
        value = self.value_network(out)
        # Output Key, Value, and truncated input lens(1/8 of original input length )
        return keys, value


# In[53]:


def plot_attention(attention):
    # utility function for debugging
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

class Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    I first use single head attention here
    
    Optional: Multi-Head Attention
        # Check "attention is all you need" Section 3.2.2
        h = Number of heads
        W_Q, W_K, W_V: Weight matrix for Q, K, V (h of them in total)
        W_O: d_v -> d_v

        Reshape K: (B, T, d_k)
        to (B, T, h, d_k // h) and transpose to (B, h, T, d_k // h)
        Reshape V: (B, T, d_v)
        to (B, T, h, d_v // h) and transpose to (B, h, T, d_v // h)
        Reshape Q: (B, d_q)
        to (B, h, d_q // h)

        energy = Q @ K^T
        energy = mask(energy)
        attention = softmax(energy)
        multi_head = attention @ V
        multi_head = multi_head reshaped to (B, d_v)
        context = multi_head @ W_O
    '''
    def __init__(self):
        super(Attention, self).__init__()
        # Optional: dropout


    def forward(self, query, key, value):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q) --> output from decoder, one for each sequence
        * Hint: d_k == d_v == d_q is often true if you use linear projections
        return:
            context: (batch_size, key_val_dim)
        
        """
        # from documentation: torch.bmm(input, mat2, *, out=None) → Tensor
        # input and mat2 must be 3-D tensors each containing the same number of matrices.
        # so we need to first "unsqueeze query to 3D"
        query_unsqueezed = torch.unsqueeze(query,2)
        # batch, seq_len, d_k) * (batch, d_q, 1) = (batch, seq, 1) since d_k == d_v == d_q
        # we squeezed (N, seq_len)
        energy_unsqueezed = torch.bmm(key, query_unsqueezed) #(batch, seq, 1)
        energy = torch.squeeze(energy_unsqueezed, 2) # (batch, seq)

        #mask = mask.cuda()
        #energy.masked_fill_(mask, -float("inf")) # need use a boolean mask
        # scaled dot product for attention--optional
        energy = energy/np.sqrt(key.shape[2])
        # compute attention and context
        attention = nn.functional.softmax(energy, dim=1) # (batch, seq)
        # to bmm, we need to unsqueeze attention to 3D
        context = torch.bmm(attention.unsqueeze(1), value) #--> (batch, 1,seq_len) * (batch_size, seq_len, d_v) = (batch, 1, d_v)
        # squeeze out to (batch, d_v)
        context = torch.squeeze(context,1)
        # return context,attention
        return context,attention


# In[54]:


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
"""
decoder architecture --->

decoder input y: None or (B, look_backward//5)

context--> 2 lstm celss -> query-->attention()-->linear out--> xgboost regressor-->produce fine tunning

"""
class Decoder(nn.Module):
    
    def __init__(self, decoder_hidden_dim, key_value_size=128):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTMCell(input_size = 1 + key_value_size, hidden_size = decoder_hidden_dim) 
        self.lstm2 = nn.LSTMCell(input_size = decoder_hidden_dim, hidden_size = key_value_size)
        self.attention = Attention()     
      
        self.linear_out = nn.Linear(2*key_value_size, 1) #: d_v -> 1 (regression)
        #self.xgb = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
        #        max_depth = 5, alpha = 10, n_estimators = 10)

        self.key_value_size = key_value_size

    def forward(self, key, value,  y=None, teacher_forcing_rate = 0.95, Gumbel_noise=False, mode = "train"):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, T//5) - Batch days of price
            mode: Train or eval mode for teacher forcing
            T --> lookback
        Return:
            predictions: predicted adjusted price
        '''
        

        B, key_seq_max_len, key_value_size = key.shape

        predictions = []
        # This is the first input to the decoder
        """if mode == "train":
          #prediction = torch.full((B,1), fill_value= y[0], device=device)
          start = torch.zeros((B,1),dtype=torch.long).fill_(y[0])
          #start = torch.unsqueeze(y,1)
          #start = torch.LongTensor(start)
        else:
          #prediction = torch.full((B,1), fill_value= 0, device=device)
          start = torch.zeros((B,1),dtype=torch.long).fill_(0)

        # The length of hidden_states vector should depend on the number of LSTM Cells defined in init
    
        hidden_states = [None, None] 
        
        # Initialize the context--》 what should I put here
        context = value[:, -1, :]"""

        if mode == 'train':
            max_len =  y.shape[1]
        else:
            max_len = 3
        predictions = []
        # This is the first input to the decoder
        # What should the fill_value be?
        prediction = torch.full((B,1), fill_value= 0, device=key.device)
        # The length of hidden_states vector should depend on the number of LSTM Cells defined in init
        # The paper uses 2
        hidden_states = [None, None] 
        
        # TODO: Initialize the context
        context = value[:, 0, :]

        attention_plot = [] # this is for debugging
        for i in range(max_len): # max_len = 6-->predcition sequence's length
            if mode == 'train':
                # TODO: only train, Implement Teacher Forcing
                using_teacher_forcing = np.random.random() < teacher_forcing_rate
                if using_teacher_forcing:
                    if i == 0:
                        start = torch.zeros((B,1)).fill_(y[0][0])
                        step = start.to(key.device) #move to gpu
                    else:
                        # Otherwise, feed the label of the **previous** time step
                        step = y[:, i-1].reshape((B,1))
                
                else:
                    if i == 0:
                        # This is the first time step
                        # Hint: How did you initialize "prediction" variable above?--> fill with index of sos(start) for all batches
                        start = torch.zeros((B,1)).fill_(0)
                        step = start.to(key.device) #move to gpu
                        
                    else:
                        previous_pred = prediction
                        step = previous_pred.reshape((B,1)) # using previous prediction as next step
                    
            else:
                # if the start
                if i == 0:
                    start = torch.zeros((B,1)).fill_(0)
                    step = start.to(key.device) #move to gpu 

                else:
                    previous_pred = prediction
                    step = previous_pred.reshape((B,1)) # using previous prediction as next step
            """
            print("step is in shape: ", step.shape)
            print("context is in shape: ", context.shape)
            print("prediction is in shape: ", prediction.shape)"""
        
            y_context = torch.cat([step, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            out_prev_layer = hidden_states[0][0]
            hidden_states[1] = self.lstm2(out_prev_layer, hidden_states[1])

            # What then is the query?
            query = hidden_states[1][0]
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention[0].detach().cpu())
            # What should be concatenated as the output context?
            output_context = torch.cat([query,context], dim=1)
            prediction = self.linear_out(output_context) 
            #print("at each time step, prediction shape is:", prediction.shape)
            # store predictions
            prediction_unsqueezed = torch.unsqueeze(prediction,1) # this is for concatenate for each batch
        
            predictions.append(prediction_unsqueezed)
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        #plot_attention(attentions)
        predictions = torch.cat(predictions, dim=1)
        # after cat--> in shape [batch, seq_len, 1]
        # index is for extracting the last timestamep price(our true prediction)
        # predictions in shape: [batch, seq_len]--> after squeeze
        return predictions.squeeze(), attentions


# In[55]:


class DailyModel(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, encoder_hidden_dim, decoder_hidden_dim, key_value_size=128):
        super(DailyModel,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size)
        self.decoder = Decoder(decoder_hidden_dim,  key_value_size)

    def forward(self, x, y=None, teacher_forcing_rate = 0.95, Gumbel_noise=False, mode='train'):
        key, value = self.encoder(x)
      
        predictions, attentions = self.decoder(key, value, y=y, teacher_forcing_rate = teacher_forcing_rate, Gumbel_noise=False, mode=mode)
        return predictions



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