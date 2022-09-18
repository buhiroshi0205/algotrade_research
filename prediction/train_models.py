import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models


# The normal training loop you see with any other supervised training in pytorch (with some parameters).
def train_loop(train_loader,
               val_loader,
               model,
               loss_function,
               optimizer,
               lrscheduler,
               device,
               num_epochs,
               eval_epochs=1,
               verbose=1):
    
    model.to(device)

    for e in range(num_epochs):
        model.train()
        running_loss = 0
        count = 0
        for X, y, _ in tqdm(train_loader, desc="Training", leave=False) if verbose else train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            y_pred = model(X, y=y, Gumbel_noise=False, mode='train').squeeze()
            single_loss = loss_function(y_pred, y)
            if not torch.isnan(single_loss) and not torch.isinf(single_loss):
                single_loss.backward()
                optimizer.step()
                running_loss += single_loss
                count += 1
            else:
                print('nan or inf invalid loss encountered, skipping backward step')

        lrscheduler.step()
        
        if eval_epochs > 0 and (e+1) % eval_epochs == 0:
            model.eval()
            running_val_loss = 0
            val_count = 0
            for X, y, _ in tqdm(val_loader, desc="Evaluating", leave=False) if verbose else val_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X, y=y, Gumbel_noise=False, mode='val').squeeze()
                single_loss = loss_function(y_pred, y)
                running_val_loss += single_loss
                val_count += 1
            if verbose: print("Epoch {:2d}: Loss = {:4.2f}, Val loss = {:4.2f}".format(e + 1, running_loss / count, running_val_loss / val_count))
            
        else:
            if verbose: print("Epoch {:2d}: Loss = {:4.2f}".format(e + 1, running_loss / count))
    
    model.cpu()
    
                
"""
Function to run an actual train loop based on the given configuration (as string).

:param train_ds: training dataset (dataset.py/DailyDataset)
:param val_ds: validation dataset (dataset.py/DailyDataset)
:param model_name: which model (in directory models/) to use. Must have name registered in 'models/__init__.py'
:param model_params: kwarg dictionary of parameters to pass to the constructor of the above model
:param optimizer_name: optimizer to use. Must be one of the ones listed below, but maybe it can be changed to something more general?
:param optimizer_params: kwarg dictionary of parameters to pass to the constructor of the optimizer
:param lr: learning rate
:param gamma: multiplicative constant for decaying the learning rate each epoch. After each epoch, this constant is multiplied to the learning rate.
              Care should be taken with this value because it was weirdly tuned and has led to some weird results. The default probably doesn't work for
              most models and purposes. Previously named 'lr_decay' where lr_decay = 1 - gamma.
:param epochs: Number of epochs to train for
:param eval_epochs: Interval of epochs before each evaluation of the model using the validation dataset. 1 is generally ok, unless you want SPEEEEEED
                    and don't care about intermediate evaluation results
:param batch_size: batch size
:param verbose: 1 to print train/eval loss info while training. 0 is recommended during multiprocessing to not interleave outputs and not mess with progress bars.
:param device_name: The torch.device(device_name) to use. Preferably manually specified. By default uses 'cuda' if available and 'cpu' if not.
:param **kwargs: any kwargs to pass to the train_loop() function (should be nothing unless I messed up)
"""
def train_with_config(train_ds, val_ds,
                      model_name, model_params={},
                      optimizer_name='AdamW', optimizer_params={},
                      lr=0.05, gamma=0.3,
                      epochs=5, eval_epochs=1, batch_size=64,
                      verbose=1, device_name=None, **kwargs):
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)
    
    model = models.get_model(model_name, **model_params)
    loss_function = models.get_loss_fn(model_name)
    
    optimizer_name_map = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD
    }
    optimizer = optimizer_name_map[optimizer_name.lower()](model.parameters(), lr=lr, **optimizer_params)
    lrscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    if device_name is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    train_loop(train_loader, val_loader, model, loss_function, optimizer, lrscheduler, device, epochs, eval_epochs, verbose, **kwargs)
    
    return model

