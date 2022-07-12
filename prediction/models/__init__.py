from . import default, deeper, attention

name_mapping = {
    'default': default,
    'deeper': deeper,
    'attention': attention
}

def get_model(name, *args, **kwargs):
    if name in name_mapping:
        return name_mapping[name].get_model(*args, **kwargs)
    else:
        raise ValueError(f'Model name {name} does not exist or is not yet implemented.')

def get_loss_fn(name):
    if name in name_mapping:
        return name_mapping[name].get_loss_fn()
    else:
        raise ValueError(f'Model name {name} does not exist or is not yet implemented.')