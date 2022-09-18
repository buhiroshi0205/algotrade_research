Each model is specified in its own file, and must implement `get_model(*args, **kwargs)` and `get_loss_fn()` functions. They should be properly imported and referenced in `__init__.py`.

- `attention.py`: An attention-based model by Shaofeng referenced in the slides.
- `default.py`: A GRU model with one layer linear then one layer GRU.
- `deeper.py`: Same GRU model with two layers linear then one layer GRU.