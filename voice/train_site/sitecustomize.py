"""Loaded by Python at startup, only in RVC's train.py and the processes it starts, and only when voice_train.TrainRunner put
this folder on PYTHONPATH because /dev/shm is too small for train.py's data loader (voice_train.loader_plan).

train.py builds its loader with num_workers=4, persistent_workers=True and prefetch_factor=8, fixed in the script. Here every
DataLoader asking for more workers than VOICE_TRAIN_LOADER_WORKERS gets that many instead; with 0 the two options that need
workers are dropped too, as torch requires. Nothing else about training changes: the same batches in the same order of work."""
import os

_WORKERS = os.environ.get("VOICE_TRAIN_LOADER_WORKERS")
if _WORKERS is not None:
    import torch.utils.data as _data

    _init = _data.DataLoader.__init__

    def _limited(self, *args, **kwargs):
        limit = int(_WORKERS)
        if kwargs.get("num_workers", 0) > limit:
            kwargs["num_workers"] = limit
            if limit == 0:
                kwargs["persistent_workers"] = False
                kwargs.pop("prefetch_factor", None)
        _init(self, *args, **kwargs)

    _data.DataLoader.__init__ = _limited
