import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    stream = io.BytesIO()

    torch.save(src.state_dict(), stream)

    stream.seek(0)

    state_dict = torch.load(stream)
    dst.load_state_dict(state_dict)

    return dst