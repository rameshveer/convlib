import torch


def set_seed(seed, cuda):
    SEED = 1
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)


def init_cuda(seed):
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    set_seed(seed, cuda)

    # set device
    device = torch.device("cuda" if cuda else "cpu")
    print("Device:", device)

    return cuda, device