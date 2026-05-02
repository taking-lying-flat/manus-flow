import math
import torch


def w1(z1):
    return torch.sin(2 * math.pi * z1 / 4)


def w2(z1):
    return 3 * torch.exp(-0.5 * ((z1 - 1) / 0.6) ** 2)


def w3(z1):
    return 3 * torch.sigmoid((z1 - 1) / 0.3)


def split_z(z):
    z1 = z[..., 0]
    z2 = z[..., 1]
    return z1, z2


def u1(z):
    z1, z2 = split_z(z)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)
    term1 = 0.5 * ((norm - 2) / 0.4) ** 2
    logit1 = -0.5 * ((z1 - 2) / 0.6) ** 2
    logit2 = -0.5 * ((z1 + 2) / 0.6) ** 2
    term2 = torch.logsumexp(torch.stack([logit1, logit2], dim=0), dim=0)
    return term1 - term2


def u2(z):
    z1, z2 = split_z(z)
    return 0.5 * ((z2 - w1(z1)) / 0.4) ** 2


def u3(z):
    z1, z2 = split_z(z)
    logit1 = -0.5 * ((z2 - w1(z1)) / 0.35) ** 2
    logit2 = -0.5 * ((z2 - w1(z1) + w2(z1)) / 0.35) ** 2
    return -torch.logsumexp(torch.stack([logit1, logit2], dim=0), dim=0)


def u4(z):
    z1, z2 = split_z(z)
    logit1 = -0.5 * ((z2 - w1(z1)) / 0.4) ** 2
    logit2 = -0.5 * ((z2 - w1(z1) + w3(z1)) / 0.35) ** 2
    return -torch.logsumexp(torch.stack([logit1, logit2], dim=0), dim=0)


ENERGY_FUNCTIONS = {
    "u1": u1,
    "u2": u2,
    "u3": u3,
    "u4": u4,
}


def log_density_from_energy(energy_fn):
    def log_density(z):
        return -energy_fn(z)

    return log_density


def density_from_energy(energy_fn):
    def density(z):
        return torch.exp(-energy_fn(z))

    return density


LOG_DENSITIES = {
    name: log_density_from_energy(energy_fn)
    for name, energy_fn in ENERGY_FUNCTIONS.items()
}


DENSITIES = {
    name: density_from_energy(energy_fn)
    for name, energy_fn in ENERGY_FUNCTIONS.items()
}
