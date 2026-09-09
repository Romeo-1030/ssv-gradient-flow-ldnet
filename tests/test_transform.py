import torch


def test_ssv_chain_rule():
    x = torch.tensor([0.3], dtype=torch.float64)
    t = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    alpha, beta = 0.37, 0.42
    s = 1 + t
    xi = x / s**alpha
    tau = torch.log(s)
    w = torch.sin(xi) * torch.exp(-0.2 * tau)
    u = w / s**beta
    direct = torch.autograd.grad(u, t)[0]
    predicted = s ** (-beta - 1) * (
        -0.2 * w - alpha * xi * torch.cos(xi) * torch.exp(-0.2 * tau) - beta * w
    )
    torch.testing.assert_close(direct, predicted)
