import torch

from rdsandwich.upper_bound import RDUBConfig, RDUBModel, UpperBoundTrainer
from rdsandwich.dataloader import GaussianSource, build_loader


def test_rdub_z_equals_y_forward():
    cfg = RDUBConfig(data_dim=4, latent_dim=4, decoder_units=[0], prior_type="std_gaussian", nats=True)
    model = RDUBModel(cfg)
    x = torch.randn(8, 4)
    loss, rate, mse = model.get_losses(x)
    assert loss.dim() == 0 and rate.item() >= 0 and mse.item() >= 0


def test_rdub_with_decoder_and_maf_prior():
    cfg = RDUBConfig(data_dim=4, latent_dim=3, decoder_units=[16], prior_type="maf", maf_stacks=1, nats=True)
    model = RDUBModel(cfg)
    x = torch.randn(8, 4)
    loss, rate, mse = model.get_losses(x)
    loss.backward()  # gradients should flow through the flow prior
    assert any(p.grad is not None for p in model.parameters())


def test_rdub_gmm_prior_loss_finite():
    cfg = RDUBConfig(data_dim=2, latent_dim=2, decoder_units=[0], prior_type="gmm_2", nats=True)
    model = RDUBModel(cfg)
    x = torch.randn(16, 2)
    loss, rate, mse = model.get_losses(x)
    assert torch.isfinite(loss)


def test_rdub_training_reduces_loss_on_gaussian():
    torch.manual_seed(0)
    cfg = RDUBConfig(data_dim=2, latent_dim=2, decoder_units=[0], prior_type="std_gaussian", nats=True, lmbda=10.0)
    model = RDUBModel(cfg)
    source = GaussianSource.standard(2)
    loader = build_loader(source, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = UpperBoundTrainer(
        model, loader, optimizer=optimizer, epochs=3, steps_per_epoch=20, verbose=False,
    )
    history = trainer.train()
    assert history["loss"][-1] <= history["loss"][0] + 1e-3
