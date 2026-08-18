import torch
from torch import nn

from methods.ada_fnp.models import DomainDiscriminator, FalseNegativePredictionModule
from methods.ada_fnp.models.fnpm import fnpm_loss


def test_domain_discriminator_preserves_spatial_shape():
    model = DomainDiscriminator()
    output = model(torch.randn(2, 512, 5, 7))
    assert output.shape == (2, 1, 5, 7)


def test_models_contain_no_batch_norm():
    modules = list(DomainDiscriminator().modules())
    modules += list(FalseNegativePredictionModule().modules())
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in modules)


def test_fnpm_output_is_non_negative_and_can_exceed_one():
    model = FalseNegativePredictionModule()
    with torch.no_grad():
        model.regressor[-2].weight.zero_()
        model.regressor[-2].bias.fill_(2.)
    output = model(torch.zeros(2, 512, 3, 3))
    assert torch.all(output > 1)


def test_fnpm_domain_means_are_added():
    result = fnpm_loss(
        torch.tensor([1., 3.]), torch.tensor([0., 1.]),
        torch.tensor([2.]), torch.tensor([0.]),
    )
    assert result.item() == 6.5
