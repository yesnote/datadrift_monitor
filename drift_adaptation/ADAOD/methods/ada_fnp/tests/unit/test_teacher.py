import torch
from torch import nn

from methods.ada_fnp.training.teacher import initialize_teacher


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Linear(2, 2)
        self.discriminator = nn.Linear(2, 1)
        self.register_buffer('counter', torch.tensor([1.]))


def test_initialize_teacher_copies_detector_discriminator_and_buffers():
    student = Branch()
    teacher = Branch()
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(3.)
        student.counter.fill_(7.)
    initialize_teacher(student, teacher)
    for key, value in student.state_dict().items():
        assert torch.equal(value, teacher.state_dict()[key])
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert not teacher.training
