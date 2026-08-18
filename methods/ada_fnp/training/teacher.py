'''Teacher initialization at the 5k UDA-to-adaptation boundary.'''

from torch import nn


def initialize_teacher(student: nn.Module, teacher: nn.Module) -> None:
    '''Copy the entire student branch and freeze the teacher exactly once.'''

    student_keys = tuple(student.state_dict())
    teacher_keys = tuple(teacher.state_dict())
    if student_keys != teacher_keys:
        raise ValueError('student and teacher state structures do not match')
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
