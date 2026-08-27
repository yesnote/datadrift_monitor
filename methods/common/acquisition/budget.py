'''Percentage-based annotation budgets shared by active methods.'''

from typing import Mapping


def resolve_percentage_budget(config: Mapping) -> int:
    '''Resolve one exact integer budget from the configured target pool.'''

    acquisition = config['acquisition']
    if 'total_budget' in acquisition:
        total_budget = int(acquisition['total_budget'])
        if total_budget < 0:
            raise ValueError('total_budget must not be negative')
        return total_budget
    target_size = int(config['dataset']['target']['expected_train_images'])
    percentage = float(acquisition['budget_percent'])
    if not 0.0 <= percentage <= 100.0:
        raise ValueError('budget_percent must be between 0 and 100')
    return int(target_size * percentage / 100.0 + 0.5)
