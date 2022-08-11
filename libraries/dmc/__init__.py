from .tasks import cheetah
from .tasks import walker
from .tasks import hopper
from .tasks import quadruped
from .tasks import jaco
from .tasks import point_mass_maze
from .dmc_tasks import DOMAINS, WALKER_TASKS, QUADRUPED_TASKS, JACO_TASKS, TASKS, PRIMAL_TASKS

def make(domain, task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    
    if domain == 'cheetah':
        return cheetah.make(task,
                            task_kwargs=task_kwargs,
                            environment_kwargs=environment_kwargs,
                            visualize_reward=visualize_reward)
    elif domain == 'walker':
        return walker.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'point_mass_maze':
        return point_mass_maze.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'hopper':
        return hopper.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'quadruped':
        return quadruped.make(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    else:
        raise f'{task} not found'

    assert None
    
    
def make_jaco(task, obs_type, seed):
    return jaco.make(task, obs_type, seed)