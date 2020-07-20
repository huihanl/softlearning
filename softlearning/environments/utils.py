from .adapters.gym_adapter import GymAdapter
#from .adapters.predictive_wrapper_env import PredictiveModelEnvWrapper

ADAPTERS = {
    'gym': GymAdapter,
}
"""
try:
    from .adapters.dm_control_adapter import DmControlAdapter
    ADAPTERS['dm_control'] = DmControlAdapter
except ModuleNotFoundError as e:
    if 'dm_control' not in e.msg:
        raise

    print("Warning: dm_control package not found. Run"
          " `pip install git+https://github.com/deepmind/dm_control.git`"
          " to use dm_control environments.")

try:
    from .adapters.robosuite_adapter import RobosuiteAdapter
    ADAPTERS['robosuite'] = RobosuiteAdapter
except ModuleNotFoundError as e:
    if 'robosuite' not in e.msg:
        raise

    print("Warning: robosuite package not found. Run `pip install robosuite`"
          " to use robosuite environments.")
"""
UNIVERSES = set(ADAPTERS.keys())


def get_environment(universe, domain, task, environment_params):
    return ADAPTERS[universe](domain, task, **environment_params)


def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    return get_environment(universe, domain, task, environment_kwargs)

def get_roboverse_env_from_params(environment_params):
    import roboverse

    use_predictive_model = environment_params['use_predictive_model']

    env_name = environment_params['env']
    randomize = environment_params['randomize_env']
    observation_mode = environment_params['obs']
    reward_type = environment_params['reward_type']
    single_obj_reward = environment_params['single_obj_reward']
    all_random = environment_params['all_random']
    trimodal_positions_choice = environment_params['trimodal_positions_choice']
    num_objects = environment_params['num_objects']

    base_env = roboverse.make(
        env_name, gui=False, randomize=randomize,
        observation_mode=observation_mode, reward_type=reward_type,
        single_obj_reward=single_obj_reward,
        normalize_and_flatten=False,
        num_objects=num_objects,
        all_random=all_random,
        trimodal_positions_choice=trimodal_positions_choice)

    if use_predictive_model:
        model_dir = environment_params['model_dir']
        num_execution_per_step = environment_params['num_execution_per_step']
        img_width = base_env.obs_img_dim
        env = PredictiveModelEnvWrapper(model_dir, num_execution_per_step, base_env=base_env, img_dim=img_width)
        return GymAdapter(domain=None, task=None, env=env)
    else:
        return GymAdapter(domain=None, task=None, env=base_env)
