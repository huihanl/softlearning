import roboverse
import numpy as np
import gym
from precog.predictive_env import PredictiveModel


class PredictiveModelEnvWrapper:

    def __init__(self, model_dir, num_execution_per_step, base_env=None, img_dim=48):
        ## base_env is assumed to be passed in, predictive_model is loaded here
        self.predictive_model = PredictiveModel(model_dir)
        self.base_env = base_env
        self.img_dim = img_dim

        self.num_execution_per_step = num_execution_per_step
        self.past_length = self.predictive_model.past_length
        self.state_dim = self.predictive_model.state_dim  ## should be smaller than 11
        self.past = np.zeros([self.past_length, self.state_dim])
        self._set_action_space()
        self.observation_space = base_env.observation_space

        # record z and trajs distribution
        from datetime import datetime
        now = datetime.now()
        curr_time = now.strftime("%H-%M-%S")
        self.file_z = open('file_z_{}.txt'.format(curr_time), 'w')
        self.file_action = open('file_action_{}.txt'.format(curr_time), 'w')
        self.file_obs = open('file_obs_{}.txt'.format(curr_time), 'w')

    def step(self, action):
        z = action
        self.file_z.write("{},{},{} \n".format(z[0], z[1], z[2]))
        obs = self.base_env.get_observation()
        obs = obs["image"].reshape([self.img_dim, self.img_dim, 3]) * 255
        # import pdb; pdb.set_trace()
        real_action = self.predictive_model.predict(self.past[-self.past_length:], obs, z)
        total_reward = 0
        for i in range(self.num_execution_per_step):
            first_predicted_action = real_action[0, 0, 0, i]
            if self.state_dim == 2:
                y, z = first_predicted_action[0], first_predicted_action[1]
                a = np.array([0, y, z, 0])
            elif self.state_dim == 3:
                x, y, z = first_predicted_action[0], first_predicted_action[1], first_predicted_action[2]
                a = np.array([x, y, z, 0])
            elif self.state_dim == 4:
                x, y, z, theta = first_predicted_action[0], first_predicted_action[1], first_predicted_action[2], first_predicted_action[3]
                a = np.array([x, y, z, theta])
            else:
                print("state_dim of ", self.state_dim, " unhandled")
                import pdb; pdb.set_trace()
            self.file_action.write("{},{},{} \n".format(first_predicted_action[0],first_predicted_action[1],first_predicted_action[2]))
            obs, reward, done, info = self.base_env.step(a)
            total_reward += reward
            # import pdb; pdb.set_trace()

        if self.state_dim == 2:
            state = np.array([obs["state"][1], obs["state"][2]]).reshape([1, self.state_dim])
        elif self.state_dim == 3:
            state = np.array([obs["state"][0], obs["state"][1], obs["state"][2]]).reshape([1, self.state_dim])
        elif self.state_dim == 4:
            state = np.array([obs["state"][0], obs["state"][1], obs["state"][2], 0]).reshape([1, self.state_dim])
        else:
            print("state_dim of ", self.state_dim, " unhandled")
            import pdb; pdb.set_trace()
        
        sl = [obs["state"][i] for i in range(3)]
        self.file_obs.write("{},{},{}\n".format(sl[0], sl[1], sl[2]))

        self.past = np.concatenate([self.past, state], axis=0)
        return obs, total_reward, done, info

    def _set_action_space(self):
        act_dim = self.predictive_model.z_dim  ## act_dim corresponds to the dim needed by the predictive model
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def reset(self):
        self.file_action.write("reset \n")
        self.file_obs.write("reset \n")
        self.past = np.zeros([self.past_length, self.state_dim])
        obs = self.base_env.reset()
        return obs

    def __getattr__(self, attr):
        return getattr(self.base_env, attr)