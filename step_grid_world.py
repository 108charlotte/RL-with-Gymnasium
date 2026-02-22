import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt

# for animation in terminal
import time
from rich.live import Live
from rich.table import Table


class GridWorldEnv(gym.Env): 
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode = None, size = 5): 
        self.size = size

        self.render_mode = render_mode # human or None

        # locations for agent, enemy, and target (currently only supports one of each)
        self._agent_location = np.array([-1, -1])
        self._target_location = np.array([-1, -1])
        self._enemy_location = np.array([-1, -1])

        self.enemy_action = 0 # default to-be-updated when enemy moves
        self.target_collected = False # tells the agent when to stop pursuing carrot since it's already been collected

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,)), 
                "target": gym.spaces.Box(0, size - 1, shape=(2,)), 
                "enemy": gym.spaces.Box(0, size - 1, shape=(2,)), 
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]), # up
            1: np.array([-1, 0]), # left
            2: np.array([0, -1]), # down
            3: np.array([1, 0]), # right
        }

        # for terminal display
        self.live = None
        self.table = None

    def _get_obs(self): 
        return {"agent": self._agent_location, "target": self._target_location, "enemy": self._enemy_location}

    def reset(self, seed = None, options = None): 
        super().reset(seed=seed, options=options)
        self._agent_location = self.np_random.integers(0, self.size, size=2)

        # set to agent location and then move until not on agent
        self._target_location = self._agent_location
        self._enemy_location = self._agent_location
        
        self.target_collected = False

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2)
        
        while np.array_equal(self._enemy_location, self._agent_location) or np.array_equal(self._target_location, self._enemy_location):
            self._enemy_location = self.np_random.integers(0, self.size, size=2)
        
        observation = self._get_obs()
        return observation, {} # info empty since not used right now

    def step(self, action): 
        direction = self._action_to_direction[action]
        enemy_direction = self._action_to_direction[self.enemy_action]

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        self._enemy_location = np.clip(self._enemy_location + enemy_direction, 0, self.size - 1)

        terminated = False
        # this could be an issue since typically you would want negative to encourage efficiency
        reward = 0.01 # positive to encourage survival after reaching carrot (and dissuade suicide)

        if (not self.target_collected) and np.array_equal(self._agent_location, self._target_location): 
            reward = 1
            self._target_location = np.array([-1, -1]) # removes carrot
            self.target_collected = True
        
        if np.array_equal(self._agent_location, self._enemy_location): 
            reward = -1
            terminated = True # agent died

        truncated = False
        observation = self._get_obs()

        return observation, reward, terminated, truncated, {} # empty dict for info since not currently using

    def render(self):
        if self.render_mode == "human":
            self.table = Table()

            if not self.live: 
                self.live = Live(self.table, refresh_per_second = 4)
                self.live.start()

            self.live.update(self.table)
            self.table.add_column("grid world")

            for y in range(self.size - 1, -1, -1): 
                row_str = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row_str += "A "  # agent
                    elif np.array_equal([x, y], self._target_location):
                        row_str += "T "  # target
                    elif np.array_equal([x, y], self._enemy_location): 
                        row_str += "E " # enemy
                    else:
                        row_str += ". "  # Empty
                self.table.add_row(f"{row_str}")

            time.sleep(0.04)
    
    def close(self): 
        if self.live: 
            self.live.stop()
        super().close()

class BunnyAgent: 
    def __init__(
        self,
        env, # training environment
        learning_rate, # how quickly to update Q values
        initial_epsilon, # starting exploration rate
        epsilon_decay, # how much to reduce epsilon each episode
        final_epsilon, # minimum exploration rate
        discount_factor = 0.95, # how much to value future rewards (encourages immediate rewards over long-term ones)
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) # array of 0s corresponding to each action
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # track learning progress
        self.training_error = []
    
    def Q(self, state, action=None): 
        if action is None: 
            return self.q_values[state]
        return self.q_values[state][action]

    # for multiple q functions, this is what would be updated
    def get_action(self, obs): # randomly selects or uses q values, but becomes less random over time 
        if np.random.random() < self.epsilon: 
            return self.env.action_space.sample()
        else: 
            # estimates q values for current state and returns action to take (highest val)
            return int(np.argmax(self.Q(obs))) # would instead be q function to determine which q function to use
    
    def update(self, obs, action, reward, terminated, next_obs): 
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action] # difference between expected and what values should be

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference) # update q values
        self.training_error.append(temporal_difference) # essentially error in q values (temporal bc estimates across time)
    
    def decay_epsilon(self): 
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay) # bottoms off at minimum epsilon


class EnemyAgent: 
    def get_action(self, obs): 
        # format of obs: (agent_x, agent_y, target_x, target_y, enemy_x, enemy_y)
        agent_x, agent_y = obs[0:2]
        enemy_x, enemy_y = obs[-2:]
        
        x_dist = agent_x - enemy_x
        y_dist = agent_y - enemy_y

        if abs(x_dist) > abs(y_dist): # needs to move left/right (x dist greater than y)
            if np.sign(x_dist) < 0: # needs to move left
                return 1
            return 3 # needs to move right
        else: # needs to move up/down (y dist greater than x dist)
            if np.sign(y_dist) < 0: # needs to move up
                return 2
            return 0 # needs to move down

# utility function for obs and next_obs
def dict_values_to_hashable(dictionary): 
        return tuple(np.concatenate(list(dictionary.values())))

# training hyperparameters
learning_rate = 0.01
n_episodes = 5000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2) # reaches halfway through training, 1st half exploring and 2nd half using learned
final_epsilon = 0.1

# create environment and agent
gym.register(
    id="gymnasium_env/GridWorld",
    entry_point=GridWorldEnv,
)

env = gym.make("gymnasium_env/GridWorld", render_mode="human", max_episode_steps=500)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BunnyAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon, 
)

enemy = EnemyAgent()

# training loop
for episode in tqdm(range(n_episodes)): # tqdm for training progress visualization
    obs, info = env.reset() # info empty
    done = False

    while not done: 
        action = agent.get_action(dict_values_to_hashable(obs))
        enemy_action = enemy.get_action(dict_values_to_hashable(obs))
        env.unwrapped.enemy_action = enemy_action
        next_obs, reward, terminated, truncated, info = env.step(action) # info will be empty
        agent.update(dict_values_to_hashable(obs), action, reward, terminated, dict_values_to_hashable(next_obs))
        done = terminated or truncated
        obs = next_obs

        if episode % 1000 == 0: 
            env.render()

    agent.decay_epsilon() # agent becomes less random over time


# training visualization from blackjack example in gymnasium tutorial

def get_moving_avgs(arr, window, convolution_mode): 
    return np.convolve(
        np.array(arr).flatten(), 
        np.ones(window), 
        mode=convolution_mode
    ) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()


env.close()