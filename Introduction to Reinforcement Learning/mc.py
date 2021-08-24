import numpy as np
from environment import *
from tqdm import tqdm


class MC_Agent:
    def __init__(self, environment, n0):
        self.n0 = float(n0)
        self.env = environment

        # N(s) is the number of times that state s has been visited
        # N(s,a) is the number of times that action a has been selected from state s.
        self.N = np.zeros(
            (
                self.env.dealer_values_count,
                self.env.player_values_count,
                self.env.actions_count,
            )
        )

        self.Q = np.zeros(
            (
                self.env.dealer_values_count,
                self.env.player_values_count,
                self.env.actions_count,
            )
        )

        # Initialise the value function to zero.
        self.V = np.zeros((self.env.dealer_values_count, self.env.player_values_count))

        self.count_wins = 0
        self.iterations = 0

    # get optimal action, with epsilon exploration (epsilon dependent on number of visits to the state)
    # ε-greedy exploration strategy with εt = N0/(N0 + N(st)),
    def get_action(self, state):
        dealer_idx = state.dealer - 1
        player_idx = state.player - 1
        try:
            n_visits = sum(self.N[dealer_idx, player_idx, :])
        except:
            n_visits = 0

        # epsilon = N0/(N0 + N(st))
        curr_epsilon = self.n0 / (self.n0 + n_visits)

        # epsilon greedy policy
        if random.random() > curr_epsilon:
            action = Actions.to_action(np.argmax(self.Q[dealer_idx, player_idx, :]))

            return action
        else:
            action = Actions.hit if random.random() < 0.5 else Actions.stick

            return action
            

    def train(self, iterations, disable_logging = False):

        # Loop episodes
        for episode in tqdm(range(iterations), disable=disable_logging):
            episode_pairs = []

            # get initial state for current episode
            s = self.env.get_start_state()

            # Execute until game ends
            while not s.terminal:

                # get action with epsilon greedy policy
                a = self.get_action(s)

                # store action state pairs
                episode_pairs.append((s, a))

                # update visits
                # N(s) is the number of times that state s has been visited
                # N(s,a) is the number of times that action a has been selected from state s.
                self.N[s.dealer - 1, s.player - 1, Actions.as_int(a)] += 1

                # execute action
                s, r = self.env.step(s, a)

            self.count_wins = self.count_wins + 1 if r == 1 else self.count_wins

            # Update Action value function accordingly
            for curr_s, curr_a in episode_pairs:
                dealer_idx = curr_s.dealer - 1
                player_idx = curr_s.player - 1
                action_idx = Actions.as_int(curr_a)

                # Use a time-varying scalar step-size of αt = 1/N(st,at)
                step = 1.0 / self.N[dealer_idx, player_idx, action_idx]
                error = r - self.Q[dealer_idx, player_idx, action_idx]
                self.Q[dealer_idx, player_idx, action_idx] += step * error

        self.iterations += iterations

        # Win probability
        print(float(self.count_wins) / self.iterations * 100)

        # Derive value function
        for d in range(self.env.dealer_values_count):
            for p in range(self.env.player_values_count):
                self.V[d, p] = max(self.Q[d, p, :])

    def plot_frame(self, ax):
        def get_stat_val(x, y):
            return self.V[x, y]

        X = np.arange(0, self.env.dealer_values_count, 1)
        Y = np.arange(0, self.env.player_values_count, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_stat_val(X, Y)
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        return surf