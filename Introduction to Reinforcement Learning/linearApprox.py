import numpy as np
from environment import *
from tqdm import tqdm
import logging

#make feature a vector

class Linear_Approximation_Agent:
    def __init__(self, environment, n0, mlambda, gamma):
        self.n0 = float(n0)
        self.env = environment
        self.mlambda = mlambda
        self.gamma = gamma

        self.dealer_features = [[1, 4], [4, 7], [7, 10]]
        self.player_features = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]

        self.number_of_parameters = len(self.dealer_features) * len(self.player_features) * 2
        

        self.theta = np.random.rand(self.number_of_parameters) * 0.1

        self.phi = np.zeros(self.number_of_parameters)

        self.Q = np.zeros(self.number_of_parameters)

        self.E = np.zeros(self.number_of_parameters)

        # Initialise the value function to zero.
        self.V = np.zeros((self.env.dealer_values_count, self.env.player_values_count))

        self.count_wins = 0
        self.iterations = 0

    def compute_phi(self, s, a):
        
        phi = np.zeros((3, 6, 2), dtype=np.int)
        
        d_features = np.array([x[0] <= s.dealer <= x[1] for x in self.dealer_features])
        p_features = np.array([x[0] <= s.player <= x[1] for x in self.player_features])
        
        for i in np.where(d_features):
            for j in np.where(p_features):
                phi[i, j, a.value] = 1

        return phi.flatten()

    def get_optimal_action(self, state):
        action = Actions.to_action(
            np.argmax([np.dot(self.compute_phi(state, Actions.hit), self.theta), np.dot(self.compute_phi(state, Actions.stick), self.theta)])
        )
        return action

    # get optimal action, with epsilon exploration (epsilon dependent on number of visits to the state)

    def get_action(self, state):

        curr_epsilon = 0.05

        # epsilon greedy policy
        if random.random() > curr_epsilon:
            action = self.get_optimal_action(state)

            return action
        else:
            action = Actions.hit if random.random() < 0.5 else Actions.stick

            return action
    


    def train(self, iterations, disable_logging=False):
        # Loop episodes
        for episode in tqdm(range(iterations), disable=disable_logging):

            self.E = np.zeros(self.number_of_parameters)

            # Initialise state and action
            s = self.env.get_start_state()
            a = self.get_action(s)
            a_prime = a

            # Repeat for each step of episode
            while not s.terminal:

                # execute action
                s_prime, r = self.env.step(s, a)

                phi = self.compute_phi(s, a)

                q = np.dot(phi, self.theta)

                if not s_prime.terminal:
                    # choose next action with epsilon greedy policy
                    a_prime = self.get_action(s_prime)
                    q_next = np.dot(self.compute_phi(s_prime, a_prime),self.theta)
                    delta = r + self.gamma * q_next - q

                else:
                    delta = r - q

                self.E = self.E + phi

                alpha = 0.01
                
                self.theta = self.theta + alpha * delta * self.E
                self.E = self.gamma * self.mlambda * self.E

                s = s_prime
                a = a_prime

            self.iterations += 1
            if r == 1:
                self.wins += 1

        print(float(self.count_wins) / self.iterations * 100)

        # Derive value function
        for i in range(1, self.env.dealer_values_count + 1):
            for j in range(1, self.env.player_values_count + 1):
                s = self.env.get_state(i,j)
                self.V[i-1][j-1] = Actions.as_int(self.get_optimal_action(s))



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
