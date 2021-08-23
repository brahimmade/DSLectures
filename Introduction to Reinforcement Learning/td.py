import numpy as np
from environment import *
from tqdm import tqdm

class Sarsa_Agent:
    def __init__(self, environment, n0, mlambda, gamma):
        self.n0 = float(n0)
        self.env = environment
        self.mlambda = mlambda
        self.gamma = gamma

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
        #Eligibility Trace
        self.E = np.zeros(
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

    def get_optimal_action(self, state):
        action = Actions.to_action(np.argmax(self.Q[state.dealer_idx(), state.player_idx(), :]))
        return action

    # get optimal action, with epsilon exploration (epsilon dependent on number of visits to the state)
    # ε-greedy exploration strategy with εt = N0/(N0 + N(st)),
    def get_action(self, state):
        dealer_idx = state.dealer - 1
        player_idx = state.player - 1

        n_visits = sum(self.N[dealer_idx, player_idx, :])

        # epsilon = N0/(N0 + N(st))
        curr_epsilon = self.n0 / (self.n0 + n_visits)

        # epsilon greedy policy
        if random.random() > curr_epsilon:
            action = self.get_optimal_action(state)

            return action
        else:
            action = Actions.hit if random.random() < 0.5 else Actions.stick

            return action
            

    def train(self, iterations):

        # Loop episodes
        for episode in tqdm(range(iterations)):
            '''
            Repeat for each episode
            E(s,a) = 0
            Initialise S, A
            Repeat for each step of episode:
                Take action A, observe R, S'
                Choose A' from S' using policy derived from Q
                delta = R +lambdaQ(S',A') - Q(S,A)
                E(S,A) = E(S,A) + 1
                For all s in S , a in A(s)
                    Q(s,a) = Q(s,a) + alpha delta E(s,a)
                    E(s,a) = gamma lambda E(s,a)
                    S = S' , A = A'
            '''
            self.E = np.zeros(
            (
                self.env.dealer_values_count,
                self.env.player_values_count,
                self.env.actions_count,
            )
        )

            # Initialise state and action
            s = self.env.get_start_state()
            a = self.get_action(s)
            a_prime = a 

            # Repeat for each step of episode
            while not s.terminal:

                # update visits
                # N(s) is the number of times that state s has been visited
                # N(s,a) is the number of times that action a has been selected from state s.
                self.N[s.dealer - 1, s.player - 1, Actions.as_int(a)] += 1

                # execute action
                s_prime, r = self.env.step(s, a)

                q = self.Q[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)]

                if not s_prime.terminal:
                    # choose next action with epsilon greedy policy
                    a_prime = self.get_action(s_prime)
                    q_next = self.Q[s_prime.dealer_idx(), s_prime.player_idx(), Actions.as_int(a_prime)]
                    delta = r + self.gamma*q_next - q

                else:
                    delta = r - q
                
                self.E[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)] += 1

                alpha = 1.0  / (self.N[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)])
                self.Q = self.Q + alpha * delta * self.E
                self.E = self.gamma * self.mlambda * self.E 

                s = s_prime
                a = a_prime

            self.count_wins = self.count_wins+1 if r==1 else self.count_wins

        self.iterations += iterations
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
