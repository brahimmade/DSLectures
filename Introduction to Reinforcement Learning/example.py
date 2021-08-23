class Sarsa_Agent:
    def __init__(self, environment, n0, mlambda):
        self.n0 = float(n0)
        self.env = environment
        self.mlambda = mlambda
        
        # N(s) is the number of times that state s has been visited
        # N(s,a) is the number of times that action a has been selected from state s.
        self.N = np.zeros((self.env.dealer_values_count,
                           self.env.player_values_count, 
                           self.env.actions_count))
        
        self.Q = np.zeros((self.env.dealer_values_count,
                           self.env.player_values_count, 
                           self.env.actions_count))
        self.E = np.zeros((self.env.dealer_values_count, self.env.player_values_count, self.env.actions_count))

        # Initialise the value function to zero. 
        self.V = np.zeros((self.env.dealer_values_count, self.env.player_values_count))
        
        self.count_wins = 0
        self.iterations = 0

        # get optimal action, with epsilon exploration (epsilon dependent on number of visits to the state)
    # ε-greedy exploration strategy with εt = N0/(N0 + N(st)), 
    def train_get_action(self, state):
        dealer_idx = state.dealer-1
        player_idx = state.player-1
        try:
            n_visits = sum(self.N[dealer_idx, player_idx, :])
        except:
            n_visits = 0        

        # epsilon = N0/(N0 + N(st)
        curr_epsilon = self.n0 / (self.n0 + n_visits)

        # epsilon greedy policy
        if random.random() < curr_epsilon:
            r_action = Actions.hit if random.random()<0.5 else Actions.stick
#             if (dealer_idx == 0 and player_idx == 0):
#                 print ("epsilon:%s, random:%s " % (curr_epsilon, r_action))
            return r_action
        else:
            action = Actions.to_action(np.argmax(self.Q[dealer_idx, player_idx, :]))
#             if (dealer_idx == 0 and player_idx == 0):
#                 print ("epsilon:%s Qvals:%s Q:%s" % (curr_epsilon, self.Q[dealer_idx, player_idx, :], action))
            return action

    def get_action(self, state):
        action = Actions.to_action(np.argmax(self.Q[state.dealer_idx(), state.player_idx(), :]))
        return action

    def validate(self, iterations):        
        wins = 0; 
        # Loop episodes
        for episode in xrange(iterations):

            s = self.env.get_start_state()
            
            while not s.term:
                # execute action
                a = self.get_action(s)
                s, r = self.env.step(s, a)
            wins = wins+1 if r==1 else wins 

        win_percentage = float(wins)/iterations*100
        return win_percentage


                
                



    def train(self, iterations):        
        
        # Loop episodes
        for episode in xrange(iterations):
            self.E = np.zeros((self.env.dealer_values_count, self.env.player_values_count, self.env.actions_count))

            # get initial state for current episode
            s = self.env.get_start_state()
            a = self.train_get_action(s)
            a_next = a
            
            # Execute until game ends
            while not s.term:
                # update visits
                self.N[s.dealer_idx(), s.player_idx(), Actions.as_int(a)] += 1
                
                # execute action
                s_next, r = self.env.step(s, a)
                
                q = self.Q[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)]
                                
                if not s_next.term:
                    # choose next action with epsilon greedy policy
                    a_next = self.train_get_action(s_next)
                    
                    next_q = self.Q[s_next.dealer_idx(), s_next.player_idx(), Actions.as_int(a_next)]
                    delta = r + next_q - q
                else:
                    delta = r - q
                
#                 alpha = 1.0  / (self.N[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)])
#                 update = alpha * delta
#                 self.Q[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)] += update
                
                self.E[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)] += 1
                alpha = 1.0  / (self.N[s.dealer_idx(), s.player_idx(),  Actions.as_int(a)])
                update = alpha * delta * self.E
                self.Q += update
                self.E *= self.mlambda

                # reassign s and a
                s = s_next
                a = a_next

            #if episode%10000==0: print "Episode: %d, Reward: %d" %(episode, my_state.rew)
            self.count_wins = self.count_wins+1 if r==1 else self.count_wins

        self.iterations += iterations
#       print float(self.count_wins)/self.iterations*100

        # Derive value function
        for d in xrange(self.env.dealer_values_count):
            for p in xrange(self.env.player_values_count):
                self.V[d,p] = max(self.Q[d, p, :])
                
    def plot_frame(self, ax):
        def get_stat_val(x, y):
            return self.V[x, y]

        X = np.arange(0, self.env.dealer_values_count, 1)
        Y = np.arange(0, self.env.player_values_count, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_stat_val(X, Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        return surf