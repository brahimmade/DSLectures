class LinearFunctionSarsaAgent(Agent):
    
    def __init__(self, environment, No=100, discount_factor=1, _lambda=1):    
        Agent.__init__(self, environment, No, discount_factor)
        
        # we can tune these parameters
        self._lambda = _lambda
        self.number_of_parameters = 36
        
        # parameters are initialize randomly
        self.theta = np.random.randn(self.number_of_parameters) * 0.1
        
        # eligibility trace
        self.E = self.get_clear_tensor()
  
        # features
        self.dealer_features = [[1, 4], [4, 7], [7, 10]]
        self.agent_features = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
    
    def get_clear_tensor(self):
        """Returns a tensor with zeros with the correct shape."""
        return np.zeros(self.number_of_parameters)

    def get_q(self, s, a):
        return np.dot(self.phi(s, a), self.theta)
    
    def get_all_q(self):
        q = np.zeros((self.env.dealer_max_value + 1,
                         self.env.agent_max_value + 1, 
                         self.env.actions_count))
        
        for i in range(1, self.env.dealer_max_value + 1):
            for j in range(1, self.env.agent_max_value + 1):
                for a in [Action.HIT, Action.STICK]:
                    s = State(i, j)
                    q[i, j, a.value] = self.get_q(s, a)
    
        return q

    def phi(self, s, a):
        d_sum = s.dealer_sum
        a_sum = s.agent_sum
        
        features = np.zeros((3, 6, 2), dtype=np.int)
        
        d_features = np.array([x[0] <= d_sum <= x[1] for x in self.dealer_features])
        a_features = np.array([x[0] <= a_sum <= x[1] for x in self.agent_features])
        
        for i in np.where(d_features):
            for j in np.where(a_features):
                features[i, j, a.value] = 1

        return features.flatten()
   
    def get_alpha(self, s, a):
        return 0.01
       
    def get_e(self, s):
        return 0.05

    def try_all_actions(self, s):
        return [np.dot(self.phi(s, Action.STICK), self.theta), np.dot(self.phi(s, Action.HIT), self.theta)]
    
    def get_max_action(self, s):
        return np.max(self.try_all_actions(s))
    
    def choose_best_action(self, s):
        return Action.HIT if np.argmax(self.try_all_actions(s)) == 1 else Action.STICK
    
    def policy(self, s):
        r = random.random()
        if r <= self.get_e(s):
            action = self.choose_random_action()
        else:
            action = self.choose_best_action(s)

        return action

    def train(self, steps):
        for e in range(steps):
            # clear eligibility trace
            self.E = self.get_clear_tensor()
            
            # get initial state for current episode
            s = self.env.initial_state()

            # choose a from s with epsilon greedy policy
            a = self.policy(s)
            next_a = a 
                 
            # while game has not ended
            while not s.is_terminal:
                
                # execute action
                next_s, r = self.env.step(copy.copy(s), a)
              
                # get parameters that represent this state and action
                phi = self.phi(s, a)
                # get q(s, a)
                q = self.get_q(s, a)
                
                if not next_s.is_terminal:
                    # choose next action with epsilon greedy policy
                    next_a = self.policy(next_s)
                    q_next = self.get_q(next_s, next_a)
                    delta = r + q_next - q
                else:
                    delta = r - q 
            
                self.E += phi
                alpha = self.get_alpha(s, a)
                update_q = alpha * delta * self.E
                self.theta += update_q
                self.E *= (self.disc_factor * self._lambda)
                
                # update state and action
                s = next_s
                a = next_a

            if e % 10000 == 0 and e != 0:
                print("Episode: %d, score: %f" % (e, (float(self.wins)/self.iterations)*100))
            
            # update wins and iterations
            self.iterations += 1
            if r == 1:
                self.wins += 1

        return self.get_value_function()