import torch
import numpy as np

class RL_agent:
    def __init__(self, architecture, n_actions, gamma=0.9,expl_rate=0.1, explore=True,
                 epochs=5, batch_size=32, memory_size=10e6):
        
        self.architecture = architecture # list of [input_nodes,hidden_nodes,output_nodes]
        self.n_actions = n_actions # list of possible actions
        self.explore = explore
        self.expl_rate = expl_rate
        self.learning_rate = 0.01
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory_size = memory_size # number of datapoints to keep
        
        
        # Use GPU for training, if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = 'cpu'
        
        # Add inputs for each action
        architecture[0] += n_actions
        
        # Create model, Loss criterion and optimizer
        self.model = MLP(self.architecture)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        #self.model.to(self.device) # move model to GPU, if possible
        
        # Dataset
        self.game_states = np.array([], dtype=np.float32).reshape(0,architecture[0])
        self.returns = []

    def set_exploration_rate(self,expl_rate):
        self.expl_rate = expl_rate
    
    def training_mode(self):
        self.explore = True
    
    def evaluation_mode(self):
        self.explore = False
        
    def choose_action(self, game_state, possible_actions=[], return_payoffs=False):     
        """
        game_state: Numpy array of the game state, can also be multidimensional
        possible_actions: Indices of possible actions. If only the first and second entry 
        of the actions list are possible, one passes [0,1].
        """
        
        if len(possible_actions)==0:
            actions = range(self.n_actions) # list of indices for each action
        else:
            actions = possible_actions
            
        game_state = game_state.flatten()
            
        # Do we explore?
        if self.explore: # if exploration is enabled
            if np.random.rand() < self.expl_rate: # if we choose to explore
                choice = np.random.choice(possible_actions)
                state = self.create_game_state(game_state,choice)
                self.add_game_state(state)
                return choice
            
        # if not explore
        
        # create a game state for every possible action
        states = []         
        for i in actions:
            state = self.create_game_state(game_state,i)
            states.append(state)
            
        
        self.model.eval() # no training mode
        self.model.to('cpu') # put on cpu since the game also runs there
        
        # get expected payoffs
        expected_payoffs = []        
        for state in states:
            # Expected returns for each action
            expected_payoff = self.model.forward(state).detach().numpy().squeeze()
            expected_payoffs.append(expected_payoff)
        
        
        best = np.argmax(expected_payoffs) # index of best expected payoff
        best_move = actions[best] # index of best (possible) action
        
        # append respective game state (without payoff)
        if self.explore:
            self.add_game_state(states[best])
        
        if return_payoffs:
            return best_move, expected_payoffs
        return best_move
    
    def create_game_state(self,game_state,idx):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[idx] = 1
        state = np.concatenate((game_state,one_hot_action))
        return torch.FloatTensor(state).resize_((1,self.model.layer_sizes[0]))
    
    def add_game_state(self,state):
        self.game_states = np.concatenate((self.game_states,state.reshape(1,-1)))
            
    def assign_payoffs(self,payoffs): 
        """
        payoffs: A list of numerical payoffs achieved during each game
        """
        
        returns = []
            
        # sum up immediate and future payoffs
        for i in range(len(payoffs)):
            future_payoffs = payoffs[i:]
            gammas = self.gamma**(np.arange(len(future_payoffs)))
            returns.append(np.sum(np.multiply(gammas,future_payoffs)))
        
        self.returns = np.concatenate((self.returns,returns))
        
    def train(self):
        
        device = self.device
        
        # Curb dataset
        if len(self.game_states)>self.memory_size:
            self.game_states = self.game_states[-self.memory_size:]
            self.returns = self.returns[-self.memory_size:]
        
        # Convert data to pytorch tensors
        X = torch.FloatTensor(self.game_states)
        y = torch.FloatTensor(self.returns) 
                
        trainloader = self.create_trainloader(X,y)
         
        self.model.to(device)
        self.model.train() # put model into training mode
        
        for epoch in range(self.epochs):  # loop over the dataset multiple times    
            running_loss = 0.0
            
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, targets = data
                
                if len(targets)==1:
                    continue # torch cannot cope with batch of size 1
                
                inputs = inputs.to(device) # pass to GPU if active
                targets = targets.to(device)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
    
                # statistics
                running_loss += loss.item()
        
    def create_trainloader(self,X,y):
        trainset = Dataset(X, y, self.batch_size) # create training set
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return trainloader
#---------------------------------------------------------------------------------------  
        
class MLP(torch.nn.Module):
        def __init__(self, layer_sizes):
            super(MLP, self).__init__()
            
            self.epochs = 0 # store number of epochs trained
            self.train_loss = []
            self.test_loss = []
            self.layer_sizes = layer_sizes
            self.NN = torch.nn.Sequential()

            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.NN.add_module(name="BN{:d}".format(i), module=torch.nn.BatchNorm1d(in_size))
                self.NN.add_module(name="D{:d}".format(i), module=torch.nn.Dropout(p=0.1))
                self.NN.add_module(name="L{:d}".format(i), module=torch.nn.Linear(in_size, out_size))
                self.NN.add_module(name="A{:d}".format(i), module=torch.nn.PReLU())    
        
        def forward(self, x):
            x = self.NN(x)
            return x
        
#---------------------------------------------------------------------------------------
            
class Dataset(torch.utils.data.Dataset):
    'Dataset class for PyTorch'
    def __init__(self, features, labels, batch_size):
        'Initialization'
        self.labels = labels
        self.features = features        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index]
        y = self.labels[index]

        return X, y