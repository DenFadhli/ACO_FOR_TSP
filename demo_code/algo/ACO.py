import numpy as np

class ACO_TSP:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_dim  # number of cities
        self.size_pop = size_pop  # number of ants
        self.max_iter = max_iter  # number of iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # The importance of fitness
        self.rho = rho  # Pheromone volatilization rate

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  # Avoid division by zero errors

        self.Tau = np.ones((n_dim, n_dim))  # Pheromone matrix, updated every iteration
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)  # The crawling path of each ant in a certain generation
        self.y = None  # The total crawling distance of each ant in a certain generation
        self.generation_best_X, self.generation_best_Y = [], []  # Documenting the best of the generations
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y  # Historical reasons, in order to maintain unity
        self.best_x, self.best_y = None, None

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):  # for each iteration
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  # Transition probability, without normalization.
            for j in range(self.size_pop):  # for each ant
                self.Table[j, 0] = 0  # start point，it can actually be random, but it makes no difference
                for k in range(self.n_dim - 1):  # each node the ants reach
                    taboo_set = set(self.Table[j, :k + 1])  # The point that has been passed and the current point cannot be passed again
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # choose among these points
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # probability normalization
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # Calculate distance
            y = np.array([self.func(i) for i in self.Table])

            # By the way, record the best situation in history
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # Calculate the pheromone that needs to be freshly applied
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # every ant
                for k in range(self.n_dim - 1):  # each node
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # Ants climb from node n1 to node n2
                    delta_tau[n1, n2] += 1 / y[j]  # smeared pheromone
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # Ants crawl from the last node back to the first node
                delta_tau[n1, n2] += 1 / y[j]  # smear pheromone

            # Pheromone drift + pheromone smear
            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    fit = run
