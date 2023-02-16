import numpy as np
import pygad

class TrackCalc:

    def __init__(self,
                 start_pos,
                 goal_pos,
                 airspeed,
                 rod=1):

        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.airspeed = airspeed
        self.rod = rod
        self.aircraft_states = ('aircraft-north', 'aircraft-east', 'aircraft-south', 'aircraft-west')
        self.movement = {'aircraft-north': (0, self.airspeed),
                         'aircraft-east': (self.airspeed, 0),
                         'aircraft-south': (0, -self.airspeed),
                         'aircraft-west': (-self.airspeed, 0)}
        self.aircraft_actions = {'aircraft-north': ('aircraft-west', 'aircraft-east'),
                                 'aircraft-east': ('aircraft-north', 'aircraft-south'),
                                 'aircraft-south': ('aircraft-east', 'aircraft-west'),
                                 'aircraft-west': ('aircraft-south', 'aircraft-north')}
        
        # self.num_generations = ga_params["num_generations"]
        # self.num_parents_mating = ga_params["num_parents_mating"]
        # self.sol_per_pop = ga_params["sol_per_pop"]
        # self.num_genes = ga_params["num_genes"]
        # self.init_range_low = ga_params["init_range_low"]
        # self.init_range_high = ga_params["init_range_high"]
        # self.parent_selection_type = ga_params["parent_selection_type"]
        # self.keep_parents = ga_params["keep_parents"]
        # self.crossover_type = ga_params["crossover_type"]
        # self.mutation_type = ga_params["mutation_type"]
        # self.mutation_percent_genes = ga_params["mutation_percent_genes"]

        # self.ga_instance = pygad.GA(num_generations=self.num_generations,
        #                             num_parents_mating=self.num_parents_mating,
        #                             fitness_func=self.fitness_func,
        #                             sol_per_pop=self.sol_per_pop,
        #                             num_genes=self.num_genes,
        #                             init_range_low=self.init_range_low,
        #                             init_range_high=self.init_range_high,
        #                             parent_selection_type=self.parent_selection_type,
        #                             keep_parents=self.keep_parents,
        #                             crossover_type=self.crossover_type,
        #                             mutation_type=self.mutation_type,
        #                             mutation_percent_genes=self.mutation_percent_genes)

    def generate_trajectory(self, actions):
        state = 'aircraft-north'
        pos = self.start_pos
        positions = [pos]
        for action in actions:
            pos = list(pos)
            pos[2] -= self.rod  # glide descent (engine fail)
            if action == 0:
                hdg = self.movement[state]
            if 1 <= action <= 2:
                # left turn (90 degrees)
                state = self.aircraft_actions[state][action - 1]
                hdg = self.movement[state]
            self.movement[state]
            pos_next = pos.copy()
            pos_next[0] += hdg[0]
            pos_next[1] += hdg[1]
            if pos_next[0] > 0 and pos_next[1] > 0 and pos_next[0] < 256 and pos_next[1] < 256:
                pos = pos_next
            positions.append(pos)
        return positions

    def calculate_goal_dist(self, actions, action_idx):
        positions = self.generate_trajectory(actions)
        # print(f"final position: {positions[-1]}")
        # print(f"error: {(positions[-1] - self.goal_pos)}")
        # print(f"error_sum: {np.sum((positions[-1] - self.goal_pos)**2)}")
        return np.sum((positions[-1] - self.goal_pos)**2)
        # return np.sum(np.sqrt(np.sum((positions - self.goal_pos)**2, axis=1)))



def make_fitness_func(start_pos=np.array([128, 128, 64]), goal_pos=np.array([24, 156, 0]), airspeed=5):
    def fitness_func(actions, action_idx):
        ga = TrackCalc(start_pos, goal_pos, airspeed)
        return -ga.calculate_goal_dist(actions, action_idx)
    return fitness_func

def make_ga(ga_params, fitness_func):

    ga = pygad.GA(num_generations=ga_params["num_generations"],
                num_parents_mating=ga_params["num_parents_mating"],
                fitness_func=fitness_func,
                sol_per_pop=ga_params["sol_per_pop"],
                num_genes=ga_params["num_genes"],
                init_range_low=ga_params["init_range_low"],
                init_range_high=ga_params["init_range_high"],
                parent_selection_type=ga_params["parent_selection_type"],
                keep_parents=ga_params["keep_parents"],
                crossover_type=ga_params["crossover_type"],
                mutation_type=ga_params["mutation_type"],
                gene_type=ga_params["gene_type"],
                mutation_percent_genes=ga_params["mutation_percent_genes"])

    return ga


if __name__=="__main__":

    ga_params = {"num_generations": 20000,
                 "num_parents_mating": 4,
                 "sol_per_pop": 8,
                 "num_genes": 64,  # action size
                 "init_range_low": 0,
                 "init_range_high": 3,
                 "parent_selection_type": "sss",
                 "keep_parents": 1,
                 "crossover_type": "single_point",
                 "mutation_type": "random",
                 "mutation_percent_genes": 10,
                 "gene_type": int}

    # start_pos = np.array([0, 0, 64])
    # goal_pos = np.array([24, 156, 0])
    # airspeed = 5
    # ga = TrackCalc(start_pos, goal_pos, airspeed)
    
    # actions = np.random.randint(0, 3, size=64)
    # print(f"actions: {actions}")
    # positions = ga.generate_trajectory(actions)
    # print(f"positions: {positions}")
    # print(f"goal_dist: {ga.calculate_goal_dist(actions, 0)}")

    fitness_func = make_fitness_func()

    ga = pygad.GA(num_generations=ga_params["num_generations"],
                  num_parents_mating=ga_params["num_parents_mating"],
                  fitness_func=fitness_func,
                  sol_per_pop=ga_params["sol_per_pop"],
                  num_genes=ga_params["num_genes"],
                  init_range_low=ga_params["init_range_low"],
                  init_range_high=ga_params["init_range_high"],
                  parent_selection_type=ga_params["parent_selection_type"],
                  keep_parents=ga_params["keep_parents"],
                  crossover_type=ga_params["crossover_type"],
                  mutation_type=ga_params["mutation_type"],
                  gene_type=ga_params["gene_type"],
                  mutation_percent_genes=ga_params["mutation_percent_genes"])
    ga.run()
    ga.plot_fitness()
    best_action, best_fitness, best_id = ga.best_solution()

    start_pos = np.array([128, 128, 64])
    goal_pos = np.array([24, 156, 0])
    airspeed = 5
    tc = TrackCalc(start_pos, goal_pos, airspeed)
    pos = tc.generate_trajectory(best_action)
    print(f'best_final_pos: {pos[-1]}')

    # ga.plot_new_solution_rate()
