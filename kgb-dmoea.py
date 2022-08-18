import numpy as np
import random
import matplotlib.pyplot as plt
from nds import ndomsort  # TODO: Use ndomsort from pymoo instead
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import Hypervolume
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import euclidean
from alive_progress import alive_bar
import json


class KGBDMOEA(NSGA2):
    def __init__(
        self,
        c_size=13,
        eps=0.0,
        verbose_clusters=False,
        verbose_kre=False,
        ps={},
        pertub_dev=0.1,  # standard deviation of noise
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.eps = eps
        self.c_size = c_size
        self.ps = ps
        self.verbose_clusters = verbose_clusters
        self.verbose = verbose_kre
        self.rng = np.random.RandomState(self.seed)
        random.seed(self.seed)
        self.nr_rand_solutions = 50 * self.pop_size
        self.t = 0
        self.PERTUB_DEV = pertub_dev

    def setup(self, problem, **kwargs):
        assert (
            not problem.has_constraints()
        ), "KGB-DMOEA only works for unconstrained problems."
        return super().setup(problem, **kwargs)

    def knowledge_reconstruction_examination(self):

        clusters = self.ps  # set historical PS set as clusters
        Nc = self.c_size  # set final nr of clusters
        size = len(self.ps)  # set size iteration to length of cluster
        run_counter = 0  # counter variable to give unique key

        with alive_bar(size - Nc) as bar:

            # while there are still clusters to be condensed
            while size > Nc:

                counter = 0
                min_distance = None
                min_distance_index = []

                for keys_i in clusters.keys():
                    for keys_j in clusters.keys():
                        if (
                            clusters[keys_i]["solutions"]
                            is not clusters[keys_j]["solutions"]
                        ):

                            dst = euclidean(
                                clusters[keys_i]["centroid"],
                                clusters[keys_j]["centroid"],
                            )

                            if min_distance == None:
                                min_distance = dst
                                min_distance_index = [keys_i, keys_j]
                            elif dst < min_distance:
                                min_distance = dst

                                min_distance_index = [keys_i, keys_j]

                            counter += 1

                if self.verbose_clusters:
                    print("---------Stats---------")
                    print("Iteration:             ", run_counter)
                    print("Comparisons:           ", counter)
                    print("Nr. of Clusters:       ", len(clusters))
                    print("Nr. of final clusters  ", Nc)
                    print("Size:                  ", size)
                    print("min. distance:         ", min_distance)
                    print("min. distance clusters:", min_distance_index)
                    print(
                        "Cluster:",
                        min_distance_index[0],
                        self.ps[min_distance_index[0]],
                    )
                    print(
                        "Cluster:",
                        min_distance_index[1],
                        self.ps[min_distance_index[1]],
                    )
                    print("Updated Cluster", min_distance_index[0])
                    print("Removing Cluster", min_distance_index[1])
                    print()

                if self.verbose_clusters:
                    print(
                        "Appending Solutions from",
                        min_distance_index[1],
                        "to",
                        min_distance_index[0],
                    )

                for solution in clusters[min_distance_index[1]]["solutions"]:
                    clusters[min_distance_index[0]]["solutions"].append(solution)

                if self.verbose_clusters:
                    print("Calculating new centroid for cluster", min_distance_index[0])

                if self.verbose_clusters:
                    print(
                        "Old Cluster Centroid",
                        clusters[min_distance_index[0]]["centroid"],
                    )
                    print(
                        "Calculated Centroid",
                        self.calculate_cluster_centroid(
                            clusters[min_distance_index[0]]["solutions"]
                        ),
                    )

                clusters[min_distance_index[0]][
                    "centroid"
                ] = self.calculate_cluster_centroid(
                    clusters[min_distance_index[0]]["solutions"]
                )

                if self.verbose_clusters:
                    print(
                        "New cluster centroid is",
                        clusters[min_distance_index[0]]["centroid"],
                    )

                del clusters[min_distance_index[1]]

                if self.verbose_clusters:
                    print(clusters.keys())

                size -= 1
                run_counter += 1
                bar()

        c = []
        c_obj = []
        pop_useful = []
        pop_useless = []

        for key in clusters.keys():
            c.append(clusters[key]["centroid"])

        # create pymoo population objected to evaluate centroid solutions
        centroid_pop = Population.new("X", c)

        # evaluate population

        self.evaluator.eval(self.problem, centroid_pop)

        for individual in centroid_pop:
            c_obj.append(individual.F)

        fronts = ndomsort.non_domin_sort(c_obj)

        for individual in centroid_pop:
            if self.list_contains_array(fronts[0], individual.F):
                for key in clusters.keys():
                    if individual.X == clusters[key]["centroid"]:
                        for cluster_individual in clusters[key]["solutions"]:
                            pop_useful.append(cluster_individual)
            else:
                for key in clusters.keys():
                    if individual.X == clusters[key]["centroid"]:
                        for cluster_individual in clusters[key]["solutions"]:
                            pop_useless.append(cluster_individual)

        return pop_useful, pop_useless, c

    def naive_bayesian_classifier(self, pop_useful, pop_useless):

        labeled_useful_solutions = []
        labeled_useless_solutions = []

        for individual in pop_useful:
            labeled_useful_solutions.append((individual, +1))

        for individual in pop_useless:
            labeled_useless_solutions.append((individual, -1))

        x_train = []
        y_train = []

        for i in range(len(labeled_useful_solutions)):
            x_train.append(labeled_useful_solutions[i][0])
            y_train.append(labeled_useful_solutions[i][1])

        for i in range(len(labeled_useless_solutions)):
            x_train.append(labeled_useless_solutions[i][0])
            y_train.append(labeled_useless_solutions[i][1])

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        model = GaussianNB()
        model.fit(x_train, y_train)

        # visualize
        # # generate a lot of random solutions with the dimensions of problem decision space
        # X_test = rng.rand(nr_rand_solutions, problem.n_var)

        # # predict wether random solutions are useful or useless
        # Y_test = model.predict(X_test)

        # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=50, cmap="RdBu")
        # lim = plt.axis()
        # plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=20, cmap="RdBu", alpha=0.2)
        # plt.axis(lim)
        # plt.show()

        return model

    def add_to_ps(self):

        PS_counter = 0

        for individual in self.opt:

            if isinstance(individual.X, list):
                individual.X = np.asarray(individual.X)

            centroid = self.calculate_cluster_centroid(individual.X)

            self.ps[str(PS_counter) + "-" + str(self.t)] = {
                "solutions": [individual.X.tolist()],
                "centroid": centroid,
            }

            PS_counter += 1

    def list_contains_array(self, lst, arr):
        return any(np.array_equal(arr, elem) for elem in lst)

    def predicted_population(self, X_test, Y_test):
        predicted_pop = []
        for i in range(len(Y_test)):
            if Y_test[i] == 1:
                predicted_pop.append(X_test[i])
        return predicted_pop

    def calculate_cluster_centroid(self, solution_cluster):
        """Function that calculates the centroid for a given cluster of solutions
        Input: Array of Solutions
        Output: Centroid Coordinates
        """
        # Get number of variable shape
        try:
            n_vars = len(solution_cluster[0])
        except TypeError:
            solution_cluster = np.array(solution_cluster)
            return solution_cluster.tolist()

        # TODO: this is lazy garbage fix whats coming in
        cluster = []
        for i in range(len(solution_cluster)):
            # cluster.append(solution_cluster[i].tolist())
            cluster.append(solution_cluster[i])
        solution_cluster = np.asarray(cluster)

        # Get number of solutions
        length = solution_cluster.shape[0]

        centroid_points = []

        # calculate centroid for each variable, by taking mean of every variable of cluster
        for i in range(n_vars):
            # calculate sum over cluster
            centroid_points.append(np.sum(solution_cluster[:, i]))

        return [x / length for x in centroid_points]

    def check_boundaries(self, pop):

        # check wether numpy array or pymoo population is given
        if isinstance(pop, Population):
            pop = pop.get("X")

        # check if any solution is outside of the bounds
        for individual in pop:
            for i in range(len(individual)):
                if individual[i] > self.problem.xu[i]:
                    individual[i] = self.problem.xu[i]
                elif individual[i] < self.problem.xl[i]:
                    individual[i] = self.problem.xl[i]
        return pop

    def random_strategy(self, N_r):
        """Function that returns a randomly generated population in boundaries"""
        # generate a random population of size N_r
        # TODO: Check boundaries
        random_pop = np.random.random((N_r, self.problem.n_var))

        # check if any solution is outside of the bounds
        for individual in random_pop:
            for i in range(len(individual)):
                if individual[i] > self.problem.xu[i]:
                    individual[i] = self.problem.xu[i]
                elif individual[i] < self.problem.xl[i]:
                    individual[i] = self.problem.xl[i]

        return random_pop

    def diversify_population(self, pop):

        # find indices to be replaced (introduce diversity)
        I = np.where(np.random.random(len(pop)) < self.PERC_DIVERSITY)[0]
        # replace with randomly sampled individuals
        pop[I] = self.initialization.sampling(self.problem, len(I))
        return pop

    def _advance(self, **kwargs):

        pop = self.pop
        X, F = pop.get("X", "F")

        # the number of solutions to sample from the population to detect the change
        n_samples = int(np.ceil(len(pop) * self.perc_detect_change))

        # choose randomly some individuals of the current population to test if there was a change
        I = np.random.choice(np.arange(len(pop)), size=n_samples)
        samples = self.evaluator.eval(self.problem, Population.new(X=X[I]))

        # calculate the differences between the old and newly evaluated pop
        delta = ((samples.get("F") - F[I]) ** 2).mean()

        self.add_to_ps()

        # if there is an average deviation bigger than eps -> we have a change detected
        change_detected = delta > self.eps

        if change_detected:

            self.t += 1

            if self.verbose:
                # count number of individual solutions in ps
                n_solutions = 0
                for key in self.ps.keys():
                    n_solutions += len(self.ps[key]["solutions"])

                print(
                    "--------------------------------------------------------------------"
                )
                print()
                print("-> Detected environment change")
                print("-> Number of Clusters in PS", len(self.ps))
                print("-> Number of Solutions in PS", n_solutions)
                print(f"-> Conducting Knowledge Reconstruction-Examination (KRE)")
                print()

            # conduct knowledge reconstruction examination
            pop_useful, pop_useless, c = self.knowledge_reconstruction_examination()

            if self.verbose:
                print()
                print("-> Number of Clusters in PS after KRE", len(self.ps))
                print("-> Training Naive Bayesian Classifier")

            # Train a naive bayesian classifier
            model = self.naive_bayesian_classifier(pop_useful, pop_useless)

            if self.verbose:
                print("-> Generating new population")

            # generate a lot of random solutions with the dimensions of problem decision space
            X_test = self.random_strategy(self.nr_rand_solutions)
            # introduce noise to vary previously useful solutions
            noise = np.random.normal(0, self.PERTUB_DEV, self.problem.n_var)
            noisy_useful_history = np.asarray(pop_useful) + noise
            # check wether solutions are within bounds
            noisy_useful_history = self.check_boundaries(noisy_useful_history)
            # add noisy useful history to randomly generated solutions
            X_test = np.vstack((X_test, noisy_useful_history))
            # predict wether random solutions are useful or useless
            Y_test = model.predict(X_test)
            # create list of useful predicted solutions
            predicted_pop = self.predicted_population(X_test, Y_test)

            if self.verbose:
                print("-> Known Useful solutions", len(pop_useful))
                print("-> Known Useless solutions", len(pop_useless))
                print(
                    f"-> Available predicted solutions {len(predicted_pop)} from {len(X_test)} random generated solutions",
                )

            # ------ POPULATION GENERATION --------
            # take a random sample from predicted pop and known useful pop

            nr_sampled_pop_useful = 0
            nr_random_filler_solutions = 0

            if len(predicted_pop) >= self.pop_size - self.c_size:
                init_pop = []
                predicted_pop = random.sample(
                    predicted_pop, self.pop_size - self.c_size
                )
                # add sampled solutions to init_pop
                for solution in predicted_pop:
                    init_pop.append(solution)

                # add cluster centroids to init_pop
                for solution in c:
                    init_pop.append(np.asarray(solution))

                if self.verbose:
                    print(
                        "-> Number of Solutions after initial Population generation",
                        len(init_pop),
                    )
            else:
                # if not enough predicted solutions are available, add all predicted solutions to init_pop
                init_pop = []

                for solution in predicted_pop:
                    init_pop.append(solution)

                # add cluster centroids to init_pop
                for solution in c:
                    init_pop.append(np.asarray(solution))

            # if there are still not enough solutions in init_pop randomly sample previously useful solutions without noise to init_pop
            if len(init_pop) < self.pop_size:

                if self.verbose:
                    print()
                    print(
                        f"Only {len(init_pop)} predicted solutions available, filling with previous useful solutions"
                    )
                # fill up init_pop with randomly sampled solutions from pop_usefull
                if len(pop_useful) >= self.pop_size - len(init_pop):

                    nr_sampled_pop_useful = self.pop_size - len(init_pop)

                    init_pop = np.vstack(
                        (
                            init_pop,
                            random.sample(pop_useful, self.pop_size - len(init_pop)),
                        )
                    )
                else:
                    # if not enough solutions are available, add all previously known useful solutions without noise to init_pop
                    for solution in pop_useful:
                        init_pop.append(solution)

                    nr_sampled_pop_useful = len(pop_useful)

            # if there are still not enough solutions in init_pop generate random solutions with the dimensions of problem decision space
            if len(init_pop) < self.pop_size:
                if self.verbose:
                    print(
                        f"Only {len(init_pop)} solutions available, filling with random solutions"
                    )

                nr_random_filler_solutions = self.pop_size - len(init_pop)

                # fill up with random solutions
                init_pop = np.vstack(
                    (init_pop, self.random_strategy(self.pop_size - len(init_pop)))
                )

            if self.verbose:
                print()
                print("Population Markup:")
                print()
                print("-> Number of predicted solutions", len(predicted_pop))
                print("-> Number of centroid solutions", len(c))
                print(
                    "-> Number of directly sampled known usefull solutions",
                    nr_sampled_pop_useful,
                )
                print(
                    "-> Number of random filler solutions", nr_random_filler_solutions
                )
                print("-> Number of final solutions", len(init_pop))
                print()
                print(
                    "=========================================================================="
                )
                print(
                    "n_gen  |  n_eval  | n_nds  |      igd      |       gd      |       hv     "
                )
                print(
                    "=========================================================================="
                )
            # recreate the current population without being evaluated
            pop = Population.new(X=init_pop)

            # reevaluate because we know there was a change
            self.evaluator.eval(self.problem, pop)

            # do a survival to recreate rank and crowding of all individuals
            pop = self.survival.do(self.problem, pop, n_survive=len(pop))

        # create the offsprings from the current population
        off = self.mating.do(self.problem, pop, self.n_offsprings, algorithm=self)
        self.evaluator.eval(self.problem, off)

        # merge the parent population and offsprings
        pop = Population.merge(pop, off)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(
            self.problem, pop, n_survive=self.pop_size, algorithm=self
        )

        # dump self.ps to file
        # TODO: WRITE PS TO FILE AND END OF OPTIMIZATION -> shorter exec time?
        with open("ps.json", "w") as fp:
            json.dump(self.ps, fp)
