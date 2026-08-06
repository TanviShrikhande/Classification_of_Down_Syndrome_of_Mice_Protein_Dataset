from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from random import randint
from sklearn.metrics import accuracy_score


def initialize_population(size: int, n_feat: int) -> list[np.ndarray]:
    population = []
    for _ in range(size):
        chromosome = np.ones(n_feat, dtype=bool)
        chromosome[: int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population: list[np.ndarray], X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model) -> tuple[list[float], list[np.ndarray]]:
    scores = []
    for chromosome in population:
        selected_train = X_train.iloc[:, chromosome]
        selected_test = X_test.iloc[:, chromosome]
        model.fit(selected_train, y_train)
        predictions = model.predict(selected_test)
        scores.append(accuracy_score(y_test, predictions))
    scores = np.array(scores)
    population = np.array(population)
    order = np.argsort(scores)
    return list(scores[order][::-1]), list(population[order][::-1])


def selection(pop_after_fit: list[np.ndarray], n_parents: int) -> list[np.ndarray]:
    return pop_after_fit[:n_parents]


def crossover(pop_after_sel: list[np.ndarray]) -> list[np.ndarray]:
    pop_nextgen = list(pop_after_sel)
    for i in range(0, len(pop_after_sel), 2):
        if i + 1 >= len(pop_after_sel):
            break
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i + 1]
        midpoint = len(child_1) // 2
        new_child = np.concatenate((child_1[:midpoint], child_2[midpoint:]))
        pop_nextgen.append(new_child)
    return pop_nextgen


def mutation(pop_after_cross: list[np.ndarray], mutation_rate: float, n_feat: int) -> list[np.ndarray]:
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []
    for chromo in pop_after_cross:
        mutated = chromo.copy()
        rand_positions = [randint(0, n_feat - 1) for _ in range(mutation_range)]
        for pos in rand_positions:
            mutated[pos] = not mutated[pos]
        pop_next_gen.append(mutated)
    return pop_next_gen


def run_genetic_selection(X: pd.DataFrame, y: pd.Series, model, size: int = 80, n_feat: int | None = None, n_parents: int = 64, mutation_rate: float = 0.20, n_gen: int = 5) -> tuple[list[np.ndarray], list[float], list[str]]:
    n_feat = n_feat or X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    population = initialize_population(size, n_feat)
    best_chromosomes = []
    best_scores = []
    for _ in range(n_gen):
        scores, pop_after_fit = fitness_score(population, X_train, X_test, y_train, y_test, model)
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population = mutation(pop_after_cross, mutation_rate, n_feat)
        best_chromosomes.append(pop_after_fit[0])
        best_scores.append(scores[0])
    selected_features = [feature for feature, selected in zip(X.columns, best_chromosomes[-1]) if selected]
    return best_chromosomes, best_scores, selected_features
