import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from pathlib import Path

WORKING_DIR = str(Path(__file__).resolve().parents[1])

SAMPLES_NUM = 500
REPEAT = 2

# uniform dist constants
UNIFORM_CONSTANTS = {'X': (-10, 2), 'Y': (18, 45)}
GAUSSIAN_CONSTANTS = {'CENTERS': [1, 2, 4], 'STD': 0.5}
CLUMPS_CONSTANTS = {'CENTERS': [(0, 0), (0, 2), (5, 0), (5, 2)], 'STD': [1, 0.3]}
CIRCLE_CENTERS = [(0, 0), (1, 0.75)]
CIRCLE_RADIUS = 1
KMEANS_RANGE = range(2, 6)
HIERARCHICAL_RANGE = range(2, 5)
AVG, MAX, MIN = 'AVG', 'MAX', 'MIN'
DISTANCE_MATRICES = {AVG: "average", MAX: "complete", MIN: "single"}


def plot_scattered_graph(x, y, graph_name, q_folder=3, labels=None):
    """
    plots a scattered graph for given 2D-samples
    :param x: x axis samples
    :param y: y axis samples
    :param graph_name: the graph's name
    :param q_folder: Q3 or Q4 folder num (int)
    :param labels: labels for Q4
    :return: None
    """
    results_path = os.path.join(WORKING_DIR, 'output', f"Q{q_folder}_images")
    results_path = os.path.join(results_path, graph_name.replace('\n', '_') + '.png')
    plt.title(graph_name)
    plt.scatter(x, y, c=labels, cmap='rainbow')
    plt.savefig(results_path)
    # plt.show()
    plt.clf()


def problem_3a():
    """
    Uniform distribution
    :return: points and graph's name
    """
    x_dist = UNIFORM_CONSTANTS['X']
    x_points = [random.uniform(x_dist[0], x_dist[1]) for i in range(SAMPLES_NUM)]

    y_dist = UNIFORM_CONSTANTS['Y']
    y_points = [random.uniform(y_dist[0], y_dist[1]) for i in range(SAMPLES_NUM)]
    return x_points, y_points, "Uniform distribution"


def problem_3b():
    """
    Three Gaussians with centers at [i, −i]
    :return: points and graph's name
    """
    x_points = []
    y_points = []
    std = GAUSSIAN_CONSTANTS['STD']
    for center in GAUSSIAN_CONSTANTS['CENTERS']:
        std_loop = std * center
        x_points += list(np.random.normal(center, std_loop, SAMPLES_NUM))
        y_points += list(np.random.normal(-center, std_loop, SAMPLES_NUM))
    return x_points, y_points, "Gaussian distribution"


def scatter_the_letter_N():
    """
    :return: scattered points for the letter N
    """
    points = SAMPLES_NUM // 2
    x_points = [random.uniform(-2.7, -2.5) for i in range(points // 3)]
    y_points = [random.uniform(0, 2) for i in range(points // 3)]

    x_points += [random.uniform(-2.5, -0.5) for i in range(points // 3)]
    y_points += [-d - 0.5 for d in x_points[points // 3:]]

    x_points += [random.uniform(-0.7, -0.5) for i in range((points // 3) + 1)]
    y_points += [random.uniform(0, 2) for i in range((points // 3) + 1)]
    return x_points, y_points


def scatter_the_letter_A():
    """
    :return: scattered points for the letter A
    """
    points = SAMPLES_NUM // 2
    x_points = [random.uniform(0, 2) for i in range(points // 3)]
    y_points = x_points[:]

    x_points += [random.uniform(1, 3.2) for i in range(points // 3)]
    y_points += [random.uniform(0.9, 1) for i in range(points // 3)]

    x_points += [random.uniform(2, 4) for i in range((points // 3) + 1)]
    y_points += [4 - d for d in x_points[2 * (points // 3):]]
    return x_points, y_points


def problem_3c():
    """
    scatters 'NA'
    :return: points and graph's name
    """
    x, y = scatter_the_letter_N()
    x_a, y_a = scatter_the_letter_A()
    return x + x_a, y + y_a, "Graph of the letters 'N' and 'A'"


def problem_3d():
    """
    Four horizontal clumps
    :return: points and graph's name
    """
    x_std = CLUMPS_CONSTANTS['STD'][0]
    y_std = CLUMPS_CONSTANTS['STD'][1]
    # plt.ylim(-4.5, 6)
    x_points = []
    y_points = []
    for center in CLUMPS_CONSTANTS['CENTERS']:
        x_points += list(np.random.normal(center[0], x_std, SAMPLES_NUM // 4))
        y_points += list(np.random.normal(center[1], y_std, SAMPLES_NUM // 4))
    return x_points, y_points, "Clumps Graph"


def problem_3e():
    """
    Two moons
    :return: points and graph's name
    """
    angles_left = np.linspace(0, np.pi, SAMPLES_NUM // 2)
    angles_right = np.linspace(np.pi, 2 * np.pi, SAMPLES_NUM // 2)
    angles = [angles_left, angles_right]

    x, y = [], []
    for i in range(2):
        cx, cy = CIRCLE_CENTERS[i]
        for angle in angles[i]:
            x.append(cx + CIRCLE_RADIUS * np.cos(angle))
            y.append(cy + CIRCLE_RADIUS * np.sin(angle))
    # plt.ylim(-1, 2)
    return x, y, "Half Moon Graph"


# todo - finish implementation
def problem_3f():
    """
    Two moons, sparsely connected
    :return: points and graph's name
    """
    angles_left = np.linspace(0, np.pi, (SAMPLES_NUM // 2) - (SAMPLES_NUM // 10))
    angles_right = np.linspace(np.pi, 2 * np.pi, (SAMPLES_NUM // 2) - (SAMPLES_NUM // 10))
    angles = [angles_left, angles_right]

    x, y = [], []
    for i in range(2):
        cx, cy = CIRCLE_CENTERS[i]
        for angle in angles[i]:
            x.append(cx + CIRCLE_RADIUS * np.cos(angle))
            y.append(cy + CIRCLE_RADIUS * np.sin(angle))
        x += [random.uniform(cx - 0.05, cx + 0.05) for i in range(SAMPLES_NUM // 10)]
        if i == 0:
            y += [random.uniform(cy + CIRCLE_RADIUS - 0.25, cy + CIRCLE_RADIUS) for i in range(SAMPLES_NUM // 10)]
        else:
            y += [random.uniform(cy + -CIRCLE_RADIUS, cy + -CIRCLE_RADIUS + 0.25) for i in range(SAMPLES_NUM // 10)]
    # plt.ylim(-1, 2)
    return x, y, "Half Moon Sparsely Connected Graph"


def problem_4a(x_samples, y_samples, graph_name):
    """
    Plots K-means for given 2D samples
    :param x_samples: x axis sample points
    :param y_samples: y axis sample points
    :param graph_name: the graphs name
    :return: None
    """
    samples = np.column_stack((np.array(x_samples), np.array(y_samples)))
    for k in KMEANS_RANGE:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(samples)
        labels = kmeans.predict(samples)
        plot_scattered_graph(x_samples, y_samples, f"KMEANS of {graph_name}, K={k}", 4, labels)


def problem_4b(x_samples, y_samples, graph_name, distance_type: str):
    """
    Plots hierarchical clustering for given 2D samples
    :param x_samples: x axis sample points
    :param y_samples: y axis sample points
    :param graph_name: the graphs name
    :param distance_type: metric - MIN, MAX, AVG
    :return: None
    """
    samples = np.column_stack((np.array(x_samples), np.array(y_samples)))
    for k in HIERARCHICAL_RANGE:
        cluster = AgglomerativeClustering(n_clusters=k, linkage=DISTANCE_MATRICES[distance_type])
        labels = cluster.fit_predict(samples)
        plot_scattered_graph(x_samples, y_samples, f"{distance_type} hierarchical clustering of {graph_name}, K={k}", 4,
                             labels)


functions_list = [problem_3a, problem_3b, problem_3c, problem_3d, problem_3e, problem_3f]


def problem_3():
    """
    Running all of the functions in problem 3 for 'REPEAT' iterations
    :return: None
    """
    print("Running Question 3")
    for func in functions_list:
        for r in range(REPEAT):
            x_sample, y_sample, graph_name = func()
            graph_name += f"\nAttempt number {r+1}"
            plot_scattered_graph(x_sample, y_sample, graph_name)


def problem_4():
    """
    Running all of the functions in problem 3 for 'REPEAT' iterations and for KMEANS & hierarchical clustering
    :return: None
    """
    print("Running Question 4")
    for func in functions_list:
        for r in range(REPEAT):
            x_samples, y_samples, graph_name = func()
            graph_name += f"\nAttempt number {r+1}"
            problem_4a(x_samples, y_samples, graph_name)
            for dist_type in DISTANCE_MATRICES.keys():
                problem_4b(x_samples, y_samples, graph_name, dist_type)


problem_3()
problem_4()
