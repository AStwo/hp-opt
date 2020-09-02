from src.space.SearchSpace import UniformInt, Choice, Uniform


# Regression
def get_rfr_search_space(n_cols):
    return {
        "n_estimators": UniformInt("n_estimators", 10, 200),
        "criterion": Choice("criterion", ["mse", "mae"]),
        "max_depth": UniformInt("max_depth", 2, 30),
        "max_features": UniformInt("max_features", 2, n_cols),
        "min_impurity_decrease": Uniform("min_impurity_decrease", 1e-7, 0.1),
        "max_samples": Uniform("max_samples", 0.6, 0.99)
    }


def get_rtree_search_space(n_cols):
    return {
        "criterion": Choice("criterion", ["mse", "friedman_mse", "mae"]),
        "max_depth": UniformInt("max_depth", 2, 15),
        "max_features": UniformInt("max_features", 2, n_cols),
        "splitter": Choice("splitter", ["best", "random"]),
        "min_impurity_decrease": Uniform("min_impurity_decrease", 1e-7, 0.1)
}


def get_rknn_search_space():
    return {
        "n_neighbors": UniformInt("n_neighbors", 5, 50),
        "weights": Choice("weights", ["uniform", "distance"]),
        "algorithm": Choice("algorithm", ["ball_tree", "kd_tree", "brute"]),
        "p": UniformInt("p", 1, 3)
    }


# Classification
def get_cfr_search_space(n_cols):
    return {
        "n_estimators": UniformInt("n_estimators", 10, 200),
        "criterion": Choice("criterion", ["gini", "entropy"]),
        "max_depth": UniformInt("max_depth", 2, 30),
        "max_features": UniformInt("max_features", 2, n_cols),
        "min_impurity_decrease": Uniform("min_impurity_decrease", 1e-7, 0.1),
        "max_samples": Uniform("max_samples", 0.6, 0.99)
    }


def get_ctree_search_space(n_cols):
    return {
        "criterion": Choice("criterion", ["gini", "entropy"]),
        "max_depth": UniformInt("max_depth", 2, 15),
        "max_features": UniformInt("max_features", 2, n_cols),
        "splitter": Choice("splitter", ["best", "random"]),
        "min_impurity_decrease": Uniform("min_impurity_decrease", 1e-7, 0.1)
}


def get_cknn_search_space():
    return {
        "n_neighbors": UniformInt("n_neighbors", 5, 50),
        "weights": Choice("weights", ["uniform", "distance"]),
        "algorithm": Choice("algorithm", ["ball_tree", "kd_tree", "brute"]),
        "p": UniformInt("p", 1, 3)
    }