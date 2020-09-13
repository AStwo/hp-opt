from datetime import datetime

import dill
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score

import exp.data as data
import exp.searchspaces as ss
from src.optimizer import RandomOptimizer, GeneticOptimizer, BayesianOptimizer


def test_optimizers(search_space, objective, iterations, func, dataset_name, alg, suffix, seed, verbose=1):
    ro = RandomOptimizer(search_space, seed)
    ga = GeneticOptimizer(search_space, 10, 1.0, 0.1, seed)
    bo = BayesianOptimizer(search_space, "matern", seed)
    solutions = {}

    if verbose:
        print(f"Optimizing {alg} on dataset {dataset_name}")
        print(f"Optimization start: {datetime.now()}\n")

    for opt, opt_name in zip([ro, ga, bo], ["ro", "ga", "bo"]):
        t0 = datetime.now()
        if verbose:
            print(f"Optimizer {opt_name} starts at {t0}")
        opt.optimize(func, iterations, objective=objective)
        solutions[opt_name] = opt.best_solution
        dill.dump(opt, open(f"results/{dataset_name}/{alg}-{opt_name}_{suffix}.pkl", "wb"))
        if verbose:
            print(f"Best target: {opt.best_target}")
            print(f"Best params: {opt.best_solution}")
            print(f"Time: {datetime.now() - t0}\n")

    if verbose:
        print(f"Optimization of {alg} on dataset {dataset_name} ended at {datetime.now()}")
        print("------\n")
    return solutions


def test_solutions(alg_name, alg, solutions, data_dict, metric, **kwargs):
    results = {}
    for opt_name in ["ro", "ga", "bo"]:
        model = alg(**solutions[opt_name], **kwargs)
        model.fit(data_dict["X"], data_dict["y"])
        y_pred = model.predict(data_dict["X_test"])
        result = metric(data_dict["y_test"], y_pred)
        results[opt_name] = result
        print(f"Test score achieved by {opt_name}: {result} ({alg_name})")

    return results


def test_long():
    data_wine_c = data.read_wine_data(task="class")
    data_wine_r = data.read_wine_data(task="reg")
    data_mnist = data.read_mnist_data()
    data_bike = data.read_bike_data()

    objective = "min"
    iterations = 100
    seed = 420

    test_wine_c = True
    test_wine_r = True
    test_mnist = True
    test_bike = True

    # Wine type
    if test_wine_c:
        def f_wine_c_tree(**kwargs):
            dt = DecisionTreeClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_wine_c["X"], data_wine_c["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_wine_c_rf(**kwargs):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_wine_c["X"], data_wine_c["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_wine_c_ad(**kwargs):
            ad = AdaBoostClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_wine_c["X"], data_wine_c["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        search_space_tr = ss.get_ctree_search_space(data_wine_c["X"].shape[1])
        search_space_rf = ss.get_cfr_search_space(data_wine_c["X"].shape[1])
        search_space_ad = ss.get_cadb_search_space()

        print("\n================================")
        print("==  Wine type classification  ==")
        print("================================")

        solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_wine_c_tree, "wine_c", "tree", "", seed)
        solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_wine_c_rf, "wine_c", "rf", "", seed)
        solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_wine_c_ad, "wine_c", "ad", "", seed)

        test_solutions("tr", DecisionTreeClassifier, solutions_tr, data_wine_c, accuracy_score, random_state=seed)
        test_solutions("rf", RandomForestClassifier, solutions_rf, data_wine_c, accuracy_score, random_state=seed, n_jobs=-2)
        test_solutions("ad", AdaBoostClassifier, solutions_ad, data_wine_c, accuracy_score, random_state=seed)

    # Wine quality
    if test_wine_r:
        def f_wine_r_tree(**kwargs):
            dt = DecisionTreeRegressor(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_wine_r["X"], data_wine_r["y"], cv=4, scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_wine_r_rf(**kwargs):
            rf = RandomForestRegressor(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_wine_r["X"], data_wine_r["y"], cv=4, scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_wine_r_ad(**kwargs):
            ad = AdaBoostClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_wine_r["X"], data_wine_r["y"], cv=4, scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        search_space_tr = ss.get_rtree_search_space(data_wine_r["X"].shape[1])
        search_space_rf = ss.get_rfr_search_space(data_wine_r["X"].shape[1])
        search_space_ad = ss.get_radb_search_space()

        print("\n===============================")
        print("==  Wine quality regression  ==")
        print("===============================")

        solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_wine_r_tree, "wine_r", "tree", "", seed)
        solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_wine_r_rf, "wine_r", "rf", "", seed)
        solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_wine_r_ad, "wine_r", "ad", "", seed)

        test_solutions("tr", DecisionTreeRegressor, solutions_tr, data_wine_r, mean_squared_error, random_state=seed)
        test_solutions("rf", RandomForestRegressor, solutions_rf, data_wine_r, mean_squared_error, random_state=seed, n_jobs=-2)
        test_solutions("ad", AdaBoostRegressor, solutions_ad, data_wine_r, mean_squared_error, random_state=seed)

    # Bike sharing
    if test_bike:
        def f_bike_tree(**kwargs):
            dt = DecisionTreeRegressor(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_bike["X"], data_bike["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_bike_rf(**kwargs):
            rf = RandomForestRegressor(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_bike["X"], data_bike["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_bike_ad(**kwargs):
            ad = AdaBoostRegressor(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_bike["X"], data_bike["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        search_space_tr = ss.get_rtree_search_space(data_bike["X"].shape[1])
        search_space_rf = ss.get_rfr_search_space(data_bike["X"].shape[1])
        search_space_ad = ss.get_radb_search_space()

        print("\n===============================")
        print("==  Bike sharing regression  ==")
        print("===============================")

        solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_bike_tree, "bike", "tree", "", seed)
        solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_bike_rf, "bike", "rf", "", seed)
        solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_bike_ad, "bike", "ad", "", seed)

        test_solutions("tr", DecisionTreeRegressor, solutions_tr, data_bike, mean_squared_error, random_state=seed)
        test_solutions("rf", RandomForestRegressor, solutions_rf, data_bike, mean_squared_error, random_state=seed, n_jobs=-2)
        test_solutions("ad", AdaBoostRegressor, solutions_ad, data_bike, mean_squared_error, random_state=seed)

    # MNIST
    if test_mnist:
        def f_mnist_tree(**kwargs):
            dt = DecisionTreeClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_mnist["X"], data_mnist["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_mnist_rf(**kwargs):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_mnist["X"], data_mnist["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_mnist_ad(**kwargs):
            ad = AdaBoostClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_mnist["X"], data_mnist["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        search_space_tr = ss.get_ctree_search_space(data_mnist["X"].shape[1])
        search_space_rf = ss.get_cfr_search_space(data_mnist["X"].shape[1])
        search_space_ad = ss.get_radb_search_space()

        print("\n============================")
        print("==  MNIST classification  ==")
        print("============================")

        solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_mnist_tree, "mnist", "tree", "", seed)
        solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_mnist_rf, "mnist", "rf", "", seed)
        solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_mnist_ad, "mnist", "ad", "", seed)

        test_solutions("tr", DecisionTreeClassifier, solutions_tr, data_mnist, accuracy_score, random_state=seed)
        test_solutions("rf", RandomForestClassifier, solutions_rf, data_mnist, accuracy_score, random_state=seed, n_jobs=-2)
        test_solutions("ad", AdaBoostClassifier, solutions_ad, data_mnist, accuracy_score, random_state=seed)
        

def test_short():
    data_wine_c = data.read_wine_data(task="class")
    data_wine_r = data.read_wine_data(task="reg")
    data_mnist = data.read_mnist_data()
    data_bike = data.read_bike_data()

    objective = "min"
    iterations = 10
    seed = 420

    test_wine_c = False
    test_wine_r = False
    test_mnist = True
    test_bike = False

    # Wine type
    if test_wine_c:
        def f_wine_c_tree(**kwargs):
            dt = DecisionTreeClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_wine_c["X"], data_wine_c["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_wine_c_rf(**kwargs):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_wine_c["X"], data_wine_c["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_wine_c_ad(**kwargs):
            ad = AdaBoostClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_wine_c["X"], data_wine_c["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        search_space_tr = ss.get_ctree_search_space(data_wine_c["X"].shape[1])
        search_space_rf = ss.get_cfr_search_space(data_wine_c["X"].shape[1])
        search_space_ad = ss.get_cadb_search_space()

        print("\n================================")
        print("==  Wine type classification  ==")
        print("================================")

        results_tr = {"ro": [], "ga": [], "bo": []}
        results_rf = {"ro": [], "ga": [], "bo": []}
        results_ad = {"ro": [], "ga": [], "bo": []}
        for i in range(10):
            print(f"Iteration {i}")
            solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_wine_c_tree, "wine_c", "tree", i, seed+i)
            solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_wine_c_rf, "wine_c", "rf", i, seed+i)
            solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_wine_c_ad, "wine_c", "ad", i, seed+i)

            res_iter_tr = test_solutions("tr", DecisionTreeClassifier, solutions_tr, data_wine_c, accuracy_score, random_state=seed+i)
            res_iter_rf = test_solutions("rf", RandomForestClassifier, solutions_rf, data_wine_c, accuracy_score, random_state=seed+i, n_jobs=-2)
            res_iter_ad = test_solutions("ad", AdaBoostClassifier, solutions_ad, data_wine_c, accuracy_score, random_state=seed)

            for opt_name in ["ro", "ga", "bo"]:
                results_tr[opt_name].append(res_iter_tr[opt_name])
                results_rf[opt_name].append(res_iter_rf[opt_name])
                results_ad[opt_name].append(res_iter_ad[opt_name])

        print(results_tr)
        print(results_rf)
        print(results_ad)

    # Wine quality
    if test_wine_r:
        def f_wine_r_tree(**kwargs):
            dt = DecisionTreeRegressor(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_wine_r["X"], data_wine_r["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_wine_r_rf(**kwargs):
            rf = RandomForestRegressor(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_wine_r["X"], data_wine_r["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_wine_r_ad(**kwargs):
            ad = AdaBoostClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_wine_r["X"], data_wine_r["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        search_space_tr = ss.get_rtree_search_space(data_wine_r["X"].shape[1])
        search_space_rf = ss.get_rfr_search_space(data_wine_r["X"].shape[1])
        search_space_ad = ss.get_radb_search_space()

        print("\n===============================")
        print("==  Wine quality regression  ==")
        print("===============================")

        results_tr = {"ro": [], "ga": [], "bo": []}
        results_rf = {"ro": [], "ga": [], "bo": []}
        results_ad = {"ro": [], "ga": [], "bo": []}
        for i in range(10):
            print(f"Iteration {i}")
            solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_wine_r_tree, "wine_r", "tree", i, seed + i)
            solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_wine_r_rf, "wine_r", "rf", i, seed + i)
            solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_wine_r_ad, "wine_r", "ad", i,seed + i)

            res_iter_tr = test_solutions("tr", DecisionTreeRegressor, solutions_tr, data_wine_r, mean_squared_error, random_state=seed + i)
            res_iter_rf = test_solutions("rf", RandomForestRegressor, solutions_rf, data_wine_r, mean_squared_error, random_state=seed + i, n_jobs=-2)
            res_iter_ad = test_solutions("ad", AdaBoostRegressor, solutions_ad, data_wine_r, mean_squared_error, random_state=seed)

            for opt_name in ["ro", "ga", "bo"]:
                results_tr[opt_name].append(res_iter_tr[opt_name])
                results_rf[opt_name].append(res_iter_rf[opt_name])
                results_ad[opt_name].append(res_iter_ad[opt_name])

        print(results_tr)
        print(results_rf)
        print(results_ad)

    # Bike sharing
    if test_bike:
        def f_bike_tree(**kwargs):
            dt = DecisionTreeRegressor(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_bike["X"], data_bike["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_bike_rf(**kwargs):
            rf = RandomForestRegressor(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_bike["X"], data_bike["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        def f_bike_ad(**kwargs):
            ad = AdaBoostRegressor(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_bike["X"], data_bike["y"], cv=4,
                                      scoring="neg_root_mean_squared_error").mean()
            return cv_acc

        search_space_tr = ss.get_rtree_search_space(data_bike["X"].shape[1])
        search_space_rf = ss.get_rfr_search_space(data_bike["X"].shape[1])
        search_space_ad = ss.get_radb_search_space()

        print("\n===============================")
        print("==  Bike sharing regression  ==")
        print("===============================")

        results_tr = {"ro": [], "ga": [], "bo": []}
        results_rf = {"ro": [], "ga": [], "bo": []}
        results_ad = {"ro": [], "ga": [], "bo": []}
        for i in range(10):
            print(f"Iteration {i}")
            solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_bike_tree, "bike", "tree", i, seed + i)
            solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_bike_rf, "bike", "rf", i, seed + i)
            solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_bike_ad, "bike", "ad", i, seed + i)

            res_iter_tr = test_solutions("tr", DecisionTreeRegressor, solutions_tr, data_bike, mean_squared_error, random_state=seed + i)
            res_iter_rf = test_solutions("rf", RandomForestRegressor, solutions_rf, data_bike, mean_squared_error, random_state=seed + i, n_jobs=-2)
            res_iter_ad = test_solutions("ad", AdaBoostRegressor, solutions_ad, data_bike, mean_squared_error, random_state=seed)

            for opt_name in ["ro", "ga", "bo"]:
                results_tr[opt_name].append(res_iter_tr[opt_name])
                results_rf[opt_name].append(res_iter_rf[opt_name])
                results_ad[opt_name].append(res_iter_ad[opt_name])

        print(results_tr)
        print(results_rf)
        print(results_ad)

    # MNIST
    if test_mnist:
        def f_mnist_tree(**kwargs):
            dt = DecisionTreeClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(dt, data_mnist["X"], data_mnist["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_mnist_rf(**kwargs):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-2, **kwargs)
            cv_acc = -cross_val_score(rf, data_mnist["X"], data_mnist["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        def f_mnist_ad(**kwargs):
            ad = AdaBoostClassifier(random_state=seed, **kwargs)
            cv_acc = -cross_val_score(ad, data_mnist["X"], data_mnist["y"], cv=4,
                                      scoring="accuracy").mean()
            return cv_acc

        search_space_tr = ss.get_ctree_search_space(data_mnist["X"].shape[1])
        search_space_rf = ss.get_cfr_search_space(data_mnist["X"].shape[1])
        search_space_ad = ss.get_radb_search_space()

        print("\n============================")
        print("==  MNIST classification  ==")
        print("============================")

        results_tr = {"ro": [], "ga": [], "bo": []}
        results_rf = {"ro": [], "ga": [], "bo": []}
        results_ad = {"ro": [], "ga": [], "bo": []}
        for i in range(10):
            print(f"Iteration {i}")
            solutions_tr = test_optimizers(search_space_tr, objective, iterations, f_mnist_tree, "mnist", "tree", i, seed + i)
            solutions_rf = test_optimizers(search_space_rf, objective, iterations, f_mnist_rf, "mnist", "rf", i, seed + i)
            solutions_ad = test_optimizers(search_space_ad, objective, iterations, f_mnist_ad, "mnist", "ad", i, seed + i)

            res_iter_tr = test_solutions("tr", DecisionTreeClassifier, solutions_tr, data_mnist, accuracy_score, random_state=seed + i)
            res_iter_rf = test_solutions("rf", RandomForestClassifier, solutions_rf, data_mnist, accuracy_score, random_state=seed + i, n_jobs=-2)
            res_iter_ad = test_solutions("ad", AdaBoostClassifier, solutions_ad, data_mnist, accuracy_score, random_state=seed)

            for opt_name in ["ro", "ga", "bo"]:
                results_tr[opt_name].append(res_iter_tr[opt_name])
                results_rf[opt_name].append(res_iter_rf[opt_name])
                results_ad[opt_name].append(res_iter_ad[opt_name])

        print(results_tr)
        print(results_rf)
        print(results_ad)


if __name__ == '__main__':
    start_time = datetime.now()
    print("Initializing long test...")
    # test_long()
    print("\n======")
    print("Initializing short test...")
    test_short()
    print("\n======")
    print(f"Total time: {datetime.now()-start_time}")