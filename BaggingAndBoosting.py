import warnings
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
import random as rand
import pandas as pd


def read_data(dtset):
    data = pd.read_csv(f'Datasets\\{dtset}', header=None, sep=',|,\s+|\t')
    data.iloc[:, -1] = data.iloc[:, -1].astype('category').cat.codes
    return data


def bagging(num_trees, main_dataset=pd.DataFrame):
    trees = []
    dataset_lenght = main_dataset.shape[0]
    for i in range(num_trees):
        idx = list(np.random.randint(0, dataset_lenght - 1, dataset_lenght))

        tree = DecisionTreeClassifier()
        tree.fit(main_dataset.iloc[idx, :-1], main_dataset.iloc[idx, -1])
        trees.append(tree)

    return trees


def classify(x, trees=list):
    predicts = []
    for tree in trees:
        predicts.append(tree.predict(np.array(x).reshape((1, -1)))[0])
    pred_, counts = np.unique(predicts, return_counts=True)
    argmax = np.argmax(counts)

    return pred_[argmax]


def bagging_tests(dataset, trees):
    y_hats = []
    for i in range(dataset.shape[0]):
        data = dataset.iloc[i, :-1]
        y_hat = classify(data, trees)
        y_hats.append(y_hat)
    acc = np.sum(y_hats == dataset.iloc[:, -1]) / len(y_hats) * 100
    return acc


def train_test_split(dataset):
    train, test = sklearn.model_selection.train_test_split(dataset, test_size=0.3, shuffle=True)
    return train, test


def noisy(dataset, p):
    dataset = dataset.copy()
    for i in range(dataset.shape[0]):
        features = rand.sample(range(dataset.shape[1] - 1), int((dataset.shape[1] - 1) * p))
        data = dataset.iloc[i, :-1]
        for j in features:
            data[j] += np.random.normal(0, 1)
        dataset.iloc[i, :-1] = data

    return dataset


def adaboost(iteration, main_dataset=pd.DataFrame):
    main_dataset.loc[main_dataset.iloc[:, -1] != 1] = -1
    m = main_dataset.shape[0]
    weights = np.full((m,), 1 / m, np.float)
    trees = []
    alphas = []
    for i in range(iteration):
        tree = DecisionTreeClassifier(max_depth=1, max_features=1)
        X, y = main_dataset.iloc[:, :-1], main_dataset.iloc[:, -1]
        tree.fit(X, y, weights)
        pred = tree.predict(X)

        epsilon = 1e-10
        e = np.sum(weights[y != pred])
        a = 0.5 * np.log((1 - e) / (e + epsilon))
        weights *= np.exp(-a * y * pred)
        weights /= np.sum(weights)
        alphas.append(a)
        trees.append(tree)

    return trees, alphas


def boosting_tests(dataset, trees, alphas):
    dataset.loc[dataset.iloc[:, -1] != 1] = -1
    y_hats = []
    for i in range(dataset.shape[0]):
        data = np.array(dataset.iloc[i, :-1]).reshape((1, -1))
        y_hat = int(np.sign(np.sum(alphas[j] * trees[j].predict(data) for j in range(len(trees))))[0])
        y_hats.append(y_hat)

    return np.sum(y_hats == np.array(dataset.iloc[:, -1])) / len(y_hats) * 100


with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    # Bagging calculate
    print('\n' + " BAGGING ".center(52, '*'))
    print('Best Accuracy And Best Number Of Tree In Each Dataset')
    print('-' * 52)

    acc_dataset_bagging = []
    acc_dataset_bagging_noisy10 = []
    acc_dataset_bagging_noisy20 = []
    acc_dataset_bagging_noisy30 = []

    dt_name = ['Wine.txt', 'Glass.txt', 'BreastTissue.txt', 'Diabetes.txt', 'Sonar.txt', 'Ionosphere.txt']
    for ds in dt_name:
        dataset = read_data(ds)
        # print('\r"Wine.txt" Training...', end='')
        train, test = train_test_split(dataset)
        best_n_trees = 0
        best_acc = 0
        best_n_trees_noise10 = 0
        best_acc_noise10 = 0
        best_n_trees_noise20 = 0
        best_acc_noise20 = 0
        best_n_trees_noise30 = 0
        best_acc_noise30 = 0

        for i in [11, 21, 31, 41]:
            # noise less
            bagging_trees = bagging(num_trees=i, main_dataset=train)
            bagging_acc = bagging_tests(test, bagging_trees)
            # noisy 10%
            tree_nois10 = bagging(num_trees=i, main_dataset=noisy(train, 0.1))
            bagging_noisy10_acc = bagging_tests(test, tree_nois10)
            # noisy 20%
            tree_nois20 = bagging(num_trees=i, main_dataset=noisy(train, 0.2))
            bagging_noisy20_acc = bagging_tests(test, tree_nois20)
            # noisy 30%
            tree_nois30 = bagging(num_trees=i, main_dataset=noisy(train, 0.3))
            bagging_noisy30_acc = bagging_tests(test, tree_nois30)

            #  noise less
            if bagging_acc >= best_acc:
                best_acc = bagging_acc
                best_n_trees = i
            #  noisy 10
            if bagging_noisy10_acc >= best_acc_noise10:
                best_acc_noise10 = bagging_noisy10_acc
                best_n_trees_noise10 = i
            #  noisy 20
            if bagging_noisy20_acc >= best_acc_noise20:
                best_acc_noise20 = bagging_noisy20_acc
                best_n_trees_noise20 = i
            #  noisy 30
            if bagging_noisy30_acc >= best_acc_noise30:
                best_acc_noise30 = bagging_noisy30_acc
                best_n_trees_noise30 = i
        # print('-'*20 + '******' + '-'*20)
        print(f'noiseless : Dataset "{ds}" {"-->":^5}  Accuracy : {best_acc:<5.2f} || Number of tree : {best_n_trees} ')
        # print(f'noisy 10% : Dataset "{ds}" {"-->":^5} Accuracy : {best_acc_noise10:<5.2f} || Number of tree : {best_n_trees_noise10} ')
        # print(f'noisy 20% : Dataset "{ds}" {"-->":^5} Accuracy : {best_acc_noise20:<5.2f} || Number of tree : {best_n_trees_noise20} ')
        # print(f'noisy 30% : Dataset "{ds}" {"-->":^5} Accuracy : {best_acc_noise30:<5.2f} || Number of tree : {best_n_trees_noise30} ')

        acc_dataset_bagging.append(best_acc)
        acc_dataset_bagging_noisy10.append(best_acc_noise10)
        acc_dataset_bagging_noisy20.append(best_acc_noise20)
        acc_dataset_bagging_noisy30.append(best_acc_noise30)

    # AdaBoost calculate

    print('\n' + " AdaBoost ".center(52, '*'))
    print('Best Accuracy And Best Number Of Iteration In Each Dataset')
    print('-' * 52)

    acc_dataset_boosting = []
    acc_dataset_boosting_noisy10 = []
    acc_dataset_boosting_noisy20 = []
    acc_dataset_boosting_noisy30 = []

    dt_adaboost = ['Diabetes.txt', 'Sonar.txt', 'Ionosphere.txt']
    for dt in dt_adaboost:
        dataset = read_data(dt)
        train, test = train_test_split(dataset)
        best_n_trees = 0
        best_acc = 0
        best_n_trees_noise10 = 0
        best_acc_noise10 = 0
        best_n_trees_noise20 = 0
        best_acc_noise20 = 0
        best_n_trees_noise30 = 0
        best_acc_noise30 = 0

        for i in [21, 31, 41, 51]:
            # noise less
            boosting_trees, alphas = adaboost(iteration=i, main_dataset=train)
            boosting_acc = boosting_tests(test, boosting_trees, alphas)
            # noisy 10%
            tree_boost10, alphas10 = adaboost(iteration=i, main_dataset=noisy(train, 0.1))
            boosting_noisy10_acc = boosting_tests(test, tree_boost10, alphas10)
            # noisy 20%
            tree_boost20, alphas20 = adaboost(iteration=i, main_dataset=noisy(train, 0.2))
            boosting_noisy20_acc = boosting_tests(test, tree_boost20, alphas20)
            # noisy 30%
            tree_boost30, alphas30 = adaboost(iteration=i, main_dataset=noisy(train, 0.3))
            boosting_noisy30_acc = boosting_tests(test, tree_boost30, alphas30)

            #  noise less
            if boosting_acc >= best_acc:
                best_acc = boosting_acc
                best_n_trees = i
            #  noisy 10
            if boosting_noisy10_acc >= best_acc_noise10:
                best_acc_noise10 = boosting_noisy10_acc
                best_n_trees_noise10 = i
            #  noisy 20
            if boosting_noisy20_acc >= best_acc_noise20:
                best_acc_noise20 = boosting_noisy20_acc
                best_n_trees_noise20 = i
            #  noisy 30
            if boosting_noisy30_acc >= best_acc_noise30:
                best_acc_noise30 = boosting_noisy30_acc
                best_n_trees_noise30 = i
        # print('-' * 20 + '******' + '-' * 20)
        print(
            f'noiseless : Dataset "{dt}" {"-->":^5} Accuracy : {best_acc:<5.2f} || Number of iteration : {best_n_trees} ')
        # print(f'noisy 10% : Dataset "{dt}" {"-->":^5} Accuracy : {best_acc_noise10:<5.2f} || Number of iteration : {best_n_trees_noise10} ')
        # print(f'noisy 20% : Dataset "{dt}" {"-->":^5} Accuracy : {best_acc_noise20:<5.2f} || Number of iteration : {best_n_trees_noise20} ')
        # print(f'noisy 30% : Dataset "{dt}" {"-->":^5} Accuracy : {best_acc_noise30:<5.2f} || Number of iteration : {best_n_trees_noise30} ')
        acc_dataset_boosting.append(best_acc)
        acc_dataset_boosting_noisy10.append(best_acc_noise10)
        acc_dataset_boosting_noisy20.append(best_acc_noise20)
        acc_dataset_boosting_noisy30.append(best_acc_noise30)

    print('\n')
    # __________________________ NOISELESS ______________________________________
    print('\n' + " NOISSLESS ".center(52, '=') + '\n')
    print(f'{" Datasets ":^20}|{" Bagging ":^10}|{" AdaBoost.M1 ":^20}')
    print('-' * 20 + '+' + '-' * 10 + '+' + '-' * 20)
    print(f'{"Wine":^20}|{acc_dataset_bagging[0]:^8.2f}% |{"None":^20}')
    print(f'{"Glass":^20}|{acc_dataset_bagging[1]:^8.2f}% |{"None":^20}')
    print(f'{"BreastTissue":^20}|{acc_dataset_bagging[2]:^8.2f}% |{"None":^20}')
    print(f'{"Diabetes":^20}|{acc_dataset_bagging[3]:^8.2f}% |     {acc_dataset_boosting[0]:^8.2f} %     ')
    print(f'{"Sonar":^20}|{acc_dataset_bagging[4]:^8.2f}% |     {acc_dataset_boosting[1]:^8.2f} %     ')
    print(f'{"Ionosphere":^20}|{acc_dataset_bagging[5]:^8.2f}% |     {acc_dataset_boosting[2]:^8.2f} %     ')

    # ____________________________ NOISY 10% ____________________________________
    print('\n' + " NOISY 10% ".center(52, '=') + '\n')
    print(f'{" Datasets ":^20}|{" Bagging ":^10}|{" AdaBoost.M1 ":^20}')
    print('-' * 20 + '+' + '-' * 10 + '+' + '-' * 20)
    print(f'{"Wine":^20}|{acc_dataset_bagging_noisy10[0]:^8.2f}% |{"None":^20}')
    print(f'{"Glass":^20}|{acc_dataset_bagging_noisy10[1]:^8.2f}% |{"None":^20}')
    print(f'{"BreastTissue":^20}|{acc_dataset_bagging_noisy10[2]:^8.2f}% |{"None":^20}')
    print(
        f'{"Diabetes":^20}|{acc_dataset_bagging_noisy10[3]:^8.2f}% |     {acc_dataset_boosting_noisy10[0]:^8.2f} %     ')
    print(f'{"Sonar":^20}|{acc_dataset_bagging_noisy10[4]:^8.2f}% |     {acc_dataset_boosting_noisy10[1]:^8.2f} %     ')
    print(
        f'{"Ionosphere":^20}|{acc_dataset_bagging_noisy10[5]:^8.2f}% |     {acc_dataset_boosting_noisy10[2]:^8.2f} %     ')

    # ____________________________ NOISY 20% ____________________________________
    print('\n' + " NOISY 20% ".center(52, '=') + '\n')
    print(f'{" Datasets ":^20}|{" Bagging ":^10}|{" AdaBoost.M1 ":^20}')
    print('-' * 20 + '+' + '-' * 10 + '+' + '-' * 20)
    print(f'{"Wine":^20}|{acc_dataset_bagging_noisy20[0]:^8.2f}% |{"None":^20}')
    print(f'{"Glass":^20}|{acc_dataset_bagging_noisy20[1]:^8.2f}% |{"None":^20}')
    print(f'{"BreastTissue":^20}|{acc_dataset_bagging_noisy20[2]:^8.2f}% |{"None":^20}')
    print(
        f'{"Diabetes":^20}|{acc_dataset_bagging_noisy20[3]:^8.2f}% |     {acc_dataset_boosting_noisy20[0]:^8.2f} %     ')
    print(f'{"Sonar":^20}|{acc_dataset_bagging_noisy20[4]:^8.2f}% |     {acc_dataset_boosting_noisy20[1]:^8.2f} %     ')
    print(
        f'{"Ionosphere":^20}|{acc_dataset_bagging_noisy20[5]:^8.2f}% |     {acc_dataset_boosting_noisy20[2]:^8.2f} %     ')

    # ____________________________ NOISY 30% ____________________________________
    print('\n' + " NOISY 30% ".center(52, '=') + '\n')
    print(f'{" Datasets ":^20}|{" Bagging ":^10}|{" AdaBoost.M1 ":^20}')
    print('-' * 20 + '+' + '-' * 10 + '+' + '-' * 20)
    print(f'{"Wine":^20}|{acc_dataset_bagging_noisy30[0]:^8.2f}% |{"None":^20}')
    print(f'{"Glass":^20}|{acc_dataset_bagging_noisy30[1]:^8.2f}% |{"None":^20}')
    print(f'{"BreastTissue":^20}|{acc_dataset_bagging_noisy30[2] :^8.2f}% |{"None":^20}')
    print(
        f'{"Diabetes":^20}|{acc_dataset_bagging_noisy30[3]:^8.2f}% |     {acc_dataset_boosting_noisy30[0]:^8.2f} %     ')
    print(f'{"Sonar":^20}|{acc_dataset_bagging_noisy30[4]:^8.2f}% |     {acc_dataset_boosting_noisy30[1]:^8.2f} %     ')
    print(
        f'{"Ionosphere":^20}|{acc_dataset_bagging_noisy30[5]:^8.2f}% |     {acc_dataset_boosting_noisy30[2]:^8.2f} %     ')
