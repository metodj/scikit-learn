import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

EPSILONS = [0.01, 0.1, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5]




def eval_preds(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: accuracy metrics
    """
    print(f"Accuracy {round(accuracy_score(y_true, y_pred), 4)}")
    print(f"Baseline accuracy {round(1 - sum(y_true) / y_true.shape[0], 4)}")
    print(f"F1 score {round(f1_score(y_true, y_pred, average='binary'), 4)}")
    return confusion_matrix(y_true, y_pred)


def accuracy_dp_check(model_, X, y, threshold_fun, epsilons=EPSILONS, f1=False):
    """
    Test accuracy-privacy tradeoff out-of-sample.

    :param model_: sklearn model to test
    :param X:
    :param y:
    :param threshold_fun: threshold function. Needed since we use regression trees for classification purposes.
    :param epsilons: list of diff. private epsilons to check performance metrics for.
    :param f1: return f1 scores for exponential mechanism in addition to accuracy. Was added since privacy-utility
                tradeoff was not observed for exponential mechanism.
    """

    leaves_dp, internal_nodes_dp, internal_nodes_dp_f1 = [], [], []
    kf = KFold(n_splits=5, shuffle=True)
    for epsilon in epsilons:
        tmp_nodes, tmp_leaves, tmp_nodes_f1 = [], [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index,], X[test_index], y[train_index], y[test_index]
            # add noise to leaves
            leaves_model = model_(max_bins=255, epsilon_dp_leaves=epsilon).fit(X_train, y_train)
            leaves_pred = np.vectorize(threshold_fun)(leaves_model.predict(X_test))
            tmp_leaves.append(accuracy_score(y_test, leaves_pred))
            # add noise to inner nodes
            nodes_model = model_(max_bins=255, epsilon_dp_internal_nodes=epsilon).fit(X_train, y_train)
            nodes_pred = np.vectorize(threshold_fun)(nodes_model.predict(X_test))
            tmp_nodes.append(accuracy_score(y_test, nodes_pred))
            if f1:
                tmp_nodes_f1.append(f1_score(y_test, nodes_pred))
        leaves_dp.append(np.mean(tmp_leaves))
        internal_nodes_dp.append(np.mean(tmp_nodes))
        if f1:
            internal_nodes_dp_f1.append(np.mean(tmp_nodes_f1))
    print("Accuracy w.r.t. leaf noise: ", leaves_dp)
    print("Accuracy w.r.t. inner node noise: ", internal_nodes_dp)
    if f1:
        print("F1 w.r.t. inner node noise: ", internal_nodes_dp_f1)

        plt.plot(*[tuple(epsilons), tuple(internal_nodes_dp_f1)])
        plt.title("EXPONENTIAL MECHNANISM")
        plt.xlabel('epsilon_dp_internal_nodes')
        plt.ylabel('f1_score')
        plt.show()

    plt.plot(*[tuple(epsilons), tuple(internal_nodes_dp)])
    plt.title("EXPONENTIAL MECHNANISM")
    plt.xlabel('epsilon_dp_internal_nodes')
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(*[tuple(epsilons), tuple(leaves_dp)])
    plt.title("LAPLACIAN MECHNANISM")
    plt.xlabel('epsilon_dp_leaves')
    plt.ylabel('accuracy')
    plt.show()



def run_model(model_, X, y):
    """
    Test utility-privacy tradeoff  in-sample.

    :param model_: sklearn model to test
    """
    epsilons = [0.01, 0.1, 0.3, 0.5, 1, 2, 5, 10]
    leaves_dp, internal_nodes_dp = [], []

    for epsilon in epsilons:
        tmp_nodes = []
        tmp_leaves = []
        for _ in range(10):
            # only adding epsilon_dp_internal_nodes
            clf = model_(max_bins=255, epsilon_dp_internal_nodes=epsilon).fit(X, y)
            tmp_nodes.append(clf.score(X, y))
            # only adding epsilon_dp_leaves
            clf = model_(max_bins=255, epsilon_dp_leaves=epsilon).fit(X, y)
            tmp_leaves.append(clf.score(X, y))
        internal_nodes_dp.append(np.mean(tmp_nodes))
        leaves_dp.append(np.mean(tmp_leaves))

    print("R^2 w.r.t. leaf noise: ", leaves_dp)
    print("R^2 w.r.t. inner node noise: ", internal_nodes_dp)

    plt.plot(*[tuple(epsilons), tuple(internal_nodes_dp)])
    plt.title("EXPONENTIAL MECHNANISM")
    plt.xlabel('epsilon_dp_internal_nodes')
    plt.ylabel('R^2')
    plt.show()

    plt.plot(*[tuple(epsilons[2:]), tuple(leaves_dp[2:])])
    plt.title("LAPLACIAN MECHNANISM")
    plt.xlabel('epsilon_dp_leaves')
    plt.ylabel('R^2')
    plt.show()


def threshold_iris(x):
    if x <= 0.5:
        return 0
    elif 0.5 < x <= 1.5:
        return 1
    else:
        return 2


def threshold_telco(x):
    if x <= 0.5:
        return 0
    else:
        return 1
