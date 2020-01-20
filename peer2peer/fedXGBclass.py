import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import time
import re

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', 500)


class XGBoostFederated:

    def __init__(self, data, labels, testsize=0.2):
        self.data = data
        self.labels = labels
        self.testsize = testsize

    @staticmethod
    def loglossobj(preds, dtrain, sigmoid=False):
        """logistic loss loss"""
        labels = dtrain.get_label()
        if sigmoid:
            def sigmoid_(x):
                return 1 / (1 + np.exp(-x))
            preds = sigmoid_(preds)
        grad = 1 / (1 + np.exp(-preds)) - labels
        hess = (1 / (1 + np.exp(-preds))) * (1 - 1 / (1 + np.exp(-preds)))
        return grad, hess

    @staticmethod
    def eval_preds(model, data, ntree_limit=0, logloss=False):
        ''' Evaluate predictions'''
        yhat = model.predict(data, ntree_limit=ntree_limit)
        yhat = 1.0 / (1.0 + np.exp(-yhat))
        yhat_labels = np.round(yhat)
        print(f"Accuracy {round(accuracy_score(data.get_label(), yhat_labels), 4)}")
        print(f"Baseline accuracy {round(1 - sum(data.get_label()) / data.get_label().shape[0], 4)}")
        print(f"F1 score {round(f1_score(data.get_label(), yhat_labels, average='binary'), 4)}")
        if logloss:
            print(f"Logloss {round(log_loss(data.get_label(), yhat), 4)}")
        return confusion_matrix(data.get_label(), yhat_labels)

    # TODO: implement nr_nodes argument
    @staticmethod
    def prepare_data(train, label, testsize, centralized, nr_nodes=2, batching=False):
        Xtrain, Xtest, ytrain, ytest = train_test_split(train, label, test_size=testsize, random_state=42)
        if centralized:
            dtrain = xgb.DMatrix(Xtrain, label=ytrain)
            dtest = xgb.DMatrix(Xtest, label=ytest)
            return dtrain, dtest
        else:
            example = xgb.DMatrix(Xtrain.iloc[:2, :], label=ytrain.iloc[:2])
            X1train, X2train, y1train, y2train = train_test_split(Xtrain, ytrain, test_size=0.5, random_state=42)
            X1test, X2test, y1test, y2test = train_test_split(Xtest, ytest, test_size=0.5, random_state=42)
            if batching:
                return X1train, X2train, y1train, y2train, X1test, X2test, y1test, y2test, example
            else:
                dtrain1 = xgb.DMatrix(X1train, label=y1train)
                dtrain2 = xgb.DMatrix(X2train, label=y2train)
                dtest1 = xgb.DMatrix(X1test, label=y1test)
                dtest2 = xgb.DMatrix(X2test, label=y2test)
                return dtrain1, dtrain2, dtest1, dtest2, example

    def train_centralized(self, params, nr_iter, booster=False, dump=False, hist=False):
        """
        Train XGBoost algorithm in centralized setting and print out the results.

        :param params: parameter dictionary
        :param nr_iter: number of training iterations
        :param booster: which API to use, train or booster [18.1.2020: better to use train,
        booster was the experimental version in the beginning]
        :param dump: whether or not to save models as .txt
        :param hist:
        :return: trained model
        """

        dtrain, dtest = self.prepare_data(self.data, self.labels, self.testsize, centralized=True)

        model = xgb.Booster(params, [dtrain])
        if dump:
            model.dump_model("initialize.txt")

        if booster:
            for _ in range(nr_iter):
                pred = model.predict(dtrain)
                g, h = self.loglossobj(pred, dtrain)
                model.boost(dtrain, g, h)
        else:
            model = xgb.train(params, dtrain, num_boost_round=nr_iter, xgb_model=model)

        if hist:
            for feature in list(X.columns):
                # if feature == 'tenure':
                hist_ = model.get_split_value_histogram(feature)
                print('---------- HISTOGRAM {} for feature {} -----------'.format(i, feature))
                print(hist_)

        print("----------Centralized results(in-sample)----------")
        print(self.eval_preds(model, dtrain))

        print("----------Centralized results(out-of-sample)----------")
        print(self.eval_preds(model, dtest))

        if dump:
            model.dump_model("out.txt")

        # # UPDATE [9.10.] get_split_value_histogram is returning different histograms than the ones I had in mind,
        # # i.e. it is not returning histograms that are passed from local nodes to master...
        # feature = "tenure"
        # xgdump = model.get_dump()
        # values = []
        # regexp = re.compile(r"\[{0}<([\d.Ee+-]+)\]".format(feature))
        # for i, _ in enumerate(xgdump):
        #     print(xgdump[i])
        #     m = re.findall(regexp, xgdump[i])
        #     print(m)
        #     values.extend([float(x) for x in m])
        # print(len(values))

        return model

    # TODO: implement nr_nodes argument (currently supports only setting with 2 cients)
    # TODO: implement random order of building trees (to improve privacy)
    def train_federated(self, params, nr_iter, node_to_eval, nr_nodes =2, booster=False, average_gradient=False,
                        dump=False, batching=False, polish_with_update=False):
        """
        :param params:
        :param nr_iter:
        :param node_to_eval:
        :param booster:
        :param average_gradient:
        :param dump:
        :param batching:
        :param polish_with_update: during first half training build new trees, in the second half update exsisting ones.
        This approach tries to circumvent problem of "tree number inflation" when training in federated setting.
        :return:
        """
        if batching:
            X1train, X2train, y1train, y2train, X1test, X2test, y1test, y2test, example = self.prepare_data(self.data,
                                            self.labels, self.testsize, centralized=False, batching=True)
            k1 = X1train.shape[0] // nr_iter
            k2 = X2train.shape[0] // nr_iter
        else:
            dtrain1, dtrain2, dtest1, dtest2, example = self.prepare_data(self.data, self.labels,
                                                                            self.testsize, centralized=False)

        model_federated = xgb.Booster(params, [example])

        if polish_with_update:
            nr_iter = nr_iter // nr_nodes

        for i in range(nr_iter):
            if batching and i < nr_iter - 1:
                dtrain1 = xgb.DMatrix(X1train.iloc[i * k1:(i + 1) * k1, :], label=y1train.iloc[i * k1:(i + 1) * k1])
                dtrain2 = xgb.DMatrix(X2train.iloc[i * k2:(i + 1) * k2, :], label=y2train.iloc[i * k2:(i + 1) * k2])
            elif batching and i == nr_iter - 1:  # adjusting for last batch...
                dtrain1 = xgb.DMatrix(X1train.iloc[i * k1:, :], label=y1train.iloc[i * k1:])
                dtrain2 = xgb.DMatrix(X2train.iloc[i * k2:, :], label=y2train.iloc[i * k2:])

            if booster:
                pred1 = model_federated.predict(dtrain1)
                pred2 = model_federated.predict(dtrain2)
                g1, h1 = self.loglossobj(pred1, dtrain1, sigmoid=True)
                g2, h2 = self.loglossobj(pred2, dtrain2, sigmoid=True)
                if average_gradient:  # looks like model is underfitting when averaging the gradients...
                    g = g1 + g2 / 2
                    h = h1 = h2 / 2
                    model_federated.boost(dtrain1, g, h)
                    model_federated.boost(dtrain2, g, h)
                else:
                    model_federated.boost(dtrain1, g1, h1)
                    model_federated.boost(dtrain2, g2, h2)

            else:  # in this case we do not consider average_gradient since it seems to perform much worse...
                model_federated = xgb.train(params, dtrain1, num_boost_round=1, xgb_model=model_federated)
                # in the framework, here we the model will be saved on the first client, then sent to the master node,
                # master will sent the model to the second client, that will load the model etc.
                model_federated = xgb.train(params, dtrain2, num_boost_round=1, xgb_model=model_federated)

        if polish_with_update:
            if dump:
                model_federated.dump_model("out_federated_pre_polish.txt")
            params['process_type'] = 'update'
            params['updater'] = 'prune,refresh'
            # params['refresh_leaf'] = True
            assert not booster
            nr_iter = nr_iter * nr_nodes
            model_federated = xgb.train(params, dtrain1, num_boost_round=nr_iter, xgb_model=model_federated)
            model_federated = xgb.train(params, dtrain2, num_boost_round=nr_iter, xgb_model=model_federated)


        node = node_to_eval
        assert node in [1, 2]

        train_local = dtrain1 if node == 1 else dtrain2
        test_local = dtest1 if node == 1 else dtest2

        print("----------Federated results for node {} (in-sample)----------".format(node))
        print(self.eval_preds(model_federated, train_local))

        print("----------Federated results for node {} (out-of-sample)----------".format(node))
        print(self.eval_preds(model_federated, test_local))

        if dump:
            model_federated.dump_model("out_federated.txt")

        return model_federated

    def train_local(self, params, nr_iter, node_to_eval, nr_nodes=2, booster=False):

        dtrain1, dtrain2, dtest1, dtest2, example = self.prepare_data(self.data, self.labels,
                                    self.testsize, centralized=False, nr_nodes=nr_nodes)

        node = node_to_eval
        assert node in [1, 2]
        train_local = dtrain1 if node == 1 else dtrain2
        test_local = dtest1 if node == 1 else dtest2

        model_local = xgb.Booster(params, [train_local])

        # for _ in range(nr_iter):
        #     if booster:
        #         pred = model_local.predict(train_local)
        #         g, h = self.loglossobj(pred, train_local)
        #         model_local.boost(train_local, g, h)
        #     else:
        #         model_local = xgb.train(params, train_local, num_boost_round=1, xgb_model=model_local)

        if booster:
            for _ in range(nr_iter):
                pred = model_local.predict(train_local)
                g, h = self.loglossobj(pred, train_local)
                model_local.boost(train_local, g, h)
        else:
            model_local = xgb.train(params, train_local, num_boost_round=nr_iter, xgb_model=model_local)

        print("----------Baseline local results for node {}(in-sample)----------".format(node))
        print(self.eval_preds(model_local, train_local))

        print("----------Baseline local results for node {}(out-of-sample)----------".format(node))
        print(self.eval_preds(model_local, test_local))
        return model_local

    def randomizedSearchCV(self, params, cv, n_iter):
        xgb_clf = xgb.XGBClassifier(objective='binary:logitraw')
        xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions=params, scoring=["accuracy", "f1"],
                                      cv=cv, verbose=1, random_state=40, n_jobs=-1, n_iter=n_iter, refit="f1")
        model_xgboost = xgb_rscv.fit(self.data, self.labels)
        print("----------Centralized results CV----------")
        print("F1 score {}".format(model_xgboost.best_score_))
        print(model_xgboost.best_params_)
        return model_xgboost


######################## TELCO DATASET ##################################################
# data preprocessing - XGB can only work with numerical features...
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0).astype('float32')
categorical_cols = list(set(list(data.columns)) - set(['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID']))
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data_encoded.TotalCharges = pd.to_numeric(data_encoded['TotalCharges'],errors='coerce')

X, y = data_encoded.loc[:, ~data_encoded.columns.isin(['Churn_Yes', 'customerID'])], data_encoded.Churn_Yes
nr_iter_ = 20


# TODO: bring in "nthread" param
params_ = {'max_depth': 4, 'eta': 0.1, 'verbosity': 1, 'max_delta_step': 0,
                'scale_pos_weight': 1.5, 'objective': 'binary:logitraw',
                    'tree_method':'hist', 'max_bin':250, 'nthread':-1}
# 'booster': 'dart','rate_drop': 0.1, 'skip_drop': 0.5}
# 'updater': 'grow_colmaker,prune,sync'
# 'grow_policy':'lossguide', 'max_leaves':2, 'colsample_bytree':0.9


parametersCV = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [10, 100, 250, 500, 1000],
               "scale_pos_weight": [1, 1.5, 1.8, 2, 2.7, 3]}


if __name__ == '__main__':
    # start = time.time()

    tmp = XGBoostFederated(X, y, 0.2)
    tmp_model3 = tmp.train_local(params_, nr_iter_, 1)
    tmp_model = tmp.train_centralized(params_, nr_iter_)
    tmp_model2 = tmp.train_federated(params_, nr_iter_, 1, dump=True, polish_with_update=True)
    tmp_model3 = tmp.train_local(params_, nr_iter_, 1)
    # tmp_model4 = tmp.randomizedSearchCV(parametersCV, 5, 10)

    # print("time elapsed: ", time.time()-start)

