import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.svm

class AutoTrainer():
    '''
        Class that contains all supported auto-training models

        For documentation, refer to https://github.com/FanaticPythoner/AutoAi
    '''
    def getModelsTypes(self):
        '''
            Return all supported autotrain models

            RETURNS -> List : models
        '''
        return [
            (sklearn.ensemble.RandomForestClassifier, {}),
            (sklearn.ensemble.RandomForestRegressor, {}),
            (sklearn.ensemble.GradientBoostingClassifier, {}),
            (sklearn.ensemble.AdaBoostClassifier, {}),
            (sklearn.ensemble.AdaBoostRegressor, {}),
            (sklearn.ensemble.ExtraTreesClassifier, {}),
            (sklearn.ensemble.GradientBoostingRegressor, {}),
            (sklearn.ensemble.BaggingClassifier, {}),
            (sklearn.ensemble.BaggingRegressor, {}),
            (sklearn.ensemble.ExtraTreesRegressor, {}),
            (sklearn.ensemble.IsolationForest, {}),
            (sklearn.ensemble.StackingClassifier, {"estimators" : [
                                                                      ('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial', dual=False, max_iter=1200000)),
                                                                      ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                                                                      ('gnb', sklearn.naive_bayes.GaussianNB())
                                                                  ],
                                                  "final_estimator" : sklearn.ensemble.RandomForestClassifier(n_estimators=50)}),
            (sklearn.ensemble.StackingRegressor, {"estimators" : [
                                                                     ('lr', sklearn.linear_model.RidgeCV()),
                                                                     ('rf', sklearn.svm.LinearSVR())
                                                                 ],
                                                  "final_estimator" : sklearn.ensemble.RandomForestRegressor(n_estimators=50)}),
            (sklearn.ensemble.VotingClassifier, {"estimators" : [
                                                                    ('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial', dual=False, max_iter=1200000)),
                                                                    ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                                                                    ('gnb', sklearn.naive_bayes.GaussianNB())
                                                                ]}),
            (sklearn.ensemble.VotingRegressor, {"estimators" : [
                                                                   ('lr', sklearn.linear_model.LinearRegression()),
                                                                   ('rf', sklearn.ensemble.RandomForestRegressor(n_estimators=50))
                                                               ]}),

            (sklearn.linear_model.LogisticRegression, {}),
            (sklearn.linear_model.LinearRegression, {}),
            (sklearn.linear_model.Ridge, {}),
            (sklearn.linear_model.RidgeCV, {}),
            (sklearn.linear_model.ARDRegression, {}),
            (sklearn.linear_model.BayesianRidge, {}),
            (sklearn.linear_model.HuberRegressor, {}),
            (sklearn.linear_model.OrthogonalMatchingPursuit, {}),
            (sklearn.linear_model.PassiveAggressiveClassifier, {}),
            (sklearn.linear_model.PassiveAggressiveRegressor, {}),
            (sklearn.linear_model.Perceptron, {}),
            (sklearn.linear_model.RidgeClassifier, {}),
            (sklearn.linear_model.SGDClassifier, {}),
            (sklearn.linear_model.SGDRegressor, {}),
            (sklearn.linear_model.TheilSenRegressor, {})
        ]
