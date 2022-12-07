import os
import time
import pickle
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class outlierDetection:
    outliers_fraction=0.2
    algorithms = {
            "Robust_covariance":EllipticEnvelope(contamination=outliers_fraction),
            "One-Class_SVM":svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma='auto'),
            "Isolation_Forest":IsolationForest(contamination=outliers_fraction, random_state=4),
            "Local_Outlier_Factor":LocalOutlierFactor(contamination=outliers_fraction),
    }
    
    def __init__(self):
        print('algorithms:')
        n=1
        for name,algo in self.algorithms.items():
            print(f'({n}) {name} : {algo}')
            n+=1

    @property
    def outliers_frac(self):
        return self.outliers_fraction
        
    @outliers_frac.setter
    def outliers_frac(self,value):
        self.outliers_fraction=value
        print('outliers_fraction=',self.outliers_fraction)
    
    def select_algo(self,algo):
        self.algo=algo
        self.model=self.algorithms[algo]
        print('%s has been selected.'%self.algo)
    
    def training(self,dataset):
        print('%s start trainig...'%self.algo)
        self.trained_model=self.model.fit(dataset)
        self.save_model(self.trained_model)
    
    def predict(self,dataset):
        if self.algo == "Local_Outlier_Factor":
            prediction = self.trained_model.fit_predict(dataset)
        else:
            prediction = self.trained_model.predict(dataset)
        return prediction
    
    def save_model(self,model):
        time_str = time.strftime("%Y%m%d%H%M")
        folder=f'model/{self.algo}'
        if not os.path.isdir(f'model/{self.algo}'):
            os.makedirs(f'model/{self.algo}')
        filename = folder+f'{time_str}-{self.algo}_nu{self.outliers_fraction}.sav'
        pickle.dump(model, open(filename, 'wb'))
        print('model_save:',filename)
        
    def load_model(self,path):
        self.trained_model = pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    outlier_algo=outlierDetection()
    outlier_algo.select_algo('One-Class_SVM')
    outlier_algo.outliers_frac=0.1
    #outlier_algo.training(train_feature)
    outlier_algo.load_model('xxx.sav')
    #prediction=outlier_algo.predict(test_feature)
   
    