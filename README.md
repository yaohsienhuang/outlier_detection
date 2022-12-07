# outlierDetection
* 收錄常用的outlier detection models，並建立成方法便於後續應用
> ref: https://scikit-learn.org/stable/modules/outlier_detection.html

## 使用方式：
```python=
# 實例化
outlier_algo=outlierDetection()

# 選擇演算法
outlier_algo.select_algo('One-Class_SVM')

# 設定outliers_frac
outlier_algo.outliers_frac=0.1

# 開始訓練
outlier_algo.training(train_feature)

# 讀取pre_train model
outlier_algo.load_model('xxx.sav')

# 開始預測
prediction=outlier_algo.predict(test_feature)
```
## Outputs:
```
algorithms:
(1) Robust_covariance : EllipticEnvelope(contamination=0.2)
(2) One-Class_SVM : OneClassSVM(gamma='auto', nu=0.2)
(3) Isolation_Forest : IsolationForest(contamination=0.2, random_state=4)
(4) Local_Outlier_Factor : LocalOutlierFactor(contamination=0.2)

Robust_covariance has been selected.

outliers_fraction= 0.1

Robust_covariance start trainig...
```
