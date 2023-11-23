
# 과제#2 성별 분류 챌린지

- 한국인의 인체 치수를 이용하여 성별을 분류하는 머신러닝 모델을 개발하는 과제입니다.

- 인체 치수에 대한 자세한 설명은 수업 e-Class의 **파일** 자료실에 있는 설명 문서를 참고하시기 바랍니다.

- 모든 코드는 `classification.py` 파일에 작성하며, 파일명을 변경하면 안 됩니다. 파일명을 변경하는 경우 평가 시스템에서 오류가 발생하게 됩니다.

- 과제 진행 중에는 각 문제별로 검증(validation) 데이터의 분류 결과인 레이블 파일만 제출하실 수 있습니다.
  본 과제에는 문제가 하나 밖에 없기 때문에, `model01_result_valid.csv`만 제출하시면 됩니다.
  `classification.py`와 마찬가지로 파일명을 변경해서는 안 되며, [CSV 파일 형식](https://en.wikipedia.org/wiki/Comma-separated_values)을 갖추어야 합니다.

- 최종 테스트(test)를 위해 모델의 훈련(training) 및 예측(prediction)을 구현한 `classification.py`를 제출하셔야 합니다. 테스트는 과제 마감 이후 진행됩니다.

- 과제의 최종 점수는 제출 기한 이후 테스트 결과 순위에 따라 차등 결정됩니다.

- [BDC 클라이언트](https://github.com/bluedragonclub/bdc-client)를 이용하여 BDC 서버에 파일을 제출하시면 됩니다.

- BDC 클라이언트를 사용하는 방법은, 수업 Discord 서버의 `announcement` 채널에서 제공하는 튜토리얼 슬라이드를 참고하시기 바랍니다.

- 제출 기한은 **2023년 12월 7일 목요일 오후 11시 59분** 입니다.



## 문제(1)

- 본 문제의 목표는 주어진 인체 치수 데이터를 이용하여 **성별**을 분류하는 것입니다.

- 사용할 수 있는 인체 치수의 개수에는 제한이 없습니다. 

- `classification.py`에 머신러닝 모델을 구현합니다.

- 사용할 수 있는 외부 Python 패키지(3rd-party package)는 `numpy`, `scipy`, `pandas`, `scikit-learn` 입니다. 
  즉, `xgboost`, `tensorflow`, `pytorch`와 같은 외부 프레임워크는 사용할 수 없습니다.

- 문제 번호에 따라 분류 모델의 클래스 이름을 정의해야 합니다.
  예를 들어, 문제(1)의 모델 클래스 이름은 반드시 `Model01`로 정의해야 합니다.
  본 과제에는 문제(1) 하나 밖에 없기 때문에 `Model01` 클래스만 정의하면 됩니다.

- 다음은 [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)를 이용하여 분류 모델을 구현하는 예시입니다.
    ```
    # classification.py

    import numpy as np
    import pandas as pd
    from sklearn.naive_bayes import GaussianNB


    class ClassificationModel:
        def __init__(self):
            self._model = None
            
        @property
        def features(self):
            return self._features
        
        
        def train(self, df):
            X = df[self._features]
            y = df["성별"]

            self._model.fit(X, y)
        
        
        def predict(self, df):
            X = df[self._features]
            pred = self._model.predict(X)
            return pred
                   

    class Model01(ClassificationModel):
        def __init__(self):
            self._model = GaussianNB()
        
            self._features = [
                "눈높이",
                "목뒤높이",
                "손목둘레"
            ]
            
        # def train(self, df):
        #   You can override this function...
        
        
        # def predict(self, df):
        #   You can override this function...    

    ```


- 본 과제에서 분류 결과의 평가 척도는 [Accuray](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers#Single_metrics), [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), [PR-AUC](https://en.wikipedia.org/wiki/Precision_and_recall)를 사용합니다.
  평가 척도를 계산하기 위해 `scikit-learn`에서 제공하는 함수를 이용할 수 있습니다.
  모델 성능의 최종 평가 척도는 Accuracy, ROC-AUC, PR-AUC의 합으로 정의됩니다.

  ```
  Metric :=  Accuracy + ROC-AUC + PR-AUC
  ```
  
- 다음은 분류 모델의 훈련, 평가 척도의 계산, 검증 데이터에 대한 예측을 포함하는 예시 코드입니다.

    ```
    # predict_valid.py

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score

    from classification import Model01

    # Load the dataset.
    df_train = pd.read_csv("data_train.csv", encoding="utf-8-sig")
    df_valid = pd.read_csv("data_valid.csv", encoding="utf-8-sig")


    # Print all column names.
    # for col in df_train.columns:
    #    print(col)

    # Create the classification models.
    model = Model01()
    
    
    # Train and predict.
    model.train(df_train)

    y_pred = model.predict(df_train)
        
    # Calculate the metrics.
    y_train = df_train["성별"]
    acc = accuracy_score(y_train, y_pred)
    rocauc = roc_auc_score(y_train, y_pred)
    prauc = average_precision_score(y_train, y_pred)

    print("[Model#01] Acc: %f"%(acc))
    print("[Model#01] ROC-AUC: %f"%(rocauc))
    print("[Model#01] PR-AUC: %f"%(prauc))
    print()

    # Predict using the validation data.
    y_pred = model.predict(df_valid)

    # Output the validation result.
    df_result = pd.DataFrame(y_pred, columns=["성별"])
    df_result.index.name = "식별자"
    df_result.to_csv("model01_result_valid.csv",
                    encoding="utf-8-sig")


    ```

- 다음은 훈련 데이터(`data_train.csv`)의 일부 내용입니다. 머신러닝 모델이 예측해야 하는 성별 정보가 포함되어 있습니다.

   ```
   성별      키    눈높이   목뒤높이   ...
     1   1581.0    1445.0    1313.0   ...
     1   1520.0    1399.0    1263.0   ...
     1   1626.0    1501.0    1373.0   ...
     1   1690.0    1569.0    1427.0   ...
     0   1795.0    1660.0    1530.0   ...
     0   1771.0    1651.0    1503.0   ...
     0   1706.0    1567.0    1451.0   ...
     1   1688.0    1548.0    1448.0   ...
     1   1681.0    1570.0    1457.0   ...
     0   1774.0    1634.0    1494.0   ...
   ```


- 다음은 검증 데이터(`data_valid.csv`)의 일부 내용입니다. 훈련 데이터와 달리 성별 정보가 없으며 식별자 정보가 있습니다.
  식별자는 특정 사람을 인식하는 ID(identifier)입니다. 

   ```
   식별자       키     눈높이    목뒤높이  ...
     0     1594.0     1465.0    1323.0   ...
     1     1714.0     1605.0    1457.0   ...
     2     1670.0     1548.0    1419.0   ...
     3     1650.0     1529.0    1391.0   ...
     4     1655.0     1535.0    1418.0   ...
     5     1645.0     1527.0    1403.0   ...
     6     1683.0     1555.0    1415.0   ...
     7     1713.0     1588.0    1460.0   ...
     8     1616.0     1494.0    1384.0   ...
     9     1628.0     1528.0    1404.0   ...
   ```

- 검증 데이터에 대한 평가 척도의 계산 결과를 얻기 위하여 BDC 서버에 `model01_result_valid.csv` 파일을 제출합니다.
  `model01_result_valid.csv`의 예시는 다음과 같습니다.

     ```
   식별자       키     눈높이    목뒤높이  ...
     0     1594.0     1465.0    1323.0   ...
     1     1714.0     1605.0    1457.0   ...
     2     1670.0     1548.0    1419.0   ...
     3     1650.0     1529.0    1391.0   ...
     4     1655.0     1535.0    1418.0   ...
     5     1645.0     1527.0    1403.0   ...
     6     1683.0     1555.0    1415.0   ...
     7     1713.0     1588.0    1460.0   ...
     8     1616.0     1494.0    1384.0   ...
     9     1628.0     1528.0    1404.0   ...
   ```

- 최종적으로 테스트 데이터에 대한 결과를 얻기 위해서는 반드시 `classification.py`를 제출해야 합니다.
  `model01_result_valid.csv`의 제출을 통해 얻은 검증 결과는 최종 점수에 포함되지 않습니다.
  `classification.py`를 잘 작성하였는지 확인하기 위하여 `predict_valid.py`의 코드를 이용하시기 바랍니다. 


