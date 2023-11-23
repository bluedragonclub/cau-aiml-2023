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
