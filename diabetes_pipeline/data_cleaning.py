import pandas as pd

# load the origin raw data
df = pd.read_csv("diabetes_raw.csv")

# classify the data by category
# 0 - no diabete; 1 - have diabete  
df_majority = df[df["Diabetes_binary"] == 0]
df_minority = df[df["Diabetes_binary"] == 1]

# obtain the quantity of minority classes
n_minority = len(df_minority)

# randomly downsample from most classes
df_majority_sampled = df_majority.sample(n=n_minority, random_state=42)

# merge data
df_balanced = pd.concat([df_majority_sampled, df_minority])

# disrupt the order of data
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# save the after-cleaning data 
df_balanced.to_csv("diabetes_binary_health_indicators_BRFSS2015.csv", index=False)