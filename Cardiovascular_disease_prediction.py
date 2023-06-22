# Import tools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ydata_profiling as pp

# Import data
df = pd.read_csv("heart.csv")
print(df.shape)

# EDA
from ydata_profiling import ProfileReport
profile = pp.ProfileReport(df)
profile.to_file("report.html")

# Preprocessing
