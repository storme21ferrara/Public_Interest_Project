import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import seaborn as sns
from fishbone_diagram import FishboneDiagram

def root_cause_analysis(data, method="fishbone", output_file="rca_output.png"):
    if method == "fishbone":
        fb = FishboneDiagram(data)
        fb.create_diagram()
        fb.save_diagram(output_file)
    elif method == "tree":
        model = tree.DecisionTreeClassifier()
        X = data.drop(columns=["target"])
        y = data["target"]
        model.fit(X, y)
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(model, feature_names=X.columns, filled=True)
        fig.savefig(output_file)
    else:
        raise ValueError("Unsupported RCA method.")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
