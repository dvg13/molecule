from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay, roc_auc_score
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

def top_k_precision(Y_true, prediction, k=20):
  top_k_idx = np.argpartition(prediction, -k, axis=0)[-k:]
  top_k_labels = Y_true[top_k_idx]
  top_k_precision = np.mean(top_k_labels)
  return top_k_precision

def false_negative_rate(Y_true, prediction, threshold=.5):
  return sum(np.logical_and(Y_true == 1,prediction < threshold)) / sum(prediction < threshold)

def get_precisions(Y_true, prediction):
  precision = {}
  recall = {}
  average_precision = {}
  for i in range(Y_true.shape[1]):
    precision[i], recall[i], _ = precision_recall_curve(Y_true[:, i], prediction[:, i])
    average_precision[i] = average_precision_score(Y_true[:, i], prediction[:, i])

  # A "micro-average": quantifying score on all classes jointly
  precision["micro"], recall["micro"], _ = precision_recall_curve(
    Y_true.ravel(), prediction.ravel()
  )
  average_precision["micro"] = average_precision_score(Y_true, prediction, average="micro")

  return precision, recall, average_precision

def plot_micro_average_precision_curve(precision, recall, average_precision):
  display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
  )
  display.plot()
  display.ax_.set_title("Micro-averaged over all classes")

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
def plot_per_label_precision_curves(n, precision, recall, average_precision):
  colors = ["b","g","r","c","m","y","k","w","tab:orange", "tab:purple", "tab:gray", "tab:brown"]
  _, ax = plt.subplots(figsize=(7, 8))
  for i in range(n):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for label {i+1}", color=colors[i])
  handles, labels = display.ax_.get_legend_handles_labels()
  ax.set_xlim([0.0, 2])
  ax.set_ylim([0.0, 1.2])
  ax.legend(handles=handles, labels=labels, loc="best")
  ax.set_title("Precision-recall by label")

def get_multilabel_stats(model, X, y, k=20):
  prediction = model.predict(X)

  #auc
  auroc = roc_auc_score(y,prediction)

  #average precision
  precision, recall, average_precision = get_precisions(y,prediction)
  my_table = PrettyTable(["", "Average"] + [str(i) for i in range(1,y.shape[1]+1)])

  #auprc
  my_table.add_row(
      ["Average Precision", "{:.2f}".format(average_precision["micro"])] +
      ["{:.2f}".format(average_precision[i]) for i in range(y.shape[1])]
  )

  #top k precision
  top_k_precisions = [top_k_precision(y[:,i], prediction[:,i]) for i in range(y.shape[1])]
  my_table.add_row(
      ["Top {} Precision".format(k), "{:.2f}".format(np.mean(top_k_precisions))] +
      ["{:.2f}".format(x) for x in top_k_precisions]
  )

  #false negative rate
  false_negative_rates = [false_negative_rate(y[:,i], prediction[:,i]) for i in range(y.shape[1])]
  my_table.add_row(
      ["False negative rate", "{:.2f}".format(np.mean(false_negative_rates))] +
      ["{:.2f}".format(x) for x in false_negative_rates]
  )

  print("Auroc: {}".format(auroc))
  print(my_table)

  #precision recall graphs
  plot_micro_average_precision_curve(precision, recall, average_precision)
  plot_per_label_precision_curves(y.shape[1], precision, recall, average_precision)
