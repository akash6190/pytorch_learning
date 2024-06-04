"""
A series of helper functions used for plotting.
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# Plot linear data or training  and test predictions
def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None
):
  """
  Plots linear training and test data and compares predictions.
  """
  plt.figure(figsize=(10,7))

  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
  
  if predictions is not None:

    plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

  plt.legend(prop={"size": 4})

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
  """
  Calculates accuracy between truth labels and predictions
  """
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = correct/ len(y_pred) * 100

  return acc


def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
  """
  Plots decision boundaries of model predicting on X in comparision 
to y
  Source: https://madewithml.com/courses/foundations/neural-networks/
  """
  # move everthing to cpu as it works well with NumPy and MatplotLib

  model.to('cpu')
  X = X.cpu()
  y = y.cpu()
  x_min, x_max = X[:, 0].min() - 0.1, X[:,0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1 ].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101 ), np.linspace(y_min, y_max, 101))

  X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel() , yy.ravel()))).float()

  model.eval() 

  with torch.inference_mode():
    y_logits = model(X_to_pred_on)

  if len(torch.unique(y)) > 2:
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
  else:
    y_pred = torch.round(torch.sigmoid(y_logits))

  y_pred = y_pred.reshape(xx.shape).detach().numpy()

  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

