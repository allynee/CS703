import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_FILE = '2024-03-26_16-51-32train_data.csv'

df = pd.read_csv(DATA_FILE)
fig = make_subplots(
  rows=3,
  cols=1,
  subplot_titles=(
    'Loss Over Epoch-Batch',
    'Average Accuracy Over Epoch-Batch',
    'Batch Accuracy Over Epoch-Batch'
  ),
  vertical_spacing=0.1
) 

# Loss
fig.add_trace(
  go.Scatter(
    x=df['epoch'].astype(str) + '-' + df['current_batch'].astype(str),
    y=df['loss'],
    mode='lines+markers',
    name='Loss'
  ),
  row=1,
  col=1
)

# Average Accuracy
fig.add_trace(
  go.Scatter(
    x=df['epoch'].astype(str) + '-' + df['current_batch'].astype(str),
    y=df['avg_acc'],
    mode='lines+markers',
    name='Average Accuracy'
  ),
  row=2,
  col=1
)

# Batch Accuracy
fig.add_trace(
  go.Scatter(
    x=df['epoch'].astype(str) + '-' + df['current_batch'].astype(str),
    y=df['batch_acc'],
    mode='lines+markers',
    name='Batch Accuracy'
  ),
  row=3,
  col=1
)

fig.update_xaxes(title_text="Epoch-Batch", row=3, col=1)

fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="Average Accuracy (%)", row=2, col=1)
fig.update_yaxes(title_text="Batch Accuracy (%)", row=3, col=1)

fig.update_layout(height=1200, showlegend=False, title_text="Training Metrics over Epoch-Batches")

fig.show()
