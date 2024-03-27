import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_FILE = '2024-03-26_16-51-32validation_data.csv'

df = pd.read_csv(DATA_FILE)
fig = make_subplots(
  rows=2,
  cols=1, 
  subplot_titles=(
    'Loss versus Epoch',
    'Average Accuracy (with CI) versus Epoch'
  ),
  vertical_spacing=0.2
)

# Loss
fig.add_trace(
  go.Scatter(
    x=df['epoch'],
    y=df['loss'],
    mode='lines+markers',
    name='Loss',
    showlegend=False
  ),
  row=1,
  col=1
)

ci_upper = df['avg_acc'] + df['ci_95']
ci_lower = df['avg_acc'] - df['ci_95']

# Average Accuracy with CI boundaries
fig.add_trace(
  go.Scatter(
    x=df['epoch'],
    y=df['avg_acc'],
    mode='lines+markers',
    name='Average Accuracy',
    showlegend=False
  ),
  row=2,
  col=1
)
fig.add_trace(
  go.Scatter(
    x=df['epoch'],
    y=ci_upper,
    mode='lines',
    name='Upper CI',
    marker=dict(color="#444"),
    showlegend=False
  ),
  row=2,
  col=1
)
fig.add_trace(
  go.Scatter(
    x=df['epoch'],
    y=ci_lower,
    mode='lines',
    name='Lower CI',
    marker=dict(color="#444"),
    fill='tonexty',
    fillcolor='rgba(68, 68, 68, 0.3)',
    showlegend=False
  ),
  row=2,
  col=1
)

fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=1)

fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="Average Accuracy (%)", row=2, col=1)

fig.update_layout(height=700, title_text="Training Metrics over Epoch")

fig.show()
