import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import colorsys
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def definecolor(h, s, v):
  clr=colorsys.hsv_to_rgb(h, s, v)
  print(clr)
  colorstr="rgba({}, {}, {}, 0.8)".format(clr[0]*256, clr[1]*256, clr[2]*256)
  return colorstr

def convertstring(s):
  if(s=="\n"):
    result=r"\n"
  else:
    result=s
  return result
    



dir='/content/drive/MyDrive/Colab Notebooks/entropix_stats'

ent_color=definecolor(0.5,1,0.6)
varent_color=definecolor(0,1,0.7)


# ent_color='rgba(135, 206, 250, 0.5)'

ent_marker=dict(
            color=ent_color,
            line=dict(
                color=ent_color,
            )
        )
varent_marker=dict(
            color=varent_color,
            line=dict(
                color=ent_color,
            )
        )

import os
files = os.listdir(dir)
json_files = [i for i in files if i.endswith('.json') == True]

filepath=dir+r"/"+json_files[0]

print(filepath)

with open(filepath) as f:
  d=json.load(f)

ents=d['ent']
varents=d['varent']
tokens=d['token']

max_value=max(max(ents), max(varents))

print("------------")
for i in range(len(tokens)):
  print(tokens[i], end="")

length=50
N=len(ents)//length+1

ratio=0.6

row_heights = [ratio/(N*2) if i%2==0 else (1-ratio)/(N*2) for i in range(N*2) ]
print(row_heights)

fig = make_subplots(
  rows=N*2, cols=1,
  shared_xaxes=True,
  row_heights=row_heights
  # shared_yaxes=True
  )

for i in range(N):
  fig.update_layout(
  hoverlabel=dict(
      font_color="white",
  ))
  if(len(ents)>length*(i+1)):
    terminalIndex=length*(i+1)
  if(len(ents)<=length*(i+1)):
    terminalIndex=len(ents)
  ents100=ents[length*(i):terminalIndex]
  varents100=varents[length*(i):terminalIndex]
  tokens100=tokens[length*(i):terminalIndex]
  index=[i for i in range(length*i, terminalIndex)]
  xvalue=[i for i in range(terminalIndex-length*i)]
  tokenheight=[i%4 for i in range(length*i, terminalIndex)]


  entdf=pd.DataFrame({'x': xvalue, 'y': ents100, 'token': tokens100, 'index':index})
  textdf=pd.DataFrame({'x':xvalue, 'y': tokenheight})
  varentdf=pd.DataFrame({'x': xvalue, 'y': varents100, 'token': tokens100, 'index':index})



  fig.add_trace(go.Scatter(
      x=entdf['x'], 
      y=entdf['y'],
      customdata=entdf[['token', 'index']],  # customdata に属性を設定
      hovertemplate='index=%{customdata[0]}<br>y=%{y}<br>token=%{customdata[0]}<extra></extra>',
  mode='lines+markers',
  # marker_color=ent_color,
  marker=ent_marker,
  name='entropy' ),
  row=i*2+1, col=1
  )

  fig.add_trace(go.Scatter(
      x=varentdf['x'], 
      y=varentdf['y'],
      customdata=varentdf[['token', 'index']],  # customdata に属性を設定
      hovertemplate='index=%{customdata[1]}<br>y=%{y}<br>token=%{customdata[0]}<extra></extra>',
  mode='lines+markers',
  marker=varent_marker,
  name='varentropy'),
  row=i*2+1, col=1
  )

  fig.add_trace(go.Scatter(
    x=xvalue,
    y=tokenheight,
    mode="markers+text",
    name="token",
    text=[convertstring(s) for s in tokens100],
    textposition="top center",
  ),
  row=i*2+2, col=1
  )
  fig.update_yaxes(
    range=[0, max_value+0.25],
    row=i*2+1, col=1
  ) 
  fig.update_yaxes(
    range=[-0.2, 4.2],
    row=i*2+2, col=1
  ) 




fig.update_layout(
  width=24*length, height=350*N,
)
fig.show()

# print(d['prompt'])


# fig.add_trace(go.Scatter(x=xvalue, y=ents,
#                     mode='lines',
#                     name='entropy'))
# fig.add_trace(go.Scatter(x=xvalue, y=varents,
#                     mode='lines',
#                     name='varentropy'))

# fig = px.line(entdf, x='x', y='y', hover_data=['Attribute'], markers=True)
# fig = px.line(varentdf, x='x', y='y', hover_data=['Attribute'], markers=True)


# plt.figure(figsize=(10, 1))


# for i in range(len(ents)):
#   x=i
#   y=ents[i]
#   hue=(np.clip(varents[i], 0, 2.5))/2.5*180+180
#   color=colorsys.hsv_to_rgb(hue, 100, 50)
#   ax.text(
#     x,y,tokens[i], 
#     horizontalalignment='center',
#     verticalalignment='center',
#   )

# plt.plot(xvalue, ents, color='blue')
# plt.plot(xvalue, varents, color='red')

# plt.show()

