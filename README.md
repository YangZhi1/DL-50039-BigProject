# DL-50039-BigProject
Deep Learning 50.039 Big Project. Given a graph dataset, predicts scam victims based on who the scammers are in the graph. 

Download the required libraries:
```pip install -r requirements.txt```

## Data:
Data is retrieved from [SNAP: Network datasets](https://snap.stanford.edu/data/ego-Facebook.html). It contains graph information for facebook datasets, with anonymized features. For this project, we specifically used the 3980 network.

## Project:
Refer to the files ```training.ipynb``` and ```deployment.ipynb```. 

We trained a Kipf Graph Convolutional Network (GCN) on the data. Potential scam victims are labelled as positive and people safe from scams are labelled as negative. 

As the dataset did not originally come with scammers and scam victims, we generated the labels by ourselves. The scammers were randomly chosen for different iterations and the scam victims were chosen based on any defining features (such as gender or languages spoken) and the hop distance from the scammer.

We also explored selecting scammers and scam victims completely randomly, but such labels caused the model to be unable to learn any useful information, resulting in lousy predictions.  

We used networkx and pyvis to visualize the graph: 

![alt tag](https://github.com/YangZhi1/DL-50039-BigProject/blob/main/resources/Network_html.png)



## To use GUI:
Run ```python gui.py``` in command line

You should get a window as shown below:

![alt tag](https://github.com/YangZhi1/DL-50039-BigProject/blob/main/resources/GUI_start.png)

Select 2 node values that will be the scammer, then click show graph


#### Predicted graph

![alt tag](https://github.com/YangZhi1/DL-50039-BigProject/blob/main/resources/Generated_graph.png)
