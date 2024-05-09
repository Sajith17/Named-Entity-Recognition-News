# Named-Entity-Recognition-News

## Overview

**Named-entity recognition** is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

I utilized **Tranformer Encoder** architecture and **Glove pretrained embeddings** for token embeddings. The whole model is built using **Tensorflow 2** and used **DVC** and **mlflow** for data versioning and experimenting the model. React JS is used to present inference results.

Some challenges I faced included **overfitting of token embeddings during training** which did not bring a decent semantic representation which resulted in model being less generalized to the real world data. To address this, I replaced the weights with **Glove 100d pre-trained embeddings** and froze them.
