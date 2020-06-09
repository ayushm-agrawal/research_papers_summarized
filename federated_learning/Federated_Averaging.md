
# Communication-Efficient Learning of Deep Netowkrs from Decentralized Data

## Problem

Modern devices have a wealth of data that can allow great learning for models. This can provide a great and personal user-experience. 

While this data is great, there is a big issue of user privacy if this data leaves their devices. We have seen that anonymizing the data is not enough to maintain user privacy from the [Netflix Prize Competition Dataset](https://arxiv.org/PS_cache/cs/pdf/0610/0610105v2.pdf "Robust De-anonymization of Large Datasets (How to Break Anonymity of the Netflix Prize Dataset)"). 

Apart from the privacy issue, this data is often very large in quantity which will require a large amount of space in the data center and then training the model. 

## Primary Contributions

1. The identification of problem of training on decentralized data from mobile devices. 
   
2. Selection of a straightforward and practival algorithm that can be applied to this setting.
3. Extensive empirical evaluation of the proposed approach.
   
## Algorithm/Solution

This paper advocates an alternative to the problems above by a new approach where the training data is left distributed on the devices. The global model rlearns by aggregating locally-computed updates. 

The paper labels this decentralized approach as **Federated Learning**. The name *Federated Learning* comes from the fact that the learning task is solved by a loose federation of participating devices (referred as clients). These clients are coordinated by a central server.

With this decentralized approach, this paper released a new algorithm labelled as **FederatedAveraging** which we will talk about more later.

### Federated Learning

The main idea of Federated Learning **(FL)** is the concept of keeping user data private. Even if the data in the data centers is "anonymized", it can still put the users at risk via join with other data. In FL, the information transmitted is the minimal update necessary to improve a particular model. The updates themselves don't contain extra information than what is actually required. The source of these updates can be transmitted of trusted third party without identifying the meta-data.

### Federated Optimization

Optimization problem in FL is referred to as Federated Optimization. The several key properties in Federated Optimization. For this paper, we will be focusing on two:

- **Non-IID** The training data on a given client is typically based on the udage of the mobile device by ap articular user, and hence any particular user's local dataset will note be representative of the population distribution.
- **Unbalanced** Some users will make much heavier use of the service or app than others, leading to varying amounts of local training data.

In the paper, the authors assume a synchronous update scheme that proceeds in the rounds of communication. It goes as follows

> There is a fixed set of K clients, each with a fixed local dataset. At the beginning of each round, a random fraction C of clients is selected, and the server sends the current global algorithm state to each of these clients. By only selecting a fraction of clients for efficiency, experiments show diminishing returns for adding more clients beyong a certain point. Each selected client then performs local computation based on teh global state and its local dataset, and sends and update to the server. **The server then applies these updates to tits global state, and the process repeats.**

The authors mentioned that although most of their focus is on non-convex neural network objectives, the algorithm is applicable to any finite-sum objective of the following form:

> $\min\limits_{w \in \R^d} f(w)\;$ where $f(w) = \frac{1}{n} \sum\nolimits_{i=1}^{n} f_i(w)$

In Machine Learning, $f_i(w)$ is the loss of the prediction on example $(x_i, y_i)$ made with model parameters $w$.

For FL, we assume that there are $K$ clients over with the data is partitioned. We have $P_k$ as the set of indexes of data points on client $k$, with $n_k = |P_k|)$. Now we can update the above objective as:

> $f(w)=\sum\nolimits_{k=1}^{K} \frac{n_k}{n} F_k(w)$ where $F_k(w) = \frac{1}{n_k} \sum_{i\in P_k} f_i(w)$ 

### The Federated Averaging Algorithm
