
# Communication-Efficient Learning of Deep Netowkrs from Decentralized Data

## Problem

Modern devices have a wealth of data that can allow great learning for models. This can provide a great and personal user-experience. 

While this data is great, there is a big issue of user privacy if this data leaves their devices. We have seen that anonymizing the data is not enough to maintain user privacy from the [Netflix Prize Competition Dataset](https://arxiv.org/PS_cache/cs/pdf/0610/0610105v2.pdf "Robust De-anonymization of Large Datasets (How to Break Anonymity of the Netflix Prize Dataset)"). 

Apart from the privacy issue, this data is often very large in quantity which will require a large amount of space in the data center and then training the model. 

## Algorithm/Solution

This paper advocates an alternative to the problems above by a new approach where the training data is left distributed on the devices. The global model rlearns by aggregating locally-computed updates. 

The paper labels this decentralized approach as **Federated Learning**. The name *Federated Learning* comes from the fact that the learning task is solved by a loose federation of participating devices (referred as clients). These clients are coordinated by a central server.

With this decentralized approach, this paper released a new algorithm labelled as **FederatedAveraging** which we will talk about more later.

## Primary Contributions

1. The identification of problem of training on decentralized data from mobile devices. 
   
2. Selection of a straightforward and practival algorithm that can be applied to this setting.
3. Extensive empirical evaluation of the proposed approach.

