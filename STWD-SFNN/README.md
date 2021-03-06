STWD-SFNN: Sequential three-way decisions with a single hidden layer feedforward neural network

#### Abstract:

The three-way decisions strategy was employed to construct neural network topology in a single hidden layer feedforward neural network (SFNN). However, this method has a long training time, and does not consider process costs, since it has fixed threshold parameters. Inspired by the sequential three-way decisions (STWD), this paper proposes STWD with an SFNN (STWD-SFNN) to enhance the performance of networks on structured datasets. STWD-SFNN adopts multi-granularity levels to dynamically learn the number of hidden layer nodes from coarse to fine, and to set the dynamic sequential threshold parameters. More specifically, in the coarse granular level, STWD-SFNN handles easy-to-classify instances by applying strict threshold conditions; with the increasing number of hidden layer nodes in the fine granular level, STWD-SFNN focuses more on disposing of the difficult-to-classify instances by applying loose threshold conditions, thereby realizing the classification of instances. More importantly, STWD-SFNN considers and reports the process cost produced from each granular level. Experiments verify that STWD-SFNN has a more compact network structure and a shorter running time on structured datasets than other competitive SFNN models, and has better generalization performance than other classification models.

---

#### Data:
[Data](https://github.com/wuc567/Machine-learning/blob/main/STWD-SFNN/data)  

#### Algorithms:
[STWD-SFNN and all competitive algorithms](https://github.com/wuc567/Machine-learning/tree/main/STWD-SFNN/algorithms)
