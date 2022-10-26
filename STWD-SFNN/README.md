STWD-SFNN: Sequential three-way decisions with a single hidden layer feedforward neural network

#### Abstract:

The three-way decisions strategy has been employed to construct network topology in a single hidden layer feedforward neural network (SFNN). However, this model has a general performance, and does not consider the process costs, since it has fixed threshold parameters. Inspired by the sequential three-way decisions (STWD), this paper proposes STWD with an SFNN (STWD-SFNN) to enhance the performance of networks on structured datasets. STWD-SFNN adopts multi-granularity levels to dynamically learn the number of hidden layer nodes from coarse to fine, and set the  sequential threshold parameters. Specifically, at the coarse granular level, STWD-SFNN handles easy-to-classify instances by applying strict threshold conditions, and with the increasing number of hidden layer nodes at the fine granular level, STWD-SFNN focuses more on disposing of the difficult-to-classify instances by applying loose threshold conditions, thereby realizing the classification of instances. Moreover, STWD-SFNN considers and reports the process cost produced from each granular level. The experimental results verify that STWD-SFNN has a more compact network on structured datasets than other SFNN models, and has better generalization performance than the competitive models.

---

#### Data:
[Data](https://github.com/wuc567/Machine-learning/blob/main/STWD-SFNN/data)  

#### Algorithms:
[STWD-SFNN and all competitive algorithms](https://github.com/wuc567/Machine-learning/tree/main/STWD-SFNN/algorithms)

#### How to run the codes of algorithms?

Taking STWD-SFNN algorithm as an example. 
Step 1 : Run the two subfiles run_stwd.m and run_twd.m of the Run-Paras folder to save the random number required by the STWD-SFNN model. 
Step 2 : Run the subfile Run_STWDSFNNAlgo_10folds_all.m of the STWD-SFNN folder, thus obtaining the experimental results of STWD-SFNN under the 10-fold cross-validation under a relatively fixed random number. We can view the 10-fold experimental value through the Result_matrix_all  variable.
