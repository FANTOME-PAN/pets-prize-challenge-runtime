# Model Name here (Track A: Financial Crime Prevention)
blabla

## Centralised Version
blabla

## Federated Version
Our FL solution has 3 phases. In Setup phase, clients generate key pairs, exchange public keys using the aggregator as relay, and generate shared secrets in 2 rounds. The Setup phase should be completed before Train phase or Test phase.

It worth noting that, to increase the system efficiency, our solution ustilise both federated fit and federated evaluate processes to facilitate communications among parties. Both fit and evaluate functions only serve as triggers for other functions, and their names do not indicate their functionality. Besides, as both fit and evaluate are used, one server round in flower is equivalent to two rounds in our implementation.

The main body of the solution is in `solution_federated.py`. `TrainSwiftClient` class, `TrainBankClient` class, `TestSwiftClient` class, and `TestBankClient` class inherit from `TrainClientTemplate` class in `fl_logic.py`. `TrainClientTemplate` class implement the core logic of a client class including the implementation of Setup phase (see `setup` func in the class) and a `__execute` function that can call corresponding functions, e.g., `stage0`, `stage1`, `stage2`, `setup_round1`, `setup_round2`, which are overrided in subclasses. Please note that `TrainClientTemplate` classs is also the superclass of `TestSwiftClient` class and `TestBankClient` class. `fl_xgboost_utils.py` comprises `fit_swift` func that trains XGBoost on SWIFT transactions dataset and returns probabilities of all samples in the train set, and `test_swift` func that returns probabilities of all samples in the test set. 

The implementation of strategies are `TrainStrategy` class and `TestStrategy` class.

## Setup phase
![image](https://user-images.githubusercontent.com/48020003/214913393-64ced52d-d406-443a-b983-0985d57c7fa0.png)

The figure above illustrate the workflow of Setup phase. ①② is the 1st round setup, ③ is a part of the 2nd round setup (empty replies from clients are omitted)

①②③④⑤⑥

## Train phase
![image](https://user-images.githubusercontent.com/48020003/214913570-d019f0c4-6e04-4942-8846-fcbd639b7f46.png)

Each training iteration is consist of 3 stages, i.e., 3 rounds. ①②, ③④, and ⑤⑥ are the stage 0, the stage 1, and the stage 2 respectively.

## Test phase
![image](https://user-images.githubusercontent.com/48020003/214913717-4707557d-f02d-4723-9c93-5d8c69a79bc1.png)

Test phase is similar to an iteration in Train phase. It does not need to compute gradients and thus it ends with the SWIFT client writing predictions to a given path at stage 2.
In the figure above, ①② is the stage 0. In stage 1, the aggregator will make the final predictions and send them to the SWIFT client. Then the SWIFT client will write predictions to a given path.

