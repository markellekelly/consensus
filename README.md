# consensus

- **data** contains all data files and our preprocessing scripts

- **update_parameters.stan** estimates the parameters for the underlying multivariate normal distribution
- **simulate_consensus.stan** gets the posterior consensus distribution given partially observed data
- **dataset.py** contains the Dataset and Example classes, which store and manage the agent prediction data. The TestDataset class is used when all data is already available (and thus no actual querying needs to happen), e.g., for evaluating our model. An instance of Example (or TestExample) corresponds to a specific row of the dataset, and manages querying information, including keeping track of which experts have been queried.
- **consensus_model.py** contains the logic for getting a posterior consensus distribution, choosing experts to query, and making a final prediction.
- **online_modeling.py** contains the code for running a full experiment, querying the consensus model for predictions and specifying when to update parameters. See documentation of `run_experiment_with_threshold()`.
- **run_experiments.py** runs our experiments, reading in our data files, setting hyperparameter values, etc. Multiple experiments can be run in parallel via `run_multiple.sh`.

