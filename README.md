# consensus

- **dataset.py** contains the Dataset and Example classes, which store and manage the agent prediction data. The TestDataset class is used when all data is already available (and thus no actual querying needs to happen), e.g., for evaluating our model. An instance of Example (or TestExample) corresponds to a specific row of the dataset, and manages querying information, including keeping track of which experts have been queried.
- **consensus_model.py** contains the logic for getting a posterior consensus distribution, choosing experts to query, and making a final prediction.
- **run.py** runs our experiments in an online fashion, updating parameters and collecting results
- **update_parameters.stan** estimates the parameters for the underlying multivariate normal distribution
- **simulate_consensus.stan** gets the posterior consensus distribution given partially observed data
