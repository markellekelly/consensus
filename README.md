# consensus

- **dataset.py** contains the Dataset and Example classes, which store and manage the agent prediction data. The TestDataset class is used when all data is already available (and thus no actual querying needs to happen), e.g., for evaluating our model. An instance of Example (or TestExample) corresponds to a specific row of the dataset, and manages querying information, including keeping track of which experts have been queried.
- **consensus_model.py**
- **online.py** 
- **offline.py**
- **learn_underlying_normal.stan**
- **simulate_consensus.stan**