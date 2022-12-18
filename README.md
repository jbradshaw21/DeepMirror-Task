# DeepMirror-Task
Time Allocation:
- Choose datasets, setup env, familiarise with new data type 'smiles string', convert to useful data type (45min)
- Build pipeline, bugfixing (3hr)
- Fit functionality for GCN (1hr)
- Collecting ideas + write-up (30min)

Status:
- Working pipeline for graph-based DL
- Faulty GCN implementation
- Background reading required to advance with Transformers/Tree models...

Challenges:
- Choose good datasets for each task
	- Select datasets with the most citations on Google Scholar
	- Exception: AstraZeneca datasets do not appear to published on Scholar - as they are a trusted source,
	I give the datasets a bye
	- Idea: implement baseline models on two datasets for each of the ADME properties (priority order: Astrazeneca -> most cited -> ... -> least cited)
- First time working with molecular modelling data
	- I found a package that converts 'smiles string' to graph format
- How to deal with neural net regression rather than classification?
	- I designated a single output class and traded the CrossEntropyLoss for an L1Loss metric
- How to deal with no node features?
	- I first initialised a placeholder vector with all ones
	- Then, I decided to implement the node 'degrees' as node features after reading it is useful to weight
	nodes with fewer degrees as more important (though, I suspect this is not an appropraite implementation)
- How to deal with node logits?
	- Attempted to convert node regressors to graph regressors by averaging the predictions of the nodes for
	each graph

Questions:
- Are there any factors I did not consider/information I am unaware of for choosing the best datasets?
- What is the standard approach for converting 'smiles string' into a graphical data type?
	- In my approach: 'Atom "[...]" contains stereochemical information that will be discarded';
	  is this a problem?
- How to setup neural net for regression rather than classificaiton?
- Can I train a GCN without node features? If not, what is a suitable placeholder?
- What is the appropriate means for converting node logits to graph logits?
- Would you take a DL approach for both the Transformers and Classical Tree models?


Ideas:
- Planned approach: optimise hyperparams (probably just dropout/batch_size + appropriate stopping criteria)
  over validation set, then produce result (MAELoss) on test set with this hyperparam setting
- Provide MAELoss metric for supervised-GNN, Transformers, Classical Tree Model over two 'best' datasets
- Regarding Transformers: I have not studied NLP before. 
	- I am confident that I could produce an implementation given a few days to do background reading
	- ... however, given the time frame I can't justify an attempt
	- I have heard of graph transformers - would you implement something like this? Or implement typical NLP
	on the smiles string for the small molecule?
- Regarding Classical Tree Model: I could have produced e.g., RandomForestRegressor with sklearn, but unsure
  how to approach this task using the provided data-types
	- I am aware that Tree models can take graph-based input, however, I have not coded this before
	- Andrea talked about Monte Carlo Tree Search as an appropriate means of performing prediction tasks with
	molecules (good explainability)
	- Similar to transformers, new method for me
	- ... given the time frame I can't justify an attempt

Next Steps:
- Spend a few days background reading on NLP and Transformers models
- Attempt a PyTorch implementation, ideally using graph-based modelling and a GATConv layer (since I have a little
  experience with this)
- Look into appropriate Classical Tree Models after gaining some more insight from meeting/completing Transformers
  implementation...
