# Mini Deep Learning Framework
EPFL EE-559 Deep Learning: Second Project, Spring 2021
Author: Virginia Bordignon

## Requirements
- Python 3.8.3
- PyTorch 1.7.1


## Structure

- **code/**: Path to all the source code
	- **figs/**: folder where figs are saved.
	- **stats/**: folder where performance indices are saved.
	- **activation.py**: Contains activation modules. 	
	- **generate_data.py**: Loads and preprocess the dataset. 
	- **models.py**: Contains different CNN architectures.
	- **initialization**: Contains initialization functions. 
	- **loss.py**: Contains loss modules. 	
	- **modules.py**: Contains main modules.
	- **optimizer**: Contains optimizer modules. 	 	
	- **plot_figs.py**: Contains the code used for generating figures for the report. 
	- **test.py**: Contains the main code used for training and evaluating two approaches. 
	- **train_eval.py**: Contains the code used for training and validating models. 
	- **training_parameters.py**: Contains the global parameters. 

- **README.md**

## How to run test.py
To train and evaluate the module using the MSE cost and the Cross Entropy cost with the SGD optimizer:
```
$ cd code && python test.py
```

