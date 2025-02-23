# GNN_Testing

## To install the required packages 
* create a new venv in this directory
  * run `python -m venv <NAME>`
* For powershell run:
  * `.\<NAME>\Scripts\Activate.ps1`
* For commandline run:
  * `.\<NAME>\Scripts\Activate.bat`
* run `pip install -r requirments.txt`

## To use tensorboard
* open a new terminal
* activate the venv
* run `tensorboard --logdir=runs`
* open the link in a browser (by default http://localhost:6006/)
* when running the training loop it should show you graphs and stuff here