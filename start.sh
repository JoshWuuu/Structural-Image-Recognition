conda activate ~/anaconda3/env/tensorflow2_latest_p37
python3 hyperparameter_tuning.py > out.txt 2> err.txt &
disown %1 
