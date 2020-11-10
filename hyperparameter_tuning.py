import numpy as np
import tensorflow as tf
from model import create_model
from util import *


if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("No GPU Found!")

def mkdir_if_not_exist(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass

def main():
    train_x, train_y, val_x, val_y = get_data(0.9)
    train_x = train_x
    train_y = train_y
    input_shape = train_x.shape[1:]
    num_class = train_y.shape[1]
    generator = keras.preprocessing.image.ImageDataGenerator(
        data_format='channels_last'
    )

    history_dir = "history"
    model_dir = "models"
    mkdir_if_not_exist(history_dir)
    mkdir_if_not_exist(model_dir)
    history_dir += "/"
    model_dir += "/"

    results_filename = 'results.csv'

    columns = ['Frozen Layers', 'Initial LR', 'LR Decay', 'Max Epochs', 'Train Loss', 'Train Accuracy', 'Train AUC', 'Val Loss', 'Val Accuracy', 'Val AUC']
    results = pd.DataFrame(columns = columns)

    for i in range(30):
        # INITIALIZE HYPERPARAMS
        # Model has 177 layers
        frozen_layers = np.random.randint(100, 176)
        exp = np.random.uniform(1, 5)
        initial_lr = 10 ** (- exp)
        decay_rate_exp = np.random.uniform(1, 2.5)
        lr_decay = 1 - 10 ** (-decay_rate_exp)
        max_epochs = np.random.randint(10, 15)

        print(f"TRAINING MODEL WITH HYPERPARAMS -- frozen_layers: {frozen_layers}, initial_lr: {initial_lr}, lr_decay: {lr_decay}, max_epochs: {max_epochs}")
        # Create Model
        model = create_model(frozen_layers, input_shape, num_class)
        # Fit Model
        model, history = compile_and_fit(model, generator, train_x, train_y, val_x, val_y, max_epochs)

        # Save everything useful
        save_history(history, history_dir + str(i) + ".csv")
        save_model(model, model_dir + str(i))

        train_perf = model.evaluate(train_x, train_y, verbose=1)
        val_perf = model.evaluate(val_x, val_y, verbose=1)
        hyper_param_arr = [frozen_layers, initial_lr, lr_decay, max_epochs]
        results.loc[i] = hyper_param_arr + train_perf + val_perf

        with open(results_filename, 'w') as f:
            results.to_csv(f)




if __name__ == '__main__':
    main()
