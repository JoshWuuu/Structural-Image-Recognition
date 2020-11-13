import numpy as np
import tensorflow as tf
from model import create_model
from util import *
import shutil


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

    columns = ['Frozen Layers', 'Initial LR', 'LR Decay', 'Max Epochs', 'Batch Size', 'Train Loss', 'Train Accuracy', 'Train AUC', 'Val Loss', 'Val Accuracy', 'Val AUC']
    #with open(results_filename, 'r') as f:
    #    results = pd.read_csv(f, index_col = 0)
    results = pd.DataFrame(columns = columns)
    print(results.columns)
    #start = len(results.index)
    #count = 0
    for i in range(0, 20):
        # count += 1
        # INITIALIZE HYPERPARAMS
        # Model has 177 layers
        # frozen_layers = np.random.randint(100, 176)
        # frozen_layers = np.random.randint(130, 150)
        exp = np.random.uniform(1, 3)
        initial_lr = 10 ** (- exp)
        # initial_lr = 0.001
        lr_decay = np.random.uniform(0.7, 1) 
        # lr_decay = np.random.uniform(0.7, 0.8)
        max_epochs = 7
        # frozen_layers = 115 + count 
        frozen_layers = 130
        batch_size = 64;

        # Identify 5th best performing model
       # best_models = results['Val Accuracy']
        #best_models = best_models.copy()
        #best_models = best_models.sort_values(ascending=False)
        #fifth_place = best_models.index[4]
       # fifth_performance = best_models[fifth_place]

        print(f"TRAINING MODEL WITH HYPERPARAMS -- frozen_layers: {frozen_layers}, initial_lr: {initial_lr}, lr_decay: {lr_decay}, max_epochs: {max_epochs}")
        # Create Model
        model = create_model(frozen_layers, input_shape, num_class)
        # Fit Model
        model, history = compile_and_fit(model, generator, train_x, train_y, val_x, 
                val_y, max_epochs, batch_size = batch_size, initial_lr = initial_lr, lr_decay_rate=lr_decay)

        # Save everything usefuli
        train_perf = model.evaluate(train_x, train_y, verbose=1)
        val_perf = model.evaluate(val_x, val_y, verbose=1)
        hyper_param_arr = [frozen_layers, initial_lr, lr_decay, max_epochs, batch_size]
        results.loc[i] = hyper_param_arr + train_perf + val_perf

        save_history(history, history_dir + str(i) + ".csv")

        #val_acc = val_perf[1]
        # if val_acc > fifth_performance:
        #    # save the new model
        #    save_model(model, model_dir + str(i))
            # delete the old model
        #    shutil.rmtree(model_dir + str(fifth_place))

        with open(results_filename, 'w') as f:
            results.to_csv(f)




if __name__ == '__main__':
    main()
