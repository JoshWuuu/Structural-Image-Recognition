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


# Do a hyperparameter search using a genetic algorithm
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
    with open(results_filename, 'r') as f:
        results = pd.read_csv(f, index_col = 0)
    print(results.columns)
    start = len(results.index)
    for i in range(start, start + 100):
        # INITIALIZE HYPERPARAMS
        # Picking an example assignment
        val_acc = results['Val Accuracy']; val_acc = val_acc.to_numpy(copy=True)
        val_acc -= np.min(val_acc); val_acc /= 0.005; val_acc = np.power(val_acc, 2); val_acc += 0.1; val_acc /= np.sum(val_acc)
        selection = np.random.choice(len(val_acc), p=val_acc)
        selected_example = results.iloc[selection].to_numpy()
        example_frozenlayers = selected_example[0]
        example_initiallr = selected_example[1]
        example_lrdecay = selected_example[2]
        # Choose parameters based on example parameters
        frozen_layers = min(176, int(example_frozenlayers+np.random.normal(0, 5)))
        example_exp = -np.log10(example_initiallr); exp = example_exp+np.random.normal(0, 0.2)
        initial_lr = 10 ** (-exp)
        lr_decay = min(1, example_lrdecay+np.random.normal(0, 0.05))
        max_epochs = 7
        # Identify 5th best performing model
        best_models = results['Val Accuracy']
        best_models = best_models.copy()
        best_models = best_models.sort_values(ascending=False)
        fifth_place = best_models.index[4]
        fifth_performance = best_models[fifth_place]

        print(f"TRAINING MODEL WITH HYPERPARAMS -- frozen_layers: {frozen_layers}, initial_lr: {initial_lr}, lr_decay: {lr_decay}, max_epochs: {max_epochs}")
        # Create Model
        model = create_model(frozen_layers, input_shape, num_class)
        # Fit Model
        model, history = compile_and_fit(model, generator, train_x, train_y, val_x, val_y, max_epochs)

        # Save everything useful
        train_perf = model.evaluate(train_x, train_y, verbose=1)
        val_perf = model.evaluate(val_x, val_y, verbose=1)
        hyper_param_arr = [frozen_layers, initial_lr, lr_decay, max_epochs]
        results.loc[i] = hyper_param_arr + train_perf + val_perf

        save_history(history, history_dir + str(i) + ".csv")
        
        val_acc = val_perf[1]
        if val_acc > fifth_performance:
            print("better than fifth")
            # save the new model
            save_model(model, model_dir + str(i))
            # delete the old model
            shutil.rmtree(model_dir + str(fifth_place))

        with open(results_filename, 'w') as f:
            results.to_csv(f)




if __name__ == '__main__':
    main()
