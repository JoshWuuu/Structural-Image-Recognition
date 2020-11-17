import os, datetime
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np

def download_data():
    os.system(
        'wget --no-check-certificate https://apps.peer.berkeley.edu/phichallenge/dataset/task1_scene_level.zip')
    os.system('unzip task1_scene_level.zip')

# Helper function to load data and create training validation split
def get_data(split):
    base_file_path = 'task1/'
    train_x_path = base_file_path + 'task1_X_train.npy'
    train_y_path = base_file_path + 'task1_y_train.npy'
    raw_train_x = np.load(train_x_path)
    np.random.seed(1)
    np.random.shuffle(raw_train_x)

    raw_train_y = np.load(train_y_path)
    np.random.seed(1)
    np.random.shuffle(raw_train_y)

    n = raw_train_x.shape[0]
    train_x = raw_train_x[0:int(n * split)]
    val_x = raw_train_x[int(n * split):]
    train_y = raw_train_y[0:int(n * split)]
    val_y = raw_train_y[int(n * split):]
    return train_x, train_y, val_x, val_y

# Save model to disk
def save_model(model, filepath):
    model.save(filepath)

# Save history to disk
def save_history(history, filename):
    df = pd.DataFrame(history.history)
    with open(filename, 'w') as f:
        df.to_csv(f)


# Evaluate model on validation data
def evaluate_model(model, model_name, train_x, train_y,
                   val_x, val_y, results_dict, verbose=1, ):
    train_perf = model.evaluate(train_x, train_y, verbose=1)
    val_perf = model.evaluate(val_x, val_y, verbose=1)
    return train_perf, val_perf


# Helper function to compile and fit the model with given hyperparameters
def compile_and_fit(model, generator, train_x_input, train_y_input,
                    val_x_input, val_y_input, MAX_EPOCHS,
                    decay_step=500, batch_size=64, initial_lr=1e-3, lr_decay_rate=0.9):
    # Use exponentially decaying learning rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_step,
        decay_rate=lr_decay_rate)

    # Compile with cross entropy, using adam optimizer and accuacy + AUC
    model.compile(loss=tf.losses.categorical_crossentropy,
                  optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()])

    callbacks = []

    # Fit model
    history = model.fit(
        train_x_input, train_y_input, batch_size = batch_size,
        epochs=MAX_EPOCHS,
        validation_data=(val_x_input, val_y_input),
        callbacks=callbacks)

    return model, history 
