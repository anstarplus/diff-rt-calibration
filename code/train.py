import os
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)

import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('../../sionna')
import sionna
from utils import *
from neural_materials import NeuralMaterials
import datetime # For logging
# import yaml
import json

training_name = "neural_materials"
# dataset_name = '../data/traced_paths/dichasus-dc01'
dataset_name = '/scratch/network/za1320/dataset/dichasus-dc01'
dataset_filename = os.path.join(dataset_name + '.tfrecords')
params_filename = os.path.join(dataset_name + '.json')

# Configure training parameters and step
batch_size = 4
learning_rate = 1e-3
num_iterations = 1000
delta = 0.999 # Parameter for exponential moving average

# Size of validation set size
# The validation set is used for early stopping, to ensure
# training does not overfit.
validation_set_size = 100
# We don't use the test set here, but need is size for splitting
test_set_size = 4900
# Sizes of the positional encoding to evaluate
position_encoding_size = 10
# Sizes of the training set to evaluate
training_set_size = 5000

with open(params_filename, 'r') as openfile:
    params = json.load(openfile)

# Scene
scene_name = params['scene_name']
# Size of the dataset
dataset_size = params['traced_paths_dataset_size']

num_subcarriers = 1024
bandwidth = 50e6
frequencies = subcarrier_frequencies(num_subcarriers, bandwidth/num_subcarriers)

# Load the TF records as a dataset
dataset = tf.data.TFRecordDataset([dataset_filename]).map(deserialize_paths_as_tensor_dicts)
# Split the dataset
# We don't use the test set
training_set, validation_set, _ = split_dataset(dataset, dataset_size, training_set_size, validation_set_size, test_set_size)

def train():

    # Training set
    training_set_iter = iter(training_set.shuffle(16, seed=42).batch(batch_size).repeat(-1))

    # Validation set
    validation_set_iter = iter(validation_set.batch(batch_size).repeat(-1))
    num_validation_iter = validation_set_size // batch_size-1

    # Load the scene
    scene = init_scene(scene_name, use_tx_array=True)

    # Place the transmitters
    place_transmitter_arrays(scene, [1,2])

    # Instantitate receivers
    instantiate_receivers(scene, batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    scaling_factor = tf.Variable(6e-9, dtype=tf.float32, trainable=True)

    # Setting up tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('../tb_logs/', training_name, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # Checkpoint
    weights_filename = os.path.join('../checkpoints/', training_name)

    scene.radio_material_callable = NeuralMaterials(scene, pos_encoding_size=position_encoding_size, learn_scattering=True)

    @tf.function
    def training_step(rx_pos, h_meas, traced_paths):

        # Placer receiver
        set_receiver_positions(scene, rx_pos)

        # Build traced paths
        traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

        with tf.GradientTape() as tape:
            # Compute paths fields
            paths = scene.compute_fields(*traced_paths,
                                         scat_random_phases=False,
                                         check_scene=False)

            a, tau = paths.cir(scattering=True) # Disable scattering

            # Compute channel frequency response
            h_rt = cir_to_ofdm_channel(frequencies, a, tau)

            # Remove useless dimensions
            h_rt = tf.squeeze(h_rt, axis=[0,2,5])

            # Normalize h to make sure that power is independent of the number of subacrriers
            h_rt /= tf.complex(tf.sqrt(tf.cast(num_subcarriers, tf.float32)), 0.)

            # Compute scaling factor
            scaling_factor.assign(delta*scaling_factor + (1-delta)*mse_power_scaling_factor(h_rt, h_meas))

            # Scale measurements
            h_meas *= tf.complex(tf.sqrt(scaling_factor), 0.)

            # Compute losses
            h_rt = sionna.utils.flatten_dims(h_rt, 3, 0)
            h_meas = sionna.utils.flatten_dims(h_meas, 3, 0)
            # Compute the average of a power and delay spread loss
            loss_ds = delay_spread_loss(h_rt, h_meas)
            loss_pow = power_loss(h_rt, h_meas)
            loss_ds_pow = loss_ds + loss_pow

        # Use loss_ds_pow for training
        grads = tape.gradient(loss_ds_pow, tape.watched_variables(), unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(grads, tape.watched_variables()))

        return loss_ds_pow, loss_ds, loss_pow, scaling_factor

    @tf.function
    def evaluation_step(rx_pos, h_meas, traced_paths, scaling_factor):

        # Placer receiver
        set_receiver_positions(scene, rx_pos)

        # Build traced paths
        traced_paths = tensor_dicts_to_traced_paths(scene, traced_paths)

        paths = scene.compute_fields(*traced_paths,
                                     scat_random_phases=False,
                                     check_scene=False)

        a, tau = paths.cir(scattering=False) # Disable scattering

        # Compute channel frequency response
        h_rt = cir_to_ofdm_channel(frequencies, a, tau)

        # Remove useless dimensions
        h_rt = tf.squeeze(h_rt, axis=[0,2,5])

        # Normalize h to make sure that power is independent of the number of subacrriers
        h_rt /= tf.complex(tf.sqrt(tf.cast(num_subcarriers, tf.float32)), 0.)

        # Scale measurements
        h_meas *= tf.complex(tf.sqrt(scaling_factor), 0.)

        # Compute losses
        h_rt = sionna.utils.flatten_dims(h_rt, 3, 0)
        h_meas = sionna.utils.flatten_dims(h_meas, 3, 0)
        # Compute the average of a power and delay spread loss
        loss_ds = delay_spread_loss(h_rt, h_meas)
        loss_pow = power_loss(h_rt, h_meas)

        return loss_ds, loss_pow

    for step in tqdm(range(num_iterations)):

        # Next set of traced paths
        next_item = next(training_set_iter, None)

        # Retreive the receiver position separately
        rx_pos, h_meas, traced_paths = next_item[0], next_item[1], next_item[2:]
        # Skip iteration if does not match the batch size
        if rx_pos.shape[0] != batch_size:
            continue

        # Batchify
        traced_paths = batchify(traced_paths)

        # Train
        tr_quantities = training_step(rx_pos, h_meas, traced_paths)
        loss_ds_pow, loss_ds, loss_pow, scaling_factor = tr_quantities

        # Logging
        if (step % 10) == 0:
            with train_summary_writer.as_default():
                # Log in TB
                tf.summary.scalar('loss_ds_pow_training', loss_ds_pow.numpy(), step=step)
                tf.summary.scalar('loss_ds_training', loss_ds.numpy(), step=step)
                tf.summary.scalar('loss_pow_training', loss_pow.numpy(), step=step)
                tf.summary.scalar('scaling_factor', scaling_factor.numpy(), step=step)
                # Save model
                save_model(scene.radio_material_callable, weights_filename, scaling_factor=scaling_factor.numpy())

        # Evaluate periodically on the evaluation set
        if ((step+1) % 1000) == 0:
            eval_loss_ds = 0.0
            eval_loss_pow = 0.0
            for _ in range(num_validation_iter):
                # Next set of traced paths
                # print('validate')
                next_item = next(validation_set_iter, None)
                # Retreive the receiver position separately
                rx_pos, h_meas, traced_paths = next_item[0], next_item[1], next_item[2:]
                # Skip iteration if does not match the batch size
                if rx_pos.shape[0] != batch_size:
                    continue

                # Batchify
                traced_paths = batchify(traced_paths)

                # Train
                eval_quantities = evaluation_step(rx_pos, h_meas, traced_paths, scaling_factor)
                loss_ds, loss_pow = eval_quantities
                eval_loss_ds += loss_ds
                eval_loss_pow += loss_pow
            eval_loss_ds /= float(num_validation_iter)
            eval_loss_pow /= float(num_validation_iter)
            # Log in TB
            with train_summary_writer.as_default():
                tf.summary.scalar('loss_ds_evaluation', eval_loss_ds, step=step)
                tf.summary.scalar('loss_pow_evaluation', eval_loss_pow, step=step)


    # Save model
    save_model(scene.radio_material_callable, weights_filename, scaling_factor=scaling_factor.numpy())
    print("saved model")

if __name__ == '__main__':
    train()
    print("Training completed.")