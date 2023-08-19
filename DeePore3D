from tqdm import tqdm
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler  


#Check Python and Tensorflow Version
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)


#Provided function to check and download dataset if required
def check_get(url, file_name):
    def download_callback(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            s = "\r%5.1f%% %*d MB / %d MB" % (percent, len(str(total_size)), read_so_far / (1024 * 1024), total_size / (1024 * 1024))
            sys.stderr.write(s)
            if read_so_far >= total_size:
                sys.stderr.write("\n")
        else:
            sys.stderr.write("Read %d\n" % (read_so_far,))

    if not os.path.isfile(file_name):
        ans = input('You dont have the file "' + file_name + '". Do you want to download it? (Y/N) ')
        if ans.lower() in ['y', 'yes']:
            print('Beginning file download. This might take several minutes.')
            urlretrieve(url, file_name, reporthook=download_callback)
    else:
        print('File "' + file_name + '" is detected on your machine.')


url = "https://zenodo.org/record/3820900/files/DeePore_Dataset.h5" 
file_name = "DeePore_Dataset.h5"
check_get(url, file_name)


#First We Need To Clean The Dataset and Apply Normalisation
def clean_and_prep(input_file, output_file, chunk_size, compression_level):
    
    def is_outlier(y):
        A = (
            int(np.isnan(y).any()) +
            int(np.isinf(y).any()) +
            int(y[1] > 120) +
            int(y[4] > 1.8) +
            int(y[0] < 1e-4) +
            int(y[2] < 1e-5) +
            int(y[14] > 0.7)
        )
        return A > 0

    with h5py.File(input_file, 'r') as file1:
        dataset_X = file1['X']
        dataset_Y = file1['Y']

        total_rows = dataset_X.shape[0]
        
        scaler_Y = MinMaxScaler()

        with h5py.File(output_file, 'w') as file2:
            cleaned_X_dset = None
            cleaned_Y_dset = None

            for start in tqdm(range(0, total_rows, chunk_size), desc='Cleaning Progress: '):
                end = min(start + chunk_size, total_rows)
                
                chunk_X = dataset_X[start:end][:]
                nan_mask_X = np.isnan(chunk_X).any(axis=(1, 2, 3, 4)) 
                inf_mask_X = np.isinf(chunk_X).any(axis=(1, 2, 3, 4))
                invalid_mask_X = nan_mask_X | inf_mask_X
                chunk_X = chunk_X[~invalid_mask_X]

                chunk_Y = dataset_Y[start:end][:]
                nan_mask_Y = np.isnan(chunk_Y).any(axis=(1, 2))
                inf_mask_Y = np.isinf(chunk_Y).any(axis=(1, 2))
                invalid_mask_Y = nan_mask_Y | inf_mask_Y
                chunk_Y = chunk_Y[~invalid_mask_Y]
                
                chunk_Y_reshaped = chunk_Y.reshape(-1, chunk_Y.shape[-1])

                for i in range(chunk_Y.shape[0]):
                    y = chunk_Y[i, :15]
                    if is_outlier(y):
                        y[:15] = np.log10(y[:15])

                    y_reshaped = y.reshape(1, -1)  
                    scaler_Y.partial_fit(y_reshaped)
                    y_normalized = scaler_Y.transform(y_reshaped)

                    if cleaned_Y_dset is None:
                        cleaned_Y_dset = file2.create_dataset('Y_cleaned', data=y_normalized, dtype='float64',
                                                              compression='gzip', compression_opts=compression_level,
                                                              maxshape=(None, y_normalized.shape[1]))
                    else:
                        cleaned_Y_dset.resize((cleaned_Y_dset.shape[0] + y_normalized.shape[0]), axis=0)
                        cleaned_Y_dset[-y_normalized.shape[0]:] = y_normalized

                if cleaned_X_dset is None:
                    cleaned_X_dset = file2.create_dataset('X_cleaned', data=chunk_X, dtype='bool',
                                                          compression='gzip', compression_opts=compression_level,
                                                          maxshape=(None, 256, 256, 256, 1))
                else:
                    cleaned_X_dset.resize((cleaned_X_dset.shape[0] + chunk_X.shape[0]), axis=0)
                    cleaned_X_dset[-chunk_X.shape[0]:] = chunk_X

            print("Dataset cleaning and preparation are complete.")

chunk_size = 100
compression_level = 1
deepore_file = 'C:/Users/Aymane/MSCPROJ/DeePore_Dataset.h5'   #Replace with your path
cleaned_file = 'C:/Users/Aymane/MSCPROJ/Cleaned_Dataset.h5'   #Replace with your path


#Data Generator Functions Used to Create Batches on the Fly During Training
#One is shuffled for training and validation. One is unshuffled for testing.
def data_generator(file_path, batch_size, subset_indices):

    while True:
        length = len(subset_indices)

        for start_idx in range(0, length, batch_size):
            excerpt = subset_indices[start_idx:start_idx + batch_size]
            X_batch = np.empty((len(excerpt), 256, 256, 256, 1), dtype=np.float32)
            y_batch = np.empty((len(excerpt), 10, 1), dtype=np.float32)

            for i, index in enumerate(excerpt):
                with h5py.File(file_path, 'r') as file:
                    data1 = file['X_cleaned'][index % length, ...]
                    data2 = file['Y_cleaned'][index % length, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
                    data2_reshaped = data2.reshape(-1, 1)
                
                X_batch[i, ...] = data1.astype('float32')
                y_batch[i, ...] = data2_reshaped.astype('float32')

            yield X_batch, y_batch
            


def shuffled_data_generator(file_path, batch_size, subset_indices):

    while True:
        length = len(subset_indices)
        np.random.shuffle(subset_indices)

        for start_idx in range(0, length, batch_size):
            excerpt = subset_indices[start_idx:start_idx + batch_size]
            X_batch = np.empty((len(excerpt), 256, 256, 256, 1), dtype=np.float32)
            y_batch = np.empty((len(excerpt), 10, 1), dtype=np.float32)

            for i, index in enumerate(excerpt):
                with h5py.File(file_path, 'r') as file:
                    data1 = file['X_cleaned'][index % length, ...]
                    data2 = file['Y_cleaned'][index % length, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
                    data2_reshaped = data2.reshape(-1, 1)
                
                X_batch[i, ...] = data1.astype('float32')
                y_batch[i, ...] = data2_reshaped.astype('float32')

            yield X_batch, y_batch

#Subsets Function
def get_subsets(total_samples, train_split, val_split, test_split):
    indices = np.arange(total_samples)

    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    return train_indices, val_indices, test_indices




#Function To Generate Training, Validation and Testing Indices
def get_subsets(total_samples, train_split, val_split, test_split):
    indices = np.arange(total_samples)

    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    return train_indices, val_indices, test_indices


#Indices Instantiation
train_split = 0.7
val_split = 0.1
test_split = 0.2

with h5py.File(cleaned_file, 'r') as f:
    total_samples = f['X_cleaned'].shape[0]
    train_indices, val_indices, test_indices = get_subsets(total_samples, train_split, val_split, test_split)



#Training, Validation and Testing Generator Split 
batch_size = 4

train_gen = shuffled_data_generator(cleaned_file, batch_size, train_indices)
val_gen = shuffled_data_generator(cleaned_file, batch_size, val_indices)
test_gen = data_generator(cleaned_file, batch_size, test_indices)


#Steps Per Epoch Function
def steps_per_epoch(cleaned_file, batch_size):
    
    total_samples = 0
    with h5py.File(cleaned_file, 'r') as f:
        total_samples = f['X_cleaned'].shape[0]

    steps_per_epoch = total_samples // batch_size

    return steps_per_epoch


#Steps Per Epoch For Training, Validation and Testing
steps_per_epoch_train = int(steps_per_epoch(cleaned_file, batch_size) * train_split)
steps_per_epoch_val = int(steps_per_epoch(cleaned_file, batch_size) * val_split)
steps_per_epoch_test = int(steps_per_epoch(cleaned_file, batch_size) * test_split)


#CNN Model 1
def DeePore3D1(input_shape, output_shape):

    input_spec = Input(input_shape)

    initialmaxpool = MaxPooling3D(pool_size=(2, 2, 2))(input_spec)
    
    conv1 = Conv3D(4, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(initialmaxpool)
    maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(8, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool1)
    maxpool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = Conv3D(16, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool2)
    maxpool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(32, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool3)
    maxpool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    flatten = tf.keras.layers.Flatten()(maxpool4)
    dense1 = tf.keras.layers.Dense(3, activation=tf.nn.relu)(flatten)
    dense2 = tf.keras.layers.Dense(3, activation=tf.nn.sigmoid)(dense1)

    output_spec = Dense(output_shape[0])(dense2)
    model = Model(inputs=[input_spec], outputs=[output_spec])
    optimiser = tf.keras.optimizers.RMSprop(1e-2)

    model.compile(optimizer=optimiser, loss='mse', metrics=['mse'])
    return model


#CNN Model 2
def DeePore3D2(input_shape, output_shape):

    input_spec = Input(input_shape)

    initialmaxpool = MaxPooling3D(pool_size=(2, 2, 2))(input_spec)
    
    conv1 = Conv3D(4, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(initialmaxpool)
    maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(8, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool1)
    maxpool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(16, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool2)
    maxpool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    flatten = tf.keras.layers.Flatten()(maxpool3)
    dense1 = tf.keras.layers.Dense(3, activation=tf.nn.relu)(flatten)
    dense2 = tf.keras.layers.Dense(3, activation=tf.nn.sigmoid)(dense1)

    output_spec = Dense(output_shape[0])(dense2)
    model = Model(inputs=[input_spec], outputs=[output_spec])
    optimiser = tf.keras.optimizers.RMSprop(1e-2)

    model.compile(optimizer=optimiser, loss='mse', metrics=['mse'])
    return model


#CNN Model 3
def DeePore3D3(input_shape, output_shape):

    input_spec = Input(input_shape)

    initialmaxpool = MaxPooling3D(pool_size=(2, 2, 2))(input_spec)
    
    conv1 = Conv3D(8, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(initialmaxpool)
    maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool1)
    maxpool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(24, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool2)

    conv4 = Conv3D(32, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(conv3)
    maxpool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    flatten = tf.keras.layers.Flatten()(maxpool4)
    
    dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)(flatten)
    dense2 = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)(dense1)

    output_spec = Dense(output_shape[0])(dense2)
    model = Model(inputs=[input_spec], outputs=[output_spec])
    optimiser = tf.keras.optimizers.RMSprop(1e-5)


#CNN Model 4
def DeePore3D4(input_shape, output_shape):

    input_spec = Input(input_shape)

    initialmaxpool = MaxPooling3D(pool_size=(2, 2, 2))(input_spec)
    
    conv1 = Conv3D(8, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(initialmaxpool)
    maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool1)
    maxpool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(24, kernel_size=(3, 3, 3), kernel_initializer='he_normal', padding='same')(maxpool2)
    maxpool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    flatten = tf.keras.layers.Flatten()(maxpool3)
    
    dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)(flatten)
    dense2 = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)(dense1)

    output_spec = tf.keras.layers.Dense(output_shape[0])(dense2)
    model = Model(inputs=[input_spec], outputs=[output_spec])
    optimiser = tf.keras.optimizers.RMSprop(1e-5)

    model.compile(optimizer=optimiser, loss='mse', metrics=['logcosh'])
    return model


#Cell to Instantiate Model/s
input_shape = (256, 256, 256, 1)
output_shape = (10, 1)
model = DeePore3D4(input_shape, output_shape)
model.summary()
    model.compile(optimizer=optimiser, loss='mse', metrics=['logcosh'])


#Training And Validation
class ProgressBar(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

epochs = 5

progress_bar = ProgressBar(epochs)

model_checkpoint = ModelCheckpoint('DeePore_Model_4_retrain.h5', save_best_only=False)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch_train,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=steps_per_epoch_val,
    callbacks=[progress_bar, model_checkpoint]
)

    return model



#Training and Validation Loss Graph
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()



#Testing Loop
trained_model = load_model('DeePore_Model_4_retrain.h5')

overall_true_values = []
overall_pred_values = []

with tf.device("cpu:0"):
    with tqdm(total=steps_per_epoch_test, desc='Testing Progress') as pbar:
        for step in range(steps_per_epoch_test):
            X_test_batch, y_test_batch = next(test_gen)
            y_pred_batch = trained_model.predict(X_test_batch)

            overall_true_values.append(y_test_batch)
            overall_pred_values.append(y_pred_batch)
            
            pbar.update(1)

    overall_true_values = np.concatenate(overall_true_values, axis=0)
    overall_pred_values = np.concatenate(overall_pred_values, axis=0)


#Scatter Plot Generation
for i in range(overall_true_values.shape[1]):
    plt.figure(figsize=(8, 6))
    plt.scatter(overall_true_values[:, i], overall_pred_values[:, i], s=10)
    plt.xlabel(f"True Values - Characteristic {i+1}")
    plt.ylabel(f"Predicted Values - Characteristic {i+1}")
    plt.title(f"Scatter Plot - Characteristic {i+1}")
    plt.grid(True)
    plt.savefig(f"scatter_plot_characteristic_{i+1}.png")
    plt.close()

print("Scatter plots saved.")



#Cell to Calculate R2 Scores
r2_scores = []
for i in range(overall_true_values.shape[1]):
    r2 = r2_score(overall_true_values[:, i], overall_pred_values[:, i])
    r2_scores.append(r2)
    print(f"R2 Score - Characteristic {i+1}: {r2:.4f}")

average_r2 = sum(r2_scores) / len(r2_scores)
print(f"Average R2 values: {average_r2:.4f}")

