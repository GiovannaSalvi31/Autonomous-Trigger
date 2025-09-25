import numpy as np
import random
import os
import matplotlib.pyplot as plt

import atlas_mpl_style as aplt

aplt.use_atlas_style()

from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Lambda, MultiHeadAttention, Add
from keras.models import Sequential, Model
import h5py
import hdf5plugin
import tensorflow as tf
from tensorflow import keras


def sort_obj0(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array

    return data_

def process_h5_file0(input_filename):
    
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 4  
        

        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt
        

        npvsGood_smr1_values = h5_file['PV_npvsGood_smr1'][:]
        Ht_values = h5_file['ht'][:]
        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values != 0  
        data_array = data_array[non_zero_mask]  
        
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        
        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array = sort_obj0(data_array)
        sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]
  
        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]


        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        return sorted_data_array, Ht_values



def sort_obj(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index,:,:]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index,:,:] = sorted_array


    data_ = np.transpose(data_, (0,2,1))

    return data_

def process_h5_file(input_filename):

    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events)
        n_jets = 8
        n_features = 3


        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events

        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)

        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt

        
        npvsGood_smr1_values = h5_file['PV_npvsGood_smr1'][:]
        Ht_values = h5_file['ht'][:]

        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values != 0
        data_array = data_array[non_zero_mask]

        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]

        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array = sort_obj(data_array)

        #sorted_data_array[:, 3, :] = npvsGood_smr1_values[:, np.newaxis]


        zero_pt_mask = sorted_data_array[:, 2, :] == 0  # Identify where pt == 0
        sorted_data_array[:, 0, :][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, 1, :][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        npvsGood_smr1_values = npvsGood_smr1_values[1:]


        non_zero_ht_mask = Ht_values > 0
        # normalize the column
        #sorted_data_array[:, 2, :][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        sorted_data_array = sorted_data_array.reshape(len(sorted_data_array),-1)
        npvsGood_smr1_values = npvsGood_smr1_values.reshape(len(npvsGood_smr1_values),-1)
        sorted_data_array = np.hstack((sorted_data_array, npvsGood_smr1_values))



        return sorted_data_array, Ht_values


@keras.utils.register_keras_serializable()

def select_first_25(x):
    return x[:, :25]  # Takes only the first 25 elements (batch-wise)

@keras.utils.register_keras_serializable()

def repeat_last_element(x):

    last_value = tf.expand_dims(x[:, -1], axis=-1)  # Shape: (batch, 1)
    repeated_last = tf.repeat(last_value, 8, axis=-1)  # Shape: (batch, 8)
    first_24 = x[:, :-1]  # Shape: (batch, 24)
    return tf.concat([first_24, repeated_last], axis=-1)  # Shape: (batch, 32)


def build_autoencoder0(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) 
    
    # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder

def build_attention_autoencoder_jet(img_shape, code_size, num_heads=1):
    inp = Input(shape=img_shape)

    # Self-attention across jets — directly on raw features
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=img_shape[-1])(inp, inp)
    x = Add()([inp, attn_out])  # Residual connection

    # Flatten and encode
    x_flat = Flatten()(x)
    encoded = Dense(code_size)(x_flat)

    # Decode
    x_decoded = Dense(np.prod(img_shape))(encoded)
    x_out = Reshape(img_shape)(x_decoded)

    model = Model(inputs=inp, outputs=x_out)
    return model


def build_autoencoder2(img_shape, code_size):


    # The Encoder
    encoder = Sequential([
        InputLayer(img_shape),  # Input: (8,4)
        #Reshape((4, 8)),  # Step 1: Change (8,4) → (4,8)
        Flatten(),  # Step 2: Flatten (4,8) → (32,)
        Lambda(select_first_25),  # Step 3: Remove last 7 items (keep first 25)
        Dense(code_size)  # Latent space
    ])

    # The Decoder
    decoder = Sequential([
        InputLayer((code_size,)),  # Latent space input
        Dense(25),  # Expand back to 25 elements
        Lambda(repeat_last_element),  # Explicitly duplicate last element to reach 32
        Reshape((4,8))  # Reshape back to (4,8)
        #Lambda(lambda x: tf.transpose(x, perm=[1, 0]))  # Swap (4,8) to (8,4)

    ])

    return encoder, decoder


def build_autoencoder1(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    #encoder.add(Dense(hidden_size))
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    #decoder.add(Dense(hidden_size))
    decoder.add(Dense(np.prod(img_shape)))

    # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    #decoder.add(Reshape(img_shape))

    return encoder, decoder

def calculate_score(autoencoder_, image_):

    reconstructed_image_ = autoencoder_.predict(image_[None],verbose=0)[0]
    mse_loss = np.mean(np.square(image_ - reconstructed_image_))
    #mse_loss = masked_mse_loss(image_,reconstructed_image_)

    return mse_loss

@keras.utils.register_keras_serializable()

def masked_mse_loss(y_true, y_pred):


    # Create a boolean mask where valid jets have eta >= 0, phi >= 0, pt > 0
    valid_mask = tf.logical_and(
        tf.logical_and(y_true[:, :, 0] >= 0, y_true[:, :, 1] >= 0),
        y_true[:, :, 2] > 0
    )  # Shape: (batch, 8)

    # Expand mask to match (batch, 8, 4)
    valid_mask_expanded = tf.expand_dims(tf.cast(valid_mask, tf.float32), -1)  # Shape: (batch, 8, 1)

    # Compute squared error only for valid jets
    squared_error = tf.square(y_true - y_pred) * valid_mask_expanded

    # Normalize by the number of valid elements (avoid divide-by-zero)
    num_valid = tf.reduce_sum(valid_mask_expanded,axis=[1, 2]) + 1e-8
    loss = tf.reduce_sum(squared_error, axis=[1, 2]) / num_valid

    return tf.math.log1p(loss)



def mse_loss(y_true, y_pred):

    squared_error = tf.abs(y_true - y_pred)

    num = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

    loss = tf.reduce_sum(squared_error, axis=[1, 2]) / num

    return tf.math.log1p(loss)

def calculate_batch_score(autoencoder_, images_):
    reconstructed_images_ = autoencoder_.predict(images_, verbose=0)
    assert reconstructed_images_.shape == images_.shape, (
        f"Shape mismatch: reconstructed_images_ has shape {reconstructed_images_.shape}, "
        f"but images_ has shape {images_.shape}"
    )
    mse_losses = np.mean(np.square(images_ - reconstructed_images_), axis=(1,2))
    #mse_losses = masked_mse_loss(images_ , reconstructed_images_)

    return mse_losses

@keras.utils.register_keras_serializable()

def calculate_score_tensor(autoencoder_, images_):

    if len(images_.shape) == 2:  # If input is (8,4) instead of (batch, 8,4)
        images_ = images_[None]

    print(images_.shape)
    # Predict the reconstructed image
    reconstructed_images_ = autoencoder_.predict(images_, verbose=0)

    print(reconstructed_images_.shape)

    # Convert to TensorFlow tensors
    image_tensor = tf.convert_to_tensor(images_, dtype=tf.float32)
    reconstructed_tensor = tf.convert_to_tensor(reconstructed_images_, dtype=tf.float32)

    # Compute masked MSE loss (returns a Tensor)
    mse_losses = masked_mse_loss(image_tensor, reconstructed_tensor)
    #mse_losses = mse_loss(image_tensor, reconstructed_tensor)
    #mse_loss = tf.math.log1p(mse_loss)
    print(mse_losses.numpy().shape)


    return mse_losses.numpy()  # Convert Tensor to a NumPy scalar


mc_bkg_jets, mc_bkg_ht = process_h5_file0("../new_Trigger_dim4/new_Data/data_Run_2016_283876.h5")
X1 = mc_bkg_jets[::100]
Npv1 = mc_bkg_jets[::100,0,3]
Jets1 = mc_bkg_jets[::100, :, :-1]
HT1 = mc_bkg_ht[::100]



print(Jets1.shape, Npv1.shape)

#Ht1_all = mc_bkg_ht[::100]
N1 = len(X1)

#X2 = np.load('Data/Jet2_bkg_h5.npy',allow_pickle=True)
#N2 = len(X2)


mc_AA_jets, mc_AA_ht = process_h5_file0("../new_Trigger_dim4/new_Data/HToAATo4B.h5")
X_AA = mc_AA_jets
NpvAA = mc_AA_jets[::,0,3]
JetsAA = mc_AA_jets[::,:, :-1]
HTAA = mc_AA_ht[::100]

print('X_AA.shape',X_AA.shape)
N_tt = len(X_AA)
print(JetsAA.shape, NpvAA.shape)


mc_tt_jets, mc_tt_ht = process_h5_file0("../new_Trigger_dim4/new_Data/ttbar_2016_1.h5")
X_tt = mc_tt_jets
Npvtt = mc_tt_jets[::,0,3]
Jetstt = mc_tt_jets[::,:, :-1]
HTtt = mc_tt_ht[::100]

print('X_tt.shape',X_tt.shape)
N_tt = len(X_tt)
print(Jetstt.shape, Npvtt.shape)




X_train, X_test, Jets1_train, Jets1_test, Npv1_train, Npv1_test, HT_train, HT_test = train_test_split(X1, Jets1, Npv1, HT1, test_size=0.5, random_state=42)

#X_train, X_test, Ht_train, Ht_test = train_test_split(X1, Ht1_all, test_size=0.6, random_state=42)


# Same as (8,4), we neglect the number of instances from shape
IMG_SHAPE_X = X1.shape[1:]
IMG_SHAPE_Jets = Jets1.shape[1:]
IMG_SHAPE = [IMG_SHAPE_X, IMG_SHAPE_Jets]
Train = [X_train, Jets1_train]
Test = [X_test, Jets1_test]
AA = [X_AA, JetsAA]
TT = [X_tt, Jetstt]

Dim = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#Dim = [1, 4]


results_npv = {}
'''
for index, img_shape in enumerate(IMG_SHAPE):
    encoder, decoder = build_autoencoder0(img_shape, 5)

    inp = Input(img_shape)
    code = encoder(inp)
    reconstruction = decoder(code)



    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    #autoencoder.compile(optimizer='adamax', loss=masked_mse_loss)

    print(autoencoder.summary())

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    history = autoencoder.fit(x=Train[index], y=Train[index], epochs=100, validation_data=[Test[index], Test[index]],callbacks=[es])

    bkg_test_scores = calculate_batch_score(autoencoder, Test[index])
    tt_test_scores = calculate_batch_score(autoencoder, TTbar[index])


    results_npv[index] = {
        "bkg_scores": bkg_test_scores,
        "tt_scores": tt_test_scores,
        "history": history.history
    }

'''

results_dim = {}
for index, dim in enumerate(Dim):
    encoder, decoder = build_autoencoder0(IMG_SHAPE_X, dim)
    #autoencoder = build_autoencoder0(IMG_SHAPE_X, dim)

    inp = Input(IMG_SHAPE_X)
    code = encoder(inp)
    reconstruction = decoder(code)



    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    #autoencoder.compile(optimizer='adamax', loss=masked_mse_loss)

    print(autoencoder.summary())

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    history = autoencoder.fit(x=Train[0], y=Train[0], epochs=100, validation_data=[Test[0], Test[0]],callbacks=[es])

    bkg_test_scores = calculate_batch_score(autoencoder, Test[0])
    AA_test_scores = calculate_batch_score(autoencoder, AA[0])
    TT_test_scores = calculate_batch_score(autoencoder, TT[0])
    percen_9975 = np.percentile(bkg_test_scores, 99.75)

    # Compute percentage of ttbar passing this threshold
    AA_passed = 100 * np.sum(AA_test_scores > percen_9975) / len(AA_test_scores)
    TT_passed = 100 * np.sum(TT_test_scores > percen_9975) / len(TT_test_scores)



    results_dim[index] = {
        "bkg_scores": bkg_test_scores,
        "AA_scores": AA_test_scores,
        "TT_scores": TT_test_scores,
        "history": history.history,
        "AA_pass":AA_passed,
        "TT_pass":TT_passed
    }





percen_9975_ht = np.percentile(HT_test, 99.75)

# Compute percentage of ttbar passing this threshold
AA_ht_passed = 100 * np.sum(HTAA > percen_9975_ht) / len(HTAA)
TT_ht_passed = 100 * np.sum(HTtt > percen_9975_ht) / len(HTtt)


print('AA_ht_passed:', AA_ht_passed)
print('TT_ht_passed:', TT_ht_passed)


AA_pass_rates = [results_dim[index]["AA_pass"] for index in range(len(Dim))]  
TT_pass_rates = [results_dim[index]["TT_pass"] for index in range(len(Dim))]  

# Plot
plt.figure(figsize=(8, 6))
plt.plot(Dim, AA_pass_rates, marker='o', linestyle='-', color='tab:green', label="HtoAAto4B Pass Rate")
plt.plot(Dim, TT_pass_rates, marker='o', linestyle='-', color='goldenrod', label="ttbar Pass Rate")

plt.hlines(AA_ht_passed,1,15,color="grey",linestyles="dashed",label=f"HToAATo4B, HT Efficiency:{AA_ht_passed:.2f}%")
plt.hlines(TT_ht_passed,1,15,color="grey",linestyles="dashed",label=f"TTBar, HT Efficiency:{TT_ht_passed:.2f}%")
plt.plot([], [], ' ', label="Threshold = 99.75 Percentile of Test Background")
# Labels and title
plt.xlabel("Latent Dimension", fontsize=18)
plt.ylabel("Signal Pass (%)", fontsize=18)
#plt.yscale("log")
#plt.title("Signal Passing 99.75% Threshold of Test Background vs. Latent Dimension", fontsize=15)
plt.grid(True)
plt.xlim(-0.05,16)
plt.ylim(0,100.5)
plt.xticks([2,4,6,8,10,12,14,16])
plt.legend(fontsize=14, loc='best', frameon=True)


# Save and show
plt.tight_layout()
plt.savefig("paper/signal_pass_vs_dimension.png")
plt.close()







# Extracting data for d=1 and d=4 from results_dim
dim_1 = results_dim[0]  
dim_4 = results_dim[3]  

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for dim = 1
threshold1 = np.percentile(dim_1["bkg_scores"],99.995)
#range=(0, threshold)

# overflow bin
n_bins = 50
bins1 = np.linspace(0, threshold1, n_bins)

# Clip scores: anything above threshold1 goes into threshold1
bkg_scores = np.clip(dim_1["bkg_scores"], None, threshold1)
AA_scores  = np.clip(dim_1["AA_scores"],  None, threshold1)
TT_scores  = np.clip(dim_1["TT_scores"],  None, threshold1)

axes[0].hist(bkg_scores, bins=bins1, alpha=1, density=True, label='test MinBias Background', histtype='step', color='tab:blue', linewidth=2)
axes[0].hist(AA_scores, bins=bins1, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green', linewidth=2)
axes[0].hist(TT_scores, bins=bins1, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod', linewidth=2)

axes[0].set_xlabel('Anomaly Score', fontsize=18)
axes[0].set_ylabel('Density', fontsize=18)
axes[0].set_yscale("log")
axes[0].set_xlim(-0.1, 600)

# Add note about overflows
axes[0].legend(fontsize=14, loc='best', frameon=True, title='Latent Dimension = 1\n(Overflows in the Last Bin', title_fontsize=14)

# Plot for dim = 4
threshold4 = np.percentile(dim_4["bkg_scores"],99.995)

bins4 = np.linspace(0, threshold4, n_bins)

bkg_scores4 = np.clip(dim_4["bkg_scores"], None, threshold4)
AA_scores4  = np.clip(dim_4["AA_scores"],  None, threshold4)
TT_scores4  = np.clip(dim_4["TT_scores"],  None, threshold4)

axes[1].hist(bkg_scores4, bins=bins4, alpha=1, density=True, label='test MinBias Background', histtype='step', color='tab:blue',linewidth=2)
axes[1].hist(AA_scores4, bins=bins4, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green',linewidth=2)
axes[1].hist(TT_scores4, bins=bins4, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod',linewidth=2)
#axes[1].set_title('Anomaly Score Distribution (d=4)')
axes[1].set_xlabel('Anomaly Score', fontsize=18)
axes[1].set_ylabel('Density', fontsize=18)
axes[1].set_yscale("log")
axes[1].set_xlim(-0.1,120)
axes[1].set_ylim(10E-6,10)
#axes[1].legend(fontsize=14,title='Latent Dimension = 4', title_fontsize=16, loc='best', frameon=True)
axes[1].legend(fontsize=14, loc='best', frameon=True, title='Latent Dimension = 4\n(Overflows in the Last Bin)', title_fontsize=14)
# Adjust layout and save

plt.tight_layout()



fig.savefig("paper/AS_hist_comparison2016.png")


import matplotlib.transforms as mtransforms

def save_subplot(fig, ax, filename, pad=0.1):
    """Save a single subplot with a little padding (in inches)."""
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    # Expand bbox by pad (fraction of inch)
    bbox = mtransforms.Bbox.from_extents(
        bbox.x0 - pad, bbox.y0 - pad,
        bbox.x1 + pad, bbox.y1 + pad
    )

    fig.savefig(filename, bbox_inches=bbox)


save_subplot(fig, axes[0], "paper/AS_hist_comparison2016-a.png", pad=0.3)
save_subplot(fig, axes[1], "paper/AS_hist_comparison2016-b.png", pad=0.3)
plt.close(fig)
