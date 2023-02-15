#!/usr/bin/env python
# coding: utf-8


# !pip install opencv-python
# !pip install tqdm


import numpy as np
import pandas as pd
import math
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns


# # PARAMETERS
LEARNING_RATE = 0.1
MAX_EPOCHS = 20

TRAINING_A_DIR = './NumtaDB/training-a/'
TRAINING_A_CSV = './NumtaDB/training-a.csv'

TRAINING_B_DIR = './NumtaDB/training-b/'
TRAINING_B_CSV = './NumtaDB/training-b.csv'

TRAINING_C_DIR = './NumtaDB/training-c/'
TRAINING_C_CSV = './NumtaDB/training-c.csv'

TRAINING_D_DIR = './NumtaDB/training-d/'
TRAINING_D_CSV = './NumtaDB/training-d.csv'

TRAINING_E_DIR = './NumtaDB/training-e/'
TRAINING_E_CSV = './NumtaDB/training-e.csv'

DONT_INVERT_IMAGE_DIR = "training-e"

MODEL_INIT_FILE = './model_desc.txt'
IMAGE_DATASET_DIRS = [TRAINING_A_DIR, TRAINING_B_DIR, TRAINING_C_DIR]
CSV_FILES = [TRAINING_A_CSV, TRAINING_B_CSV, TRAINING_C_CSV]
MINI_BATCH_SIZE = 64
IMAGE_DIM = 28 # Height and width of the image

X_train = []
y_train = []
X_val, y_val = [], []
confusionMatrix = None
Model = None


# # PROCESS DATASET
# https://stackoverflow.com/questions/64304446/efficient-way-to-detect-white-background-in-image
def find_white_background(imgArr, threshold=0.3):
    background = np.array([200])
    # print(imgArr)
    percent = (imgArr >= background).sum() / imgArr.size
    
    if percent >= threshold:
        # print("White % : ",percent)
        return True
    else:
        # print("White % : ",percent)
        return False

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    # Apply cropping by dropping rows and columns with average pixel intensity >= 253
    
    # Inverting the image
    if find_white_background(img):
        img = 255 - img
    
    img = img.astype('float32')
    kernel = np.ones((2,2), np.uint8)
    # eroding the image
    img = cv2.erode(img, kernel, iterations=1)
    # dilating the image
    img = cv2.dilate(img, kernel, iterations=1)
    # print(img.shape)
    # reshaping the image
    img = cv2.resize(img,(IMAGE_DIM, IMAGE_DIM))
    # plt.imshow(img, cmap='gray')
    # plt.show()
    img /= 255.0
    img = img.reshape(IMAGE_DIM, IMAGE_DIM, 1) # 1 for grayscale
    return img


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomImageDataset:
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, self.img_labels.columns.get_loc('filename')])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, self.img_labels.columns.get_loc('digit')] # 3 is the column index of the label
        #one hot encoding
        label = np.eye(10)[label]
        return image, label

class CustomDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        # Get the next batch.
        if self.current_idx + self.batch_size > len(self.dataset):
            batch = [self.dataset[i] for i in range(self.current_idx, len(self.dataset))]
            self.current_idx = len(self.dataset)
        else:
            batch = [self.dataset[i] for i in range(self.current_idx, self.current_idx + self.batch_size)]
            self.current_idx += self.batch_size
        

        if self.shuffle:
            np.random.shuffle(batch)

        images, labels = zip(*batch)
        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels


# # READ DATASET
def READ_DATASET():
    global X_train, y_train
    print("Reading Images...")
    for i, IMAGE_DATASET_DIR in enumerate(IMAGE_DATASET_DIRS):
        CSV_FILE = CSV_FILES[i]
        dataset = CustomImageDataset(annotations_file= CSV_FILE, img_dir=IMAGE_DATASET_DIR)
        dataloader = CustomDataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=False)
        for images, labels in dataloader:
            # print(images.shape, labels.shape)
            X_train.append(images)
            y_train.append(labels)
            # break
        # break
                
    print("Reading Images Completed")

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    print(X_train.shape, y_train.shape)



# <h3> WINDOWS: as_strided </h3>
def getSlidingWindows(input, output_size, filter_dim, padding=0, stride=1, dilate=0):
    # Create a working tensor, starting with the original input
    working_tensor = input
    current_padding = padding
    
    # Dilate the input if necessary
    if dilate != 0:
        working_tensor = np.insert(working_tensor, range(1, input.shape[1]), 0, axis=1)
        working_tensor = np.insert(working_tensor, range(1, input.shape[2]), 0, axis=2)

    # Pad the input if necessary
    if current_padding != 0:
        working_tensor = np.pad(working_tensor, pad_width=((0,), (current_padding,), (current_padding,), (0,)), mode='constant', constant_values=(0.,))
    
    # Get the batch size, output height, output width, and output channels
    in_b, out_h, out_w, in_c = output_size
    out_b, _, _, out_c = input.shape
    # Get the stride values for each dimension of the tensor
    batch_str, filter_h_str, filter_w_str, channel_str = working_tensor.strides

    sliding_windows = np.lib.stride_tricks.as_strided(working_tensor,
        shape = (out_b, out_h, out_w, filter_dim, filter_dim, out_c),
        strides = (batch_str, stride * filter_h_str, stride * filter_w_str, filter_h_str, filter_w_str, channel_str)
    )

    return sliding_windows

# # CLASSES FOR CNN
# <h2>SOFTMAX LAYER</h2>
# done
# https://stats.stackexchange.com/questions/304758/softmax-overflow?fbclid=IwAR0jL84MQqvY2Xk_CeBwaUM7kwtRBvV6O7yKtJhtcvGtawp0BwhKsZ4Buwk
class Softmax_Layer:
    def __init__(self):
        self.layer_type = 'Softmax'
    
    def __str__(self):
        return f"{self.layer_type} Layer"
    
    def forward(self, X):
        x_max = np.max(X, axis=1)
        x_max = x_max.reshape(x_max.shape[0], 1)
        Z = np.exp(X-x_max)
        # Z = np.exp(X)
        sum = np.einsum('ij->i', Z)
        sum = sum.reshape(sum.shape[0], 1)
        return Z / sum
    
    def backward(self, dZ, learning_rate=0.0001):
        return np.copy(dZ)
    
    def clear_cache(self):
        pass


# <h2>ReLU ACTIVATION </h2>
# done
class ReLU_Activation:
    def __init__(self):
        self.layer_type = 'ReLU'
    
    def __str__(self):
        return f"{self.layer_type} Activation"
    
    def forward(self, X):
        self.X = X
        Z = np.copy(X)
        Z[Z < 0] = 0
        return Z
    
    def backward(self, dZ, learning_rate=0.0001):
        dX = np.copy(self.X)
        dX[dX < 0] = 0
        dX[dX > 0] = 1
        return dX * dZ
    
    def clear_cache(self):
        self.X = None


# <h2>FULLY CONNECTED LAYER</h2>
# done
class Fully_Connected_Layer:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.W = None
        self.b = None

    def __str__(self):
        return f"Fully Connected Layer(output_dim={self.output_dim})"
    
    def forward(self, X):
        self.X = X

        if self.W is None:
            self.W = np.random.randn(X.shape[1], self.output_dim) * math.sqrt(2 / X.shape[0])
        
        if self.b is None:
            self.b = np.zeros((1, self.output_dim))

        Z = np.einsum('ij,jk->ik', X, self.W) + self.b
        
        return Z
    
    def backward(self, dZ, learning_rate=0.0001):
        dW = np.einsum('ij,ik->jk', self.X, dZ) / self.X.shape[0]
        db = np.einsum('ij->j', dZ) / self.X.shape[0] 
        dX = np.einsum('ij,jk->ik', dZ, self.W.T)

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db

        return dX
    
    def clear_cache(self):
        self.X = None


# <h2>FLATENNING LAYER</h2>
class Flatenning_Layer:
    def __init__(self):
        self.layer_type = 'Flatten'
    
    def __str__(self):
        return f"{self.layer_type} Layer"
    
    def forward(self, X):
        self.input_shape = X.shape
        # print(f"input shape : {X.shape}")
        # print(f"output shape : {X.reshape((X.shape[0], -1)).shape}")
        return X.reshape((X.shape[0], -1))
    
    def backward(self, dZ, learning_rate=0.0001):
        dX = np.copy(dZ)
        return dX.reshape(self.input_shape)
    
    def clear_cache(self):
        self.input_shape = None


# <h1>MAX POOLING</h1>
# new max pooling
class Max_Pooling:
    def __init__(self, filter_dim, stride):
        self.layer_type = 'Max Pooling'
        self.filter_dim = filter_dim
        self.stride = stride
    
    def __str__(self):
        return f"{self.layer_type} (filter_dim={self.filter_dim}, stride={self.stride})"
    

    def forward(self, X):
        self.X_shape = X.shape
        n, h, w, c = X.shape
        out_h = (h - self.filter_dim) // self.stride + 1
        out_w = (w - self.filter_dim) // self.stride + 1
        
        X_strided = getSlidingWindows(X, (n, out_h, out_w, c), self.filter_dim, stride=self.stride)
        
        maxs = np.max(X_strided, axis=(3, 4))

        maximums = maxs.repeat(self.filter_dim, axis=1).repeat(self.filter_dim, axis=2)

        X_window = X[:, :out_h * self.stride, :out_w * self.stride, :]
        
        self.Z_max_idx = np.equal(X_window, maximums).astype(int)

        return maxs

    def backward(self, dZ, learning_rate=0.0001):
        # print(f"dZ shape : {dZ.shape}")
        dA = dZ.repeat(self.filter_dim, axis=1).repeat(self.filter_dim, axis=2)
        # print(f"Z_max_idx shape : {self.Z_max_idx.shape}") 
        # print(f"dA shape : {dA.shape}")
        dA = np.multiply(dA, self.Z_max_idx)
        
        # this is the shape of the input
        dX = np.zeros(self.X_shape)
        dX[:, :dA.shape[1], :dA.shape[2], :] = dA 
        
        return dX

    def clear_cache(self):
        self.X_shape = None
        self.Z_max_idx = None


# <h1>CONVOLUTION</h1>

# https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
# https://blog.ca.meron.dev/Vectorized-CNN/?fbclid=IwAR2GWeVd2AnOGLJrpDqNAFC6F2m5dhrNamB-km8y7D-0TKm4K7Uz1W-2L6Y
class Convolution:
    def __init__(self, num_output_channels, filter_dim, stride=1, padding=0):
        self.layer_type = 'Convolution'
        self.num_output_channels = num_output_channels
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.W = None
        self.b = None
    
    def __str__(self):
        return f"{self.layer_type} (num_output_channels={self.num_output_channels}, filter_dim={self.filter_dim}, stride={self.stride}, padding={self.padding})"
    
    def forward(self, X):
        self.X = X
        n, h, w, c = X.shape
        out_h = (h - self.filter_dim + 2*self.padding) // self.stride + 1
        out_w = (w - self.filter_dim + 2*self.padding) // self.stride + 1

        if self.W is None:
            self.W = np.random.randn(self.num_output_channels, self.filter_dim, self.filter_dim, X.shape[3]) * math.sqrt(2 / X.shape[0])
        if self.b is None:
            self.b = np.zeros((self.num_output_channels))
        
        X_strided = getSlidingWindows(X, (n, out_h, out_w, c), self.filter_dim, self.padding, self.stride)
        self.X_strided = X_strided

        Z = np.einsum('ijklmn,olmn->ijko', X_strided, self.W) + self.b

        return Z
    
        
    def backward(self, dZ, learning_rate=0.0001):
        # print(f"dZ shape : {dZ.shape}")
        padding = self.filter_dim - 1 if self.padding == 0 else self.padding # if padding is 0, then we need to pad the input with the filter size - 1
        # print(f"padding : {padding}")
        dZ_strided = getSlidingWindows(dZ, self.X.shape, self.filter_dim, padding=padding, stride=1, dilate=self.stride - 1)
        # print(f"dZ_strided shape : {dZ_strided.shape}")
        weight_rotatedBy180 = np.rot90(self.W, 2, axes=(1, 2))
        # print(f"weight_rotatedBy180 shape : {weight_rotatedBy180.shape}")

        db = np.einsum('mijc->c', dZ)/dZ.shape[0] # mean over batch size
        dW = np.einsum('ijkmno,ijkl->lmno', self.X_strided, dZ)/dZ.shape[0]
          
        dX = np.einsum('mhwijc,cijk->mhwk', dZ_strided, weight_rotatedBy180)
    
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db
        return dX
    
    def clear_cache(self):
        self.X = None
        self.X_strided = None


# <h1>MODEL</h1>

class Model:
    def __init__(self, filePath):
        self.layers = []
        self.filePath = filePath
        self.build_model()

    def __str__(self):
        string = 'MODEL DETAILS:\n\n'
        for i, layer in enumerate(self.layers):
            string += f"Layer {i+1}: {layer}\n"
        return string
    
    def build_model(self):
        #check if file exists
        if not os.path.exists(self.filePath):
            print('File does not exist')
            return
        with open(self.filePath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue

                line = line.strip()
                
                if line == '':
                    continue

                line_split = line.split(' ')
                layer_name = str(line_split[0]).upper()
                
                if layer_name == 'FC':
                    output_dim = int(line_split[1])
                    self.layers.append(Fully_Connected_Layer(output_dim))

                elif layer_name == 'CONV':
                    num_output_channels = int(line_split[1])
                    filter_dim = int(line_split[2])
                    stride = int(line_split[3])
                    padding = int(line_split[4])
                    self.layers.append(Convolution(num_output_channels, filter_dim, stride, padding))

                elif layer_name == 'MAXPOOL':
                    filter_dim = int(line_split[1])
                    stride = int(line_split[2])
                    self.layers.append(Max_Pooling(filter_dim, stride))

                elif layer_name == 'FLATTEN':
                    self.layers.append(Flatenning_Layer())

                elif layer_name == 'RELU':
                    self.layers.append(ReLU_Activation())

                elif layer_name == 'SOFTMAX':
                    self.layers.append(Softmax_Layer())
                
                else:
                    print('Invalid layer name')
                    return
        
    def forward(self, X):
        for layer in self.layers:
            # print("forward : ", layer)
            X = layer.forward(X)
        return X
    
    def backward(self, dZ, learning_rate=0.0001):
        for layer in reversed(self.layers):
            # print("Backward : ",layer)
            dZ = layer.backward(dZ, learning_rate)
        return dZ
    
    def train(self, X, Y, X_val, y_val, learning_rate=0.0001, epochs=10, batch_size=64):
        epochs = int(epochs)
        results = []
        for epoch in tqdm(range(epochs)):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                Z = self.forward(X_batch)
                dZ = Z - Y_batch
                self.backward(dZ, learning_rate)
                # print(f"progress: {i+batch_size}/{X.shape[0]} completed. Loss: {self.loss(X_batch, Y_batch)}")
            train_loss = self.loss(X, Y)
            train_acc, y_pred = self.evaluate(X, Y)
            y_true = np.argmax(Y, axis=1)
            F1_score = f1_score(y_true, y_pred, average='macro')
            
            val_loss = self.loss(X_val, y_val)
            val_acc, _ = self.evaluate(X_val, y_val)
            results.append([train_loss, train_acc, val_loss, val_acc, F1_score])
            print(f"Epoch {epoch+1} completed. Train Loss: {train_loss}, Train Accuracy: {train_acc}, f1 score: {F1_score},\nValidation Loss: {val_loss} ,Validation Accuracy: {val_acc}")
            if val_acc>90:
                print("\n==================>>>Early Stopping. Validation Accuracy > 90% <<<=======================\n")
                break
        return results

    def predict(self, X):
        Z = self.forward(X)
        return np.argmax(Z, axis=1)
    
    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y, axis=1)
        return np.sum(Y_pred == Y_true) / len(Y_true) * 100, Y_pred
    
    def loss(self, X, Y):
        Z = self.forward(X)
        return -np.mean(Y * np.log(Z))
    
    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()
    
    def getConfusionMatrix(self, X, Y):
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y, axis=1)
        return confusion_matrix(Y_true, Y_pred)

# confusion matrix
def save_matrix(matrix):
    plt.figure(figsize=(10, 10))
    axis = sns.heatmap(matrix, annot=True)
    axis.set_xlabel('Predicted labels')
    axis.set_ylabel('True labels')
    axis.set_title('Confusion Matrix')
    plt.savefig('1705027_confusion_matrix.png')

# # BUILD MODEL
def train_model():
    model = Model(MODEL_INIT_FILE)
    print(model)

    results = model.train(X_train, y_train, X_val, y_val, learning_rate=LEARNING_RATE, epochs=MAX_EPOCHS, batch_size=64)
    
    confsuionMatrix = model.getConfusionMatrix(X_val, y_val)
    model.clear_cache()
    return results, confsuionMatrix, model

# # SAVE MODEL

def plot_loss_acc(txt_path):
    output = np.loadtxt(txt_path)
    train_loss = output[:,0]
    train_acc = output[:,1]/100
    val_loss = output[:,2]
    val_acc = output[:,3]/100
    f1_score = output[:,4]
    print(train_loss)
    epochs = range(1, len(train_loss)+1, 1)
    print(epochs)
    # plot all in one graph
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'y', label='Validation accuracy')
    # plt.plot(epochs, f1_score, 'c', label='F1 score')
    plt.title('Training and validation')
    plt.legend()
    plt.savefig("1705027_plot.png")

def save_model(results, confsuionMatrix, model):
    # to save output of model
    if len(results) != 0:
        with open("1705027_output.txt", "w") as file:
            for result in results:
                file.write(f"{result[0]} {result[1]} {result[2]} {result[3]} {result[4]}\n")
        plot_loss_acc("1705027_output.txt")

        # to save confusion matrix
        save_matrix(confsuionMatrix)

    with open('1705027_model.pickle', 'wb') as file:
        pickle.dump(model, file)


# # TESTING
def test_model(model):
    # TEST_MODEL_PATH = './saved_model/model_cnn.pkl'
    TEST_DIR = TRAINING_D_DIR
    TEST_CSV = TRAINING_D_CSV
    TEST_DATA_PATH = TEST_DIR
    TEST_DATA_LABEL_PATH = TEST_CSV

    test_dataset = CustomImageDataset(annotations_file= TEST_DATA_LABEL_PATH, img_dir=TEST_DATA_PATH)
    test_dataloader = CustomDataLoader(test_dataset, batch_size=128, shuffle=False)

    X_test = []
    y_test = []

    for images, labels in test_dataloader:
        # print(images.shape, labels.shape)
        X_test.append(images)
        y_test.append(labels)

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    print(X_test.shape, y_test.shape)
    # print(model.predict(images))
    acc, _ = model.evaluate(X_test, y_test)
    print("Test Accuracy: ", acc)

if __name__ == '__main__':
    READ_DATASET()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    print(X_train.shape, y_train.shape,"\n", X_val.shape, y_val.shape)
    results, confsuionMatrix, model = train_model()
    save_model(results, confsuionMatrix, model)
    save_matrix(confsuionMatrix)
    test_model(model)