# https://stackoverflow.com/questions/6523791/why-is-python-running-my-module-when-i-import-it-and-how-do-i-stop-it
from train_1705027 import *
import sys
import csv

TEST_MODEL_PATH = '1705027_model.pickle'
TEST_DIR = sys.argv[1]
IMAGE_DIM = 28

with open(TEST_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# LOAD DATA
# https://stackoverflow.com/questions/64304446/efficient-way-to-detect-white-background-in-image

results = []
results.append(['FileName', 'Digit'])
for image in sorted(os.listdir(TEST_DIR)):
    # print(image)
    if image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        path = os.path.join(TEST_DIR, image)
        img = read_image(path)
        img = np.array(img)
        img = img.reshape(1, IMAGE_DIM, IMAGE_DIM, 1)
        y_pred = model.predict(img)
        results.append([image, y_pred[0]])

# write to csv
with open(f'{TEST_DIR}/1705027_prediction.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(results)
