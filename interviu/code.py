from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score

# TODO
train = pd.read_csv("train.csv")
test = pd.read_csv("test_features.csv")
validation = pd.read_csv("validation.csv")
from pandas import DataFrame
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score

def create_model(vector_length=128):
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # print summary of the model
    model.summary()
    return model

train_new = train._get_numeric_data()

kmeans = KMeans(n_clusters=3).fit(train_new.drop(columns=["labels"]))

labels = kmeans.predict(train_new.drop(columns=["labels"]))
print(labels)
print(rand_score(labels, train_new["labels"]))

labels = kmeans.predict(validation._get_numeric_data().drop(columns=["labels"]))
print(labels)
print(rand_score(labels, validation["labels"]))


model = create_model()

# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
# define early stopping to stop training after 5 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

batch_size = 64
epochs = 100


model.fit(train_new.drop(columns=["labels"]), train_new["labels"], epochs=epochs, batch_size=batch_size, validation_data=(validation._get_numeric_data().drop(columns=["labels"]), validation["labels"]),
          callbacks=[tensorboard, early_stopping])
