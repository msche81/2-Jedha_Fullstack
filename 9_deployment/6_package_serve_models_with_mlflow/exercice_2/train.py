import argparse
import mlflow 
import tensorflow as tf 
import tensorflow_hub as hub
import pandas as pd 
from mlflow.models.signature import infer_signature
import os

# Set tracking URI to your Hugging Face application
# mlflow.set_tracking_uri(os.environ["APP_URI"])

if __name__=="__main__":

    # Parse arguments given in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs")
    parser.add_argument("--lr")
    args = parser.parse_args()
    

    ### Create autolog 
    mlflow.tensorflow.autolog()

    ### Import dataset of french reviews of Disneyland
    french_reviews = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/images/M08-DeepLearning/NLP/french_review_clean.csv")

    # Shuffle your dataset 
    shuffle_df = french_reviews.sample(frac=1)

    # Define a size for your train set 
    train_size = int(0.7 * len(french_reviews))

    # Split your dataset 
    train_set = shuffle_df.loc[:train_size]
    test_set = shuffle_df[train_size:]

    # Extract only reviews and target
    X_train = tf.convert_to_tensor(train_set["review_format"])
    y_train = tf.convert_to_tensor(train_set["stars"]-1)

    X_test = tf.convert_to_tensor(test_set["review_format"])
    y_test = tf.convert_to_tensor(test_set["stars"]-1)


    pre_trained_model="https://tfhub.dev/google/nnlm-en-dim50/2"

    model = tf.keras.Sequential([
                    # Pretrained model
                    hub.KerasLayer(pre_trained_model, input_shape=[], dtype=tf.string, trainable=False),

                    # Dense layers once the data is flat
                    #tf.keras.layers.Dense(64, activation='relu'),
                    #tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    # output layer with as many neurons as the number of classes
                    # for the target variable and softmax activation
                    tf.keras.layers.Dense(5, activation="softmax")
    ])

    ### Configure learning rate

    optimizer= tf.keras.optimizers.Adam(learning_rate= float(args.lr))

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    weights = 1/(french_reviews["stars"]-1).value_counts()
    weights = weights * len(french_reviews)/5
    weights = {index : values for index , values in zip(weights.index,weights.values)}

    # Log experiment to MLFlow
    with mlflow.start_run() as run:
        #### Training model 
        epochs = int(args.epochs)
        model.fit(X_train,
                y_train,
                epochs=epochs, 
                batch_size=64,
                validation_data=(X_test, y_test),
                class_weight=weights)


        predictions = model.predict(X_train)

        # Log model seperately to have more flexibility on setup 
        #mlflow.keras.log_model(
        #    keras_model=model,
        #    artifact_path="Sentiment_detector",
        #    registered_model_name="Sentiment_detector_RNN",
        #    signature=infer_signature(french_reviews, predictions)
        #)
        
