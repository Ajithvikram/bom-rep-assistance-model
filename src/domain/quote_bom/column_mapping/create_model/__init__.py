import tensorflow as tf

def createModel(embeddingDim):
    try:
        print("Create model Started...")
        tower = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(embeddingDim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu")
        ])
        print("Tower created...")

        input_a = tf.keras.Input(shape=(embeddingDim,))
        input_b = tf.keras.Input(shape=(embeddingDim,))
        print("Input created...")

        emb_a = tower(input_a)
        emb_b = tower(input_b)
        print("Embedding done for inputs...")

        # L1 distance
        distance = tf.abs(emb_a - emb_b)
        print("Distance calculated...")

        # Output similarity
        output = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
        print("Output created...")

        model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
        print("Create compile Started...")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print("Create compile Ended...")
        print("Create model Ended...")
        return model, tower
    except Exception as e:
        print("Error occurred on create model:", e)