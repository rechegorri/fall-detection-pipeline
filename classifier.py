import tensorflow as tf
import numpy as np

class Classifier:
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model("./pipeline/model_mbnv2.01-0.10.keras")

    def predict(self,data) -> str:
        prediction = self.model.predict(np.stack((data,),axis=0), verbose=0)
        return np.max(tf.nn.sigmoid(prediction))
