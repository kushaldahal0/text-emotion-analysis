from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np


app = Flask(__name__)

# model = tf.keras.models.load_model('models/modell.tf')
model = tf.keras.models.load_model('models/modell93.tf')

categs = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
batch_size = 32
def predict(text):
  text_slice  = tf.data.Dataset.from_tensor_slices([text])
  prefetched = text_slice.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  res = model.predict(prefetched)
  # print(res)
  probabilities = np.exp(res) / np.sum(np.exp(res), axis=1, keepdims=True)
  predicted_class_index = np.argmax(probabilities)
  return(categs[predicted_class_index])

def res(text):
    return f"<p class= 'text-6xl text-white sm:text-xl'>Emotion :<br>{predict(text)}<br> Text :<br>{text} </p>"


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        text = request.form['text']
        result = res(text)
        return render_template('index.html', res=result)
    else:
        return render_template('index.html',  res='')

if __name__ == '__main__':
    app.run(debug=True)

