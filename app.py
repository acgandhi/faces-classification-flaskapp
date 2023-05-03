from flask import Flask, url_for, redirect, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from secrets import token_urlsafe

import numpy as np
import tensorflow as tf


class UploadForm(FlaskForm):
    file = FileField(validators=[FileRequired()])


model = tf.keras.saving.load_model('./model')
preprocessing = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=1./255),
    tf.keras.layers.Resizing(384, 384),
])
class_names = ["Would take out for drinks.", "Wouldn't take out for drinks."]

app = Flask(__name__)
app.config['SECRET_KEY'] = token_urlsafe(10)


def load_image(filename):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_image(raw, channels=3)
    # the `print` executes during tracing.
    print("Initial shape: ", image.shape)
    processed_image = preprocessing(image)
    print("Final shape", processed_image.shape)
    return processed_image


@app.route('/', methods=['GET', 'POST'])
def index():  # put application's code here
    form = UploadForm()
    
    if form.validate_on_submit():
        filename = "static/" + secure_filename(form.file.data.filename)
        form.file.data.save(filename)
        image = load_image(filename)
        prediction = model(np.expand_dims(image, axis=0)).numpy()
        prediction_index = np.argmax(prediction)
        result = {
            'image_name': filename,
            'result': class_names[prediction_index],
            'score_drink': str(prediction[0, 0]),
            'score_nodrink': str(prediction[0, 1]),
        }
        return render_template('index.html', form=UploadForm(), result=result)
    
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
