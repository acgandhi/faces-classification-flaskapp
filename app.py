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
# rescaling: [0, 255] -> [0, 1]
# resizing: n x n x 3 -> 384 x 384 x 3
# layers must be in this order to work!
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
    print("Initial shape: ", image.shape)
    processed_image = preprocessing(image)
    print("Final shape", processed_image.shape)
    return processed_image


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    
    if form.validate_on_submit():
        # save image to disk in 'static' dir so we can serve later
        filename = "static/" + secure_filename(form.file.data.filename)
        form.file.data.save(filename)
        # load image as tensor
        image = load_image(filename)
        # expand from (384, 384, 3) -> (1, 384, 384, 3) and run inference
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
    # host 0.0.0.0 needed for docker
    app.run(debug=False, port=5000, host='0.0.0.0')
