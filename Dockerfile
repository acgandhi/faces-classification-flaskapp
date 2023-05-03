FROM tensorflow/tensorflow

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./model /code/model

COPY . /code/

CMD ["python", "app.py"]
