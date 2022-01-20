FROM python:3.7
WORKDIR /src
COPY ./requirements.txt /src/requirements.txt
COPY ./TweetProcess.py /src/TweetProcess.py
EXPOSE 80
RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt
ENV PYTHONPATH /src
COPY ./app /src/app
CMD ["uvicorn", "app.model_serve:app", "--host", "0.0.0.0", "--port", "80"]