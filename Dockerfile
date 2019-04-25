FROM continuumio/miniconda3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install -y --no-install-recommends --allow-unauthenticated \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip \
    libffi-dev \
    software-properties-common

RUN mkdir /fever

WORKDIR /fever
RUN mkdir -pv src
RUN mkdir -pv configs
RUN mkdir -pv data


ADD src src
ADD configs configs

ADD requirements.txt /fever/

export PYTHONPATH="src"

RUN pip install spacy==2.1.3
RUN python -m spacy download en
RUN wget -O data/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db
RUN ln -s data/fever.db
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

ADD predict.sh .

ENV PYTHONPATH src
ENV FLASK_APP sample_application:my_sample_fever

#ENTRYPOINT ["/bin/bash","-c"]
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "sample_application:my_sample_fever"]
