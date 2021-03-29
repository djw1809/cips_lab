# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

#FROM ubuntu:16.04

#MAINTAINER Amazon AI <sage-learner@amazon.com>


#RUN apt-get -y update && apt-get install -y --no-install-recommends \
#         wget \
#         python \
#         nginx \
#         ca-certificates \
#    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
#RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
#        (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
#        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.6.0-gpu-py36-cu101-ubuntu16.04
#RUN /bin/bash
RUN apt-get -y update
RUN apt-get -y install nginx
#RUN apt-get -y install gunicorn

RUN pip install transformers
RUN pip install flask
RUN pip install gevent
RUN pip install gunicorn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
#COPY model /opt/program/model
EXPOSE 8080
COPY prompt_model /opt/program
WORKDIR /opt/program
CMD ["serve"]
ENTRYPOINT ["serve"]
