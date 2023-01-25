# BaseImage (OS)
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN mkdir -p /service

WORKDIR /service
COPY . /service

RUN apt-get -y update
RUN pip install --upgrade jinja2
RUN pip install typing-extensions --upgrade
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "./api_server.py"]