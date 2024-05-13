FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm

WORKDIR /opt/algorithm
COPY --chown=algorithm:algorithm weights/ /opt/algorithm/weights/

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python3 -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python3 -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT ["python3", "process.py"]

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=picai_unet_baseline_processor
