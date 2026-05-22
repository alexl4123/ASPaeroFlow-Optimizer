FROM mambaorg/micromamba:1.5-alpine

USER root
RUN apk add --no-cache bash

# Copy environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Install dependencies
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Ensure environment is active by default
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

# Copy your optimizer and controller source code
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

USER $MAMBA_USER

# Expose necessary ports (update if the optimizer also needs one)
EXPOSE 8080

# Base entrypoint: everything passed as a command will be appended to this
ENTRYPOINT ["micromamba", "run", "-n", "base", "python"]
