#!/bin/bash
DOCKER_BUILDKIT=1 docker build --progress=plain . -f docker/Dockerfile -t playertr/tsgrasp