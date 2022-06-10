#!/bin/bash
ulimit -c unlimited
rm /dev/shm/coredump/*
rm /dev/shm/*.jpg
rm /dev/shm/*.bmp
make -j -s
./boe
