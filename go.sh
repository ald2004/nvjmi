#!/bin/bash
ulimit -c unlimited
rm /dev/shm/coredump/*
make -j -s
./boe
