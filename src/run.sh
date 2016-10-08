#!/bin/bash
rm compute.log
python train_and_test.py blocks t
vim compute.log
