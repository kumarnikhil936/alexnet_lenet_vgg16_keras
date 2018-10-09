#!/bin/bash

echo -e "\n\n"
echo "**************************************"
echo "***                                ***"
echo "*** Open your browser to port 6006 ***"
echo "***                                ***"
echo "**************************************"
echo -e "\n\n"

/home/student/myenv/bin/tensorboard --logdir=./simple_model_logs
#/home/student/myenv/bin/tensorboard --logdir=./logs
#/home/student/myenv/bin/tensorboard --logdir=./logs
