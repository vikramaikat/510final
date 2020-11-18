#!/bin/bash

##############
# Run tests. #
##############

echo "Testing BasicDistributedModel..."
python -m unittest -v code/test_basic_distributed.py
echo

echo "Testing GpipeModel..."
python -m unittest -v code/test_gpipe.py
echo

echo "Testing RefinementModel..."
python -m unittest -v code/test_refinement_model.py
echo

echo "Done."
