README.txt for First Training and Testing Attempt, People Detection in an image

To train: 
In current directory, set python path: export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Train with the following settings: python train.py --logtostderr --pipeline_config_path=./models/customized.config --train_dir=./data/${PATH to training folder}

At the same time, to monitor using Tensorboard, use in another window: tensorboard --logdir=./models

After training, export model using: python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./models/customized.config --trained_checkpoint_prefix ./models/train1/model.ckpt-96 --output_directory ./models

To test: 
Run test.ipynb in the object_detection folder
Change TEST_IMAGE_PATHS



