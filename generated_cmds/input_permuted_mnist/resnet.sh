python3 core/run/run.py --task input_permuted_mnist_convolution --learner adam --seed 0 --lr 0.001 --network resnet8_sigmoid  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner adam --seed 0 --lr 0.001 --network resnet8_relu  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner adam --seed 0 --lr 0.001 --network resnet8_leakyrelu  --n_samples 1000000

python3 core/run/run.py --task input_permuted_mnist_convolution --learner sgd --seed 0 --lr 0.001 --network resnet8_sigmoid  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner sgd --seed 0 --lr 0.001 --network resnet8_relu  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner sgd --seed 0 --lr 0.001 --network resnet8_leakyrelu  --n_samples 1000000

python3 core/run/run.py --task input_permuted_mnist_convolution --learner upgd_fo_global --seed 0 --lr 0.001 --beta_utility 0.9999 --sigma 0.01 --weight_decay 0.01 --network resnet8_sigmoid --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner upgd_fo_global --seed 0 --lr 0.001 --beta_utility 0.9999 --sigma 0.01 --weight_decay 0.01 --network resnet8_relu --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner upgd_fo_global --seed 0 --lr 0.001 --beta_utility 0.9999 --sigma 0.01 --weight_decay 0.01 --network resnet8_leakyrelu --n_samples 1000000

python3 core/run/run.py --task input_permuted_mnist_convolution --learner upgd_fo_global --seed 0 --lr 0.001 --beta_utility 0.9999 --sigma 0.01 --network resnet8_sigmoid --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner upgd_fo_global --seed 0 --lr 0.001 --beta_utility 0.9999 --sigma 0.01 --network resnet8_relu --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner upgd_fo_global --seed 0 --lr 0.001 --beta_utility 0.9999 --sigma 0.01 --network resnet8_leakyrelu --n_samples 1000000

python3 core/run/run.py --task input_permuted_mnist_convolution --learner adam --seed 0 --lr 0.001 --weight_decay 0.01 --network resnet8_sigmoid  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner adam --seed 0 --lr 0.001 --weight_decay 0.01 --network resnet8_relu  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner adam --seed 0 --lr 0.001 --weight_decay 0.01 --network resnet8_leakyrelu  --n_samples 1000000

python3 core/run/run.py --task input_permuted_mnist_convolution --learner sgd --seed 0 --lr 0.001 --weight_decay 0.01 --network resnet8_sigmoid  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner sgd --seed 0 --lr 0.001 --weight_decay 0.01 --network resnet8_relu  --n_samples 1000000
python3 core/run/run.py --task input_permuted_mnist_convolution --learner sgd --seed 0 --lr 0.001 --weight_decay 0.01 --network resnet8_leakyrelu  --n_samples 1000000