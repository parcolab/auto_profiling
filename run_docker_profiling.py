import argparse
import os
parser = argparse.ArgumentParser(description='auto profiling tool')
parser.add_argument('--ncu', help="use nsight compute", action='store_true')
parser.add_argument('--mode', help="pytorch or tensorflow")
parser.add_argument('--model', help="chooses model")
parser.add_argument('--batchs', type=int, help="choolse_batchsize", default=-1)
parser.add_argument('--gpus', type=int, help="number of gpus to proflie", default=1)
parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )
args = parser.parse_args()

pytorch_dirs = {
    'ConvNets': ['resnet18','resnet18', 'resnet34','resnet50', 'resnet101', 'resnet152','resnext101-32x4d'],
    'imagenet' :
        ['alexnet', 'densenet121','densenet169', 'densenet201', 'densenet161', 'googlenet', 'inception_v3_google', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
            'mobilenet_v2', 'resnet18', 'resnet34','resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
            'shufflenetv2_x0.5', 'shufflenetv2_x1.0', 'shufflenetv2_x1.5', 'shufflenetv2_x2.0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 
            'vgg13_bn', 'vgg13_bn', 'vgg13_bn'
        ],
    'efficient_det': 
        ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7'],
    'BERT': ['bert-large-uncased']
    }
tensorflow_dirs = {}
pytorch_commands = {
        'ConvNets' : f'python ./multiproc.py --nproc_per_node {args.gpus} ./main.py --arch {args.model} -b {args.batchs} --training-only -p 10 --prof 100 --epochs 1 --data-backend pytorch --memory-format {args.memory_format} /data/ILSVRC2012',
        'imagenet' : f'python main.py -a {args.model} --epochs 1 -b {args.batchs} -j {args.gpus} --multiprocessing-distributed --gpus {args.gpus} --rank=0 /data/ILSVRC2012',
        'efficient_det': f'python train_prof.py --dataset VOC --dataset_root /data/VOCdevkit/ --network {args.model} --batch_size {args.batchs} --iter 100 --wramup 30',
        'BERT': f'{args.batchs} {args.gpus} 130'
    }
tensorflow_commands = {}
working_dir = f'/data/auto_profiling/{args.mode}/'
working_dirs = {}
commands = {}
result_dir = f'/data/outputs/{args.model}-{args.gpus}-{args.batchs}-{args.memory_format}'

if args.mode == 'pytorch':
    working_dirs = pytorch_dirs
    commands = pytorch_commands
elif args.mode == 'tensorflow':
    working_dirs = tensorflow_dirs
    commands = tensorflow_commands

model_dir = ''
docker_cmd =f'sbatch --gres=gpu:8 sbatch_pytorch_docker.sh '
profile_cmd = f'nsys profile -c cudaProfilerApi --stop-on-range-end true -t cuda,nvtx -f true --export sqlite -o {result_dir}/nsys_profile '
if args.ncu:
    profiler_cmd = 'ncu '
for keys, model_list in working_dirs.items():
    if args.model in model_list:
        model_dir = keys
        break
if model_dir == '':
    print("cannnot find model")
    exit(-1)

working_dir = working_dir + model_dir
command = commands[model_dir]

if model_dir == 'BERT':
    os.chdir('/home/hhk971' + working_dir)
    command =f'sbatch --gres=gpu:8 -p A100 scripts/docker/launch.sh \"{profile_cmd}\" \"{command}\"'
else:
    command = f'{docker_cmd} \"{working_dir}\" \"{profile_cmd} {command}\"'
print(command)
os.makedirs('/home/hhk971'+result_dir, exist_ok=True)
os.system(command)
