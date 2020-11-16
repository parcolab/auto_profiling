import argparse
import os
parser = argparse.ArgumentParser(description='auto profiling tool')
parser.add_argument('--mode', help="pytorch or tensorflow")
parser.add_argument('--model', help="chooses model")
parser.add_argument('--batchs', type=int, help="choolse_batchsize", default=-1)
parser.add_argument('--gpus', type=int, help="number of gpus to proflie", default=1)
pytorch_dirs = {
    'imagenet' :
        ['alexnet', 'densenet121','densenet169', 'densenet201', 'densenet161', 'googlenet', 'inception_v3_google', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
            'mobilenet_v2', 'resnet18', 'resnet34','resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
            'shufflenetv2_x0.5', 'shufflenetv2_x1.0', 'shufflenetv2_x1.5', 'shufflenetv2_x2.0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 
            'vgg13_bn', 'vgg13_bn', 'vgg13_bn'
        ],
    'efficient_det': 
        ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7']
    }
tensorflow_dirs = {}
pytorch_commands = {
        'imagenet' : '\"python main.py -a {model} --epochs 1 -b {batchs} --multiprocessing-distributed --rank=0 /home/shared/ILSVRC2012\"',
        #'imagenet' : '\"python main.py -a {model} --epochs 1 -b {batchs}  --rank=0 /home/shared/ILSVRC2012\"',
        'efficient_det': '\"python train_prof.py --dataset VOC --dataset_root /home/shared/VOCdevkit/ --network {model} --batch_size {batchs}\"'
    }
tensorflow_commands = {}
working_dir = os.getcwd()
working_dirs = {}
commands = {}
result_dir = os.getcwd() + '/results'
args = parser.parse_args()
working_dir = working_dir+ '/'+args.mode

if args.mode == 'pytorch':
    working_dirs = pytorch_dirs
    commands = pytorch_commands
elif args.mode == 'tensorflow':
    working_dirs = tensorflow_dirs
    commands = tensorflow_commands

model_dir = ''
for keys, model_list in working_dirs.items():
    if args.model in model_list:
        model_dir = keys
        break
if model_dir == '':
    print("cannnot find model")
    exit(-1)

working_dir = working_dir + '/' + model_dir
command = commands[model_dir]
if args.batchs > 0:
    result_dir = result_dir + '/' + args.mode + '/' + args.model + '/' +args.model+'gpu-'+ str(args.gpus)+'batchs-'+str(args.batchs) 
    os.makedirs(result_dir, exist_ok=True)
    sbatch_command = 'sbatch sbatch_gpu'+ str(args.gpus) + '.sh ' + command.format(model=args.model, batchs=args.batchs) + ' \"'+ working_dir  + '\" \"' + result_dir + '\"'
    os.system(sbatch_command)
else:
    batchsize=16
    for i in range(4):
        result_dir = result_dir + '/' + args.mode + '/' + args.model + '/' +args.model+'gpu-'+ str(args.gpus)+'batchs-'+str(batchsize) 
        os.makedirs(result_dir, exist_ok=True)
        sbatch_command = 'sbatch sbatch_gpu'+ str(args.gpus) + '.sh ' + command.format(model=args.model, batchs=batchsize) + ' \"'+ working_dir  + '\" \"' + result_dir + '\"'
        os.system(sbatch_command)    
        batchisize*=2
