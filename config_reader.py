import copy
import multiprocessing as mp


def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    # ctx = mp.get_context('spawn')
    run_args, _run_config, _run_repeat = _return_configs(arg_parser, args)
    return run_args

def _read_config(path):
    """
    读取模型config的设置
    path：模型config的地址
    """
    lines = open(path).readlines()

    runs = []  # 一个字典列表
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()  # 去除空格的行

        # continue in case of comment
        if stripped_line.startswith('#'):
            # 如果设置被注释，则跳过该设置
            continue

        ### 如果遇到空行， 表示一个新的模型config
        if not stripped_line:  # stripped_line不为空
            if run[1]:  # 如果run的dict字典内有值，则runs把该值加入
                runs.append(run)

            run = [1, dict()]  # 重置run
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            # 将设置按照键值对的格式存储到run
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs  # # [[1, {model_config}],[2, {model_config}]]


def _convert_config(config:dict):
    config_list = []
    for k, v in config.items():
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))

    return config_list


def _return_configs(arg_parser, args, verbose=True):
    """
    arg_parse:parser对象
    args:NameSpace
    """
    _print = (lambda x: print(x)) if verbose else lambda x: x

    if args.config:  #train_config path
        # 如过可选参数config不为空
        config = _read_config(args.config)  # config为

        for run_repeat, run_config in config:
            # run_repeat: number表示模型config的个数
            print("-" * 50)
            print("Config:")
            print(run_config)

            args_copy = copy.deepcopy(args)
            config_list = _convert_config(run_config)  # ['--label', 'conll04_train', '--model_type', 'spert']
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)
            run_args_dict = vars(run_args)

            # set boolean values
            for k, v in run_config.items():
                if v.lower() == 'false':
                    run_args_dict[k] = False

            print("Repeat %s times" % run_repeat)
            print("-" * 50)

            for iteration in range(run_repeat):
                _print("Iteration %s" % iteration)
                _print("-" * 50)

                return run_args, run_config, run_repeat

    else:
        return args, None, None
