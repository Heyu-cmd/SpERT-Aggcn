import argparse

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.spert_trainer import SpERTTrainer


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    args = process_configs(target=__eval, arg_parser=arg_parser)
    __eval(args)

def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _predict():
    arg_parser = predict_argparser()
    args = process_configs(target=__predict, arg_parser=arg_parser)
    __predict(args)


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'test'")
    # args, _ = arg_parser.parse_known_args()
    args, _ = arg_parser.parse_known_args(["predict", "--config", "configs/example_predict.conf"])

    if args.mode == 'train':
        _train()
    elif args.mode == 'test':
        _eval()
    elif args.mode == 'predict':
        _predict()
    else:
        raise Exception("Mode not in ['train', 'test', 'predict'], e.g. 'python spert.py train ...'")
