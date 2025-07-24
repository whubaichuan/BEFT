import argparse
import os
import logging

from utils import setup_logging
from BEFT_evaluator import GLUEvaluator, set_seed

import sys

setup_logging()
LOGGER = logging.getLogger(__file__)

PADDING = "max_length"
MAX_SEQUENCE_LEN = 128

RAND_UNIFORM_MASK_SIZE = {'bert-base-cased': 100000, 'bert-large-cased': 280000, 'roberta-base': 105000}


def _parse_args():
    parser = argparse.ArgumentParser(description='BEFT evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output-path', '-o', required=False, type=str, default='./results',
                        help='output directory path for evaluation products.')
    parser.add_argument('--task-name', '-t', required=False, type=str, default='sst2', help='GLUE/SuperGLUE task name for evaluation.',
                        choices={'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli','cb','wic'})
    parser.add_argument('--model-name', '-m', type=str, default='bert-base-cased', help='model-name to evaluate with.',
                        choices={'bert-base-cased', 'bert-large-cased', 'roberta-base'})

    parser.add_argument('--training-data-number', type=str, default='gradual', help='how many training data used (all or gradual)',
                        choices={'all', 'gradual'})

    parser.add_argument('--bias-terms-loop', type=bool, default=False, help='whether to loop bias terms for b_v, b_q, b_k')
    parser.add_argument('--fisher', type=bool, default=False, help='whether to calculate the fisher information')
                        
    parser.add_argument('--fine-tune-type', '-f', required=False, type=str, default='bitfit',
                        help='Which fine tuning process to perform, types are the types that were performed.',
                        choices={'full_ft', 'bitfit', 'frozen', 'rand_uniform', 'rand_row_col'})
    parser.add_argument('--bias-terms', metavar='N', type=str, nargs='+', default=['all'],
                        choices={'intermediate', 'key', 'query', 'value', 'output', 'output_layernorm',
                                 'attention_layernorm', 'all','gLN1','gLN2'},
                        help='bias terms, should be given in case --fine-tune-type is bitfit '
                             '(choose \'all\' for all bias terms)')

    parser.add_argument('--gpu-device', '-d', type=int, default=None,
                        help='GPU id for BEFT, if not mentioned will train on CPU.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed value to set.')
    parser.add_argument('--learning-rate', '-l', type=float, help='learning rate for training.')
    parser.add_argument('--epochs', '-e', type=int, default=16, help='number of training epochs.')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='training and evaluation batch size.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices={'adam', 'adamw'})
    parser.add_argument('--save-evaluator', action='store_true', default=False,
                        help='if given, will save the evaluator for later inference/examination.')
    parser.add_argument('--predict-test', action='store_true', default=False,
                        help='if given, will infer on test set using the fine-tuned model (predictions file will be in '
                             'GLUE/SuperGLUE benchmark test server format). Predictions will be saved to output_path.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='if given, will plot a list of trainable weights.')

    return parser.parse_args()


def _validate_args(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.isdir(args.output_path):
        raise ValueError("--output_path must be a path to directory")
    # if len(os.listdir(args.output_path)):
    #     raise ValueError("--output_path directory isn't empty, please supply an empty directory path.")
    if args.fine_tune_type == 'rand_uniform' and args.model_name not in RAND_UNIFORM_MASK_SIZE.keys():
        raise ValueError(f'Currently the rand_uniform fine-tune type is not supported for {args.model_name}.')

def _plot_training_details(args,data_size):
    [LOGGER.info('############################################################################################') for _
     in range(3)]
    LOGGER.info('')

    LOGGER.info('Training Details: ')
    LOGGER.info('----------------------------------------------')
    LOGGER.info(f'Model Name: {args.model_name}')
    LOGGER.info(f'Task Name: {args.task_name}')
    LOGGER.info(f'Fine Tuning Type: {args.fine_tune_type}')
    LOGGER.info(f'Output Directory: {args.output_path}')

    if args.gpu_device is not None:
        LOGGER.info(f'Running on GPU #{args.gpu_device}')
    else:
        LOGGER.info(f'Running on CPU')

    LOGGER.info(f'Training Data Number: {args.training_data_number}')

    if args.fine_tune_type == 'bitfit':
        LOGGER.info(f"Bias Trainable Terms: {'all bias terms' if 'all' in args.bias_terms else args.bias_terms}")

    lr_bert_base = {'sst2':4e-4}

    LOGGER.info(f'Epochs: {args.epochs}')
    if args.fine_tune_type == 'bitfit':
        args.learning_rate = lr_bert_base[args.task_name]
    elif args.fine_tune_type == 'full_ft':
        args.learning_rate = 1e-5

    if data_size =='all':
        LOGGER.info('training data size: all')
    else:
        LOGGER.info(f'training data size: {data_size}')
    LOGGER.info(f'Learning Rate: {args.learning_rate}')
    LOGGER.info(f'Batch Size: {args.batch_size}')
    LOGGER.info(f"Optimizer: {'AdamW' if args.optimizer == 'adamw' else 'Adam'}")

    LOGGER.info('')
    [LOGGER.info('############################################################################################') for _
     in range(3)]


def _perform_training_preparations(evaluator, args, trainable_components):
    if args.fine_tune_type == 'frozen':
        trainable_components = []

    if args.fine_tune_type == 'full_ft':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       optimizer=args.optimizer,
                                       encoder_trainable=True,
                                       verbose=args.verbose)
    elif args.fine_tune_type in {'bitfit', 'frozen'}:
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       optimizer=args.optimizer,
                                       encoder_trainable=False,
                                       trainable_components=trainable_components,
                                       verbose=args.verbose)
    else:
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       optimizer=args.optimizer,
                                       encoder_trainable=True,
                                       verbose=False)

        # randomizing mask
        if args.fine_tune_type == 'rand_uniform':
            evaluator.set_uniform_mask(mask_size=RAND_UNIFORM_MASK_SIZE[args.model_name])
        else:  # args.fine_tune_type == 'rand_row_col'
            evaluator.set_row_and_column_random_mask()


def main(args):
    # args parsing
    _validate_args(args)

    if args.training_data_number == 'gradual':
        train_size_loop = [1000] # sst2

        for train_size in train_size_loop:
            _plot_training_details(args,train_size)
            # seed
            set_seed(args.seed)

            # evaluator creation
            evaluator = GLUEvaluator(args.task_name, args.model_name, args.gpu_device,str(int(train_size)),args.fine_tune_type,args.bias_terms)

            evaluator.preprocess_dataset(PADDING, MAX_SEQUENCE_LEN, args.batch_size,train_size=train_size)

            # training preparation
            trainable_components = GLUEvaluator.convert_to_actual_components(args.bias_terms)
            _perform_training_preparations(evaluator, args, trainable_components)

            if args.fisher == True:
                evaluator.fisher_information(args,trainable_components,os.path.join(args.output_path, 'fisher_changes_'+str(train_size)))
                sys.exit()
            # train and evaluate
            evaluator.train_and_evaluate(args.epochs, args.output_path)

            # saving artifacts
            if not args.bias_terms_loop:
                evaluator.plot_terms_changes(os.path.join(args.output_path, 'bias_term_changes_'+args.fine_tune_type+'_'+args.task_name+'_'+str(int(train_size))))
                evaluator.plot_terms_angles(os.path.join(args.output_path, 'ours_term_changes_'+args.fine_tune_type+'_'+args.task_name+'_'+str(int(train_size))))

            # save model
            if args.save_evaluator:
                evaluator.save(os.path.join(args.output_path, 'evaluator'))

            # export model test set predictions
            if args.predict_test:
                evaluator.export_model_test_set_predictions(args.output_path)

if __name__ == '__main__':

    args = _parse_args()
    bias_terms_loop = ['value','query','key'] 
    if args.bias_terms_loop:
        for bias_terms in bias_terms_loop:
            args.bias_terms = [bias_terms]
            main(args)
    elif not args.bias_terms_loop:
        main(args)

