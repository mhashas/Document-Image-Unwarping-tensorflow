import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from parser_options import ParserOptions
from core.trainers.trainer import Trainer
from util.general_functions import print_training_info
from constants import *

def main():
    tf.enable_eager_execution(device_policy=tf.contrib.eager.DEVICE_PLACEMENT_SILENT)
    args = ParserOptions().parse()  # get training options
    print_training_info(args)

    trainer = Trainer(args)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):

        if args.trainval:
            trainer.run_epoch(epoch, split=TRAINVAL)
        else:
            trainer.run_epoch(epoch, split=TRAIN)

            if epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.run_epoch(epoch, split=TEST)

    if not args.trainval:
        trainer.run_epoch(trainer.args.epochs, split=TEST)

    trainer.summary.add_scalar('test/best_result', trainer.best_loss, args.epochs)
    trainer.summary.close_writer()
    trainer.save_network()

if __name__ == "__main__":
    main()