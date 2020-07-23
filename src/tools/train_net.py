#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: train_net.py
# Author: Owen Lu
# Date:
# Email: jiangxiluning@gmail.com
# Description:
from typing import *
import functools

from loguru import logger
from easydict import EasyDict
import pprint
import tensorflow as tf
from notifiers import get_notifier

from ..dataset.dataset import LmdbDataset
from ..dataset import lmdb_data_generator, benchmark_data_generator
from ..dataset import utils as dataset_utils
from ..model.model import MasterModel
from ..model.metrics import WordAccuary



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def get_optimizer(config):

    if config.train.lr_scheduler.name:
        if config.train.lr_scheduler.name == 'CustomSchedule':
            lr = CustomSchedule(**config.train.lr_scheduler.args)
        else:
            lr = getattr(tf.keras.optimizers.schedules, config.train.lr_scheduler.name)(**config.train.lr_scheduler.args)
    else:
        lr = config.train.optim.args.lr


    if config.train.optim.name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             epsilon=1e-9,
                                             beta_1=0.9,
                                             beta_2=0.98)
    elif config.train.optim.name == 'Adadelta':
        #optimizer = tf.optimizers.Adadelta(learning_rate=lr, rho=0.9, epsilon=1e-6)
        #optimizer = tf.optimizers.Nadam(learning_rate=lr)
        raise NotImplementedError(r'Adadelta is currently not supported. Due the issue https://github.com/tensorflow/tensorflow/issues/38779')

    elif config.train.optim.name == 'SGD':
        optimizer = tf.optimizers.SGD(learning_rate=lr, momentum=0.9)

    else:
        raise RuntimeError('Optimizer {} is not supported.'.format(config.train.optim.name))

    return optimizer

def get_dataset(config: EasyDict):
    dataset_config = config.dataset
    train_config = config.train
    test_config = config.eval

    train_ds = LmdbDataset(lmdb_generator=benchmark_data_generator.generator_lmdb,
                           lmdb_paths=dataset_config.train.datasets,
                           rgb=False,
                           image_width=dataset_config.train.width,
                           image_height=dataset_config.train.height,
                           batch_size=train_config.batch_size,
                           workers=train_config.loader_workers)

    eval_datasets = dict()
    for key, lmdb_path in dataset_config.eval.datasets.items():
        eval_ds = LmdbDataset(lmdb_generator=benchmark_data_generator.generator_lmdb,
                              lmdb_paths=lmdb_path,
                              rgb=False,
                              image_width=dataset_config.eval.width,
                              image_height=dataset_config.eval.height,
                              batch_size=test_config.batch_size,
                              workers=test_config.loader_workers)
        eval_datasets[key]=eval_ds


    return train_ds, eval_datasets

def train(config: EasyDict):
    pprint.pprint(config, indent=2, compact=True)

    # writer = tf.summary.create_file_writer(config.system.outputs.tb_log_dir)
    notifier = get_notifier('slack')
    notifier = functools.partial(notifier.notify, webhook_url=config.system.slack_api)


    if config.system.debug:
        mirroed_strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
    else:
        mirroed_strategy = tf.distribute.MirroredStrategy()


    with mirroed_strategy.scope():
        train_ds, eval_datasets = get_dataset(config)
        logger.info('Training samples are {}'.format(train_ds.num_samples))

        model = MasterModel(config.model,
                            dataset_utils.LabelTransformer.nclass,
                            (config.dataset.train.width, config.dataset.train.height))

        optimizer = get_optimizer(config)

        step = tf.Variable(1)
        epoch = tf.Variable(1)
        best_score = tf.Variable(0, dtype=tf.float32)
        checkpoint = tf.train.Checkpoint(step=step,
                                         epoch=epoch,
                                         optimizer=optimizer,
                                         model=model,
                                         best_score=best_score)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                        directory=config.system.outputs.checkpoints,
                                                        max_to_keep=10,
                                                        step_counter=step,
                                                        checkpoint_name=config.train.checkpoints.name)

    # with mirroed_strategy.scope():
    #     model(tf.random.uniform(shape=(1, config.dataset.train.height, config.dataset.train.width, 1)),
    #           tf.random.uniform(shape=[1, dataset_utils.LabelTransformer.max_length],
    #                             minval=0,
    #                             maxval=len(dataset_utils.LabelTransformer.dict.keys())-1,
    #                             dtype=tf.int32))
    #     model.summary()

    with mirroed_strategy.scope():
        if config.train.checkpoints.finetune:
            status = checkpoint.restore(config.train.checkpoints.finetune)
            status.expect_partial()
            logger.info("Restored from {}".format(config.train.checkpoints.finetune))

            optimizer = get_optimizer(config)
            step = tf.Variable(1)
            epoch = tf.Variable(1)
            best_score = tf.Variable(0, dtype=tf.float32)
        else:
            if config.train.checkpoints.resume:
                if checkpoint_manager.latest_checkpoint:
                    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
                    status.expect_partial()
                    logger.info("Restored from latest checkpoint {}".format(checkpoint_manager.latest_checkpoint))
                else:
                    logger.info("Initializing from scratch.")
            else:
                logger.info("Initializing from scratch.")


    test_acc_dict = dict()
    for key, ds in eval_datasets.items():
        test_acc_dict[key] = WordAccuary()

    test_loss_dict = dict()
    with mirroed_strategy.scope():
        for key, ds in eval_datasets.items():
            test_loss_dict[key] = tf.keras.metrics.Mean(name='test_{}_loss'.format(key))


    def train_step(batch):
        #images = batch[0]
        transcipts = batch[1]

        with tf.GradientTape() as tape:
            logits = model(batch, training=True)

            flatten_gt = tf.reshape(transcipts[:, 1:], shape=(-1, 1))
            flatten_pred = tf.reshape(logits, shape=(-1, logits.shape[-1]))

            # prob = tf.nn.softmax(logits, axis=-1)
            # prob = tf.argmax(prob, axis=-1, output_type=flatten_gt.dtype)
            # prob = tf.reshape(prob, shape=(-1,1))
            # tf.print(tf.reduce_sum(tf.cast(flatten_gt==prob, dtype=tf.int32)))

            loss = tf.keras.losses.sparse_categorical_crossentropy(flatten_gt, flatten_pred, from_logits=True)
            loss_mask = tf.cast(tf.math.not_equal(flatten_gt, dataset_utils.LabelTransformer.dict['<PAD>']), dtype=loss.dtype)
            loss_mask = tf.squeeze(loss_mask)
            replica_loss = tf.reduce_sum(loss * loss_mask) / tf.reduce_sum(loss_mask)

        grads = tape.gradient(replica_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss * loss_mask, loss_mask

    def eval_step(batch):
        images = batch[0]
        transcripts = batch[1]

        pred, logits = model.decode(images)
        flatten_gt = tf.reshape(transcripts[:, 1:], shape=(-1, 1))
        flatten_pred = tf.reshape(logits, shape=(-1, logits.shape[-1]))
        #flatten_pred = tf.reshape(pred, shape=(-1, 1))

        loss = tf.keras.losses.sparse_categorical_crossentropy(flatten_gt, flatten_pred, from_logits=True)
        #loss = tf.cast(flatten_gt == flatten_pred, dtype=pred.dtype)
        loss_mask = tf.cast(flatten_gt != dataset_utils.LabelTransformer.dict['<PAD>'], dtype=loss.dtype)
        loss_mask = tf.squeeze(loss_mask)

        return loss*loss_mask, loss_mask, pred, transcripts


    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses, per_replica_losses_mask  = mirroed_strategy.run(train_step, args=(dataset_inputs, ))
        all_losses = mirroed_strategy.experimental_local_results(per_replica_losses)
        all_losses_mask = mirroed_strategy.experimental_local_results(per_replica_losses_mask)
        all_losses_masked = tf.reduce_sum(tf.concat(all_losses, axis=0))
        all_total_num = tf.reduce_sum(tf.concat(all_losses_mask, axis=0))

        return all_losses_masked / all_total_num


    @tf.function
    def distributed_eval_step(dataset_inputs):
        per_replica_losses, per_replica_losses_mask, pred, gt  = mirroed_strategy.run(eval_step, args=(dataset_inputs, ))
        all_losses = mirroed_strategy.experimental_local_results(per_replica_losses)
        all_losses_mask = mirroed_strategy.experimental_local_results(per_replica_losses_mask)
        all_losses_masked = tf.reduce_sum(tf.concat(all_losses, axis=0))
        all_total_num = tf.reduce_sum(tf.concat(all_losses_mask, axis=0))

        pred = mirroed_strategy.experimental_local_results(pred)
        gt = mirroed_strategy.experimental_local_results(gt)

        pred = tf.concat(pred, axis=0)
        gt = tf.concat(gt, axis=0)

        return all_losses_masked / all_total_num, pred, gt

    # train_ds = eval_datasets['iiit5k']

    for current_epoch in tf.range(epoch, config.train.epochs+1):
        for batch in train_ds:
            current_step = int((step.numpy() - 1) % train_ds.steps) + 1

            loss_batch = distributed_train_step(batch)

            # loss_batch_eval, pred, gt = distributed_eval_step(batch)
            # pred: tf.Tensor = dataset_utils.LabelTransformer.decode_tensor(pred.numpy())
            # gt: tf.Tensor = dataset_utils.LabelTransformer.decode_tensor(gt.numpy())

            # logging training
            if ((current_step) % config.train.log_interval == 0):
                logger.info('''Epoch {current_epoch}/{epochs} Step {current_step}/{steps} Loss: {loss_batch}'''.format(
                    current_epoch=current_epoch,
                    epochs=config.train.epochs,
                    current_step=current_step,
                    steps=train_ds.steps,
                    loss_batch=loss_batch
                ))

            # eval phase
            if (((current_step) % config.eval.interval_iter == 0) or \
                ((current_step) % config.eval.interval_iter == 0)):

                for key, eval_dataset in eval_datasets.items():
                    acc = test_acc_dict[key]
                    all_loss = list()
                    for index, batch in enumerate(eval_dataset):
                        loss_batch, pred, gt = distributed_eval_step(batch)

                        pred: List = dataset_utils.LabelTransformer.decode_tensor(pred.numpy())
                        gt: List = dataset_utils.LabelTransformer.decode_tensor(gt.numpy())
                        acc.update(pred, gt)

                        all_loss.append(loss_batch)
                    all_loss = tf.reduce_mean(tf.concat(all_loss, axis=0))
                    logger.info('Eval {} acc: {} loss: {}'.format(key, acc.compute(), all_loss))

                    if key == 'iiit5k':
                        if acc.compute() > float(best_score.numpy()):
                            best_score.assign(acc.compute())
                            logger.info('Best iiit5k model!')
                            saved_path=checkpoint.write(config.system.outputs.checkpoints+'/OCRTransformer-Best')
                            logger.info("Saved best checkpoint for {}: {}".format(int(current_step), saved_path))

                            if config.system.slack_api:
                                notifier(message='Best score: {}'.format(acc.compute()))

                    acc.reset()

                saved_path = checkpoint_manager.save()
                if saved_path:
                    logger.info("Saved checkpoint for {}: {}".format(int(current_step), saved_path))
            step.assign_add(1)
            if (current_step % train_ds.steps) == 0:
                break

        # for key, eval_dataset in eval_datasets.items():
        #     if key != 'iiit5k':
        #         continue
        #
        #     acc = test_acc_dict[key]
        #     all_loss = list()
        #     for index, batch in enumerate(eval_dataset):
        #         loss_batch, pred, gt = distributed_eval_step(batch)
        #
        #         pred: tf.Tensor = dataset_utils.LabelTransformer.decode_tensor(pred)
        #         gt: tf.Tensor = dataset_utils.LabelTransformer.decode_tensor(gt)
        #         acc.update(pred.numpy().tolist(), gt.numpy().tolist())
        #
        #         all_loss.append(loss_batch)
        #     all_loss = tf.reduce_mean(tf.concat(all_loss, axis=0))
        #     logger.info('Eval {} acc: {} loss: {}'.format(key, acc.compute(), all_loss))
        #     acc.reset()

        epoch.assign_add(1)








