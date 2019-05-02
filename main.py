"""Main script for ADDA."""

import params
import os
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed, print_log

if __name__ == '__main__':
    # init random seed
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.num_gpu)

    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print_log("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print_log("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print_log("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print_log("=== Evaluating classifier for encoded target domain ===")
    print_log(">>> source only final <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print_log(">>> domain adaption final <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)

    for i in range(100, 2000, 100):
        file_name = "/home/hhjung/hhjung/ADDA/ADDA-target-encoder-{}.pt".format(i)
        tgt_encoder = init_model(net=LeNetEncoder(),
            restore=file_name)
        print_log(">>> source only {} <<<".format(i))
        eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
        print_log(">>> domain adaption {} <<<".format(i))
        eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)

