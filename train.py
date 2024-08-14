import argparse
import torch
import LG3CHiDe
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

log = LG3CHiDe.utils.get_logger()

def main(args):
    LG3CHiDe.utils.set_seed(args.seed)

    log.debug("Loading data from '%s'." % args.data)
    data = LG3CHiDe.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = LG3CHiDe.Dataset(data["train"], args.batch_size)
    devset = LG3CHiDe.Dataset(data["dev"], args.batch_size)
    testset = LG3CHiDe.Dataset(data["test"], args.batch_size)

    log.debug("Building model...")
    model_file = "./save/model.pt"
    model = LG3CHiDe.LGGCN(args).to(args.device)
    opt = LG3CHiDe.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    coach = LG3CHiDe.Coach(trainset, devset, testset, model, opt, args)

    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    log.info("Start training...")
    ret = coach.train()

    checkpoint = {
        "best_dev_f1": ret[0],
        "best_tes_f1": ret[4],
        "test_f1_when_best_dev": ret[3],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")

    # Training parameters
    parser.add_argument("--from_begin", action="store_true",
                        help="Training from begin.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.0003,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.4,
                        help="Dropout rate.")

    # Model parameters
    parser.add_argument("--wp", type=int, default=10,
                        help="Past context window size. Set wp to -1 to use all the past context.")
    parser.add_argument("--wf", type=int, default=10,
                        help="Future context window size. Set wp to -1 to use all the future context.")
    parser.add_argument("--n_speakers", type=int, default=2,
                        help="Number of speakers.")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="transformer",
                        choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    parser.add_argument("--class_weight", action="store_true",
                        help="Use class weights in nll loss.")
    # others
    parser.add_argument("--seed", type=int, default=23,
                        help="Random seed.")
    # 24seed 63
    args = parser.parse_args()
    log.debug(args)

    main(args)
