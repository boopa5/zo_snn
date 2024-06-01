import torch
import optimizee
from train import train_update_rnn, train_model_with_optimizer

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available()  else "cpu"

    optimizer_train_args = {
        "num_epoch": 2,
        "updates_per_epoch": 5,
        "optimizer_steps": 200,
        "truncated_bptt_step": 10,
        "batch_size": 32,
        "optimizee": optimizee.mnist.MnistSpikingConvModel
    }

    optimizee_train_args = {
        "num_epoch": 1,
        "batch_size": 128,
        "optimizee": optimizee.mnist.MnistSpikingConvModel
    }

    train_update_rnn(optimizer_train_args, device)
    train_model_with_optimizer(optimizee_train_args, device)