import argparse
import itertools

import cv2
import torch
import torch.nn as nn
import torch.utils.data
import wandb
from torch.utils.data import DataLoader

from cyclegan.data import CycleGANDataset
from cyclegan.model import Discriminator, Generator, debug_sample, weights_init_normal


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


def load_in_gpu(model):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).to("cuda")
    return model.to("cuda")


def train(args):

    if args.wandb_log:
        wandb.init(
            project="cyclegan",
        )

    device = torch.device("cuda")

    train_set = CycleGANDataset(f"{args.dataset_path}/train")
    train_loader = DataLoader(train_set, args.batch, shuffle=True, drop_last=True)

    gen_s_t = Generator()
    gen_t_s = Generator()
    dis_t = Discriminator()
    dis_s = Discriminator()

    weights_init_normal(gen_s_t)
    weights_init_normal(gen_t_s)
    weights_init_normal(dis_t)
    weights_init_normal(dis_s)

    gen_s_t = load_in_gpu(gen_s_t)
    gen_t_s = load_in_gpu(gen_t_s)
    dis_t = load_in_gpu(dis_t)
    dis_s = load_in_gpu(dis_s)

    gen_s_t.train()
    gen_t_s.train()
    dis_t.train()
    dis_s.train()

    criterion = nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_dis_s = torch.optim.Adam(
        dis_s.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_dis_t = torch.optim.Adam(
        dis_t.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    optimizer_gen = torch.optim.Adam(
        itertools.chain(gen_s_t.parameters(), gen_t_s.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    lr_scheduler_dis_s = torch.optim.lr_scheduler.LambdaLR(
        optimizer_dis_s, lr_lambda=LambdaLR(args.epochs, 0, 100).step
    )
    lr_scheduler_dis_t = torch.optim.lr_scheduler.LambdaLR(
        optimizer_dis_t, lr_lambda=LambdaLR(args.epochs, 0, 100).step
    )
    lr_scheduler_gen = torch.optim.lr_scheduler.LambdaLR(
        optimizer_gen, lr_lambda=LambdaLR(args.epochs, 0, 100).step
    )

    y_real = (torch.ones(args.batch, requires_grad=False) * 1.0).to(device)
    y_fake = (torch.ones(args.batch, requires_grad=False) * 0.0).to(device)

    for epoch in range(args.epochs):
        for _, (source, target) in enumerate(train_loader):

            source = source.to(device)
            target = target.to(device)

            optimizer_gen.zero_grad()

            # Generators
            identity_target = gen_s_t(target)
            identity_source = gen_s_t(source)
            fake_target = gen_s_t(source)
            fake_source = gen_t_s(target)
            cycled_source = gen_t_s(fake_target)
            cycled_target = gen_s_t(fake_source)
            identity_loss_t = criterion_identity(identity_target, target) * 5
            identity_loss_s = criterion_identity(identity_source, source) * 5
            gen_loss_s_t = criterion(dis_t(fake_target).view(-1), y_real)
            gen_loss_t_s = criterion(dis_s(fake_source).view(-1), y_real)
            cycle_loss_s = criterion_cycle(source, cycled_source) * 10
            cycle_loss_t = criterion_cycle(target, cycled_target) * 10
            total_cycle_loss = (
                identity_loss_t
                + identity_loss_s
                + gen_loss_s_t
                + gen_loss_t_s
                + cycle_loss_s
                + cycle_loss_t
            )
            total_cycle_loss.backward()
            optimizer_gen.step()

            # Discriminator S
            optimizer_dis_s.zero_grad()
            y_hat_real_t_s = dis_s(source).view(-1)
            y_hat_fake_t_s = dis_s(fake_source.detach()).view(-1)
            loss_dis_real_t_s = criterion(y_hat_real_t_s, y_real)
            loss_dis_fake_t_s = criterion(y_hat_fake_t_s, y_fake)
            loss_dis_s = (loss_dis_real_t_s + loss_dis_fake_t_s) * 0.5
            loss_dis_s.backward()
            optimizer_dis_s.step()

            # Discriminator T
            optimizer_dis_t.zero_grad()
            y_hat_real_s_t = dis_t(target).view(-1)
            y_hat_fake_s_t = dis_t(fake_target.detach()).view(-1)
            loss_dis_fake_s_t = criterion(y_hat_fake_s_t, y_fake)
            loss_dis_real_s_t = criterion(y_hat_real_s_t, y_real)
            loss_dis_t = (loss_dis_real_s_t + loss_dis_fake_s_t) * 0.5
            loss_dis_t.backward()
            optimizer_dis_t.step()

            total = total_cycle_loss.item()
            g_id = identity_loss_s.item() + identity_loss_t.item()
            g_gan = gen_loss_t_s.item() + gen_loss_s_t.item()
            g_cycle = cycle_loss_s.item() + cycle_loss_t.item()
            d = loss_dis_s.item() + loss_dis_t.item()
            print(
                f"""{epoch + 1}/{args.epochs}, Total: {total:.2f} Id: {g_id:.2f} Gan: {g_gan:.2f} Cycle {g_cycle:.2f} Disc: {d:.2f}"""
            )

            if args.wandb_log:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "Total": total,
                        "Id": g_id,
                        "Gan": g_gan,
                        "Cycle": g_cycle,
                        "Disc": d,
                    }
                )

        with torch.no_grad():
            debug_sample(gen_s_t, "assets/debug_in.jpg")

        if args.wandb_log:
            img = cv2.imread("assets/debug_out.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images = wandb.Image(img, caption="Top: Output, Bottom: Input")
            wandb.log(
                {
                    "debug_out": images,
                }
            )
        lr_scheduler_dis_s.step()
        lr_scheduler_dis_t.step()
        lr_scheduler_gen.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataset_path", type=str, default="dataset/horse2zebra")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--wandb_log", type=bool, default=True)
    args = parser.parse_args()

    train(args)
