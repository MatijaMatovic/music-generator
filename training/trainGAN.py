from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models.DCGAN import DiscConvNet, GenConvNet
from data_preprocess.data_prep import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def start_training():
    z_dim = 100
    disc = DiscConvNet()
    gen = GenConvNet()
    loader = get_loader()
    disc.apply(weights_init)
    gen.apply(weights_init)
    # gen.load_state_dict(torch.load(os.path.join(root_dir, 'GAN Checkpoints', 'gan_25apr_10_checkpoint_gen_999')))
    # disc.load_state_dict(torch.load(os.path.join(root_dir, 'GAN Checkpoints', 'gan_25apr_10_checkpoint_disc_999')))
    train_full_GAN(gen, disc, loader, z_dim, epochs=1000, start_epoch=0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ConvNet') == -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def progress(batch, loss, batches):
    return f'Training done {batch/batches}%. Current running loss: {loss}'


def sigmoid_cross_entropy_with_logits(inputs, labels):
    loss = nn.BCEWithLogitsLoss()
    output = loss(inputs, labels)
    return output


def train_full_GAN(gen, disc,
                   loader, z_dim,
                   epochs=5, disp_batch_size=24, start_epoch=0):
    gen.to(device).train()
    disc.to(device).train()

    root_dir = 'weights'

    disc_opt = torch.optim.Adam(disc.parameters(), lr=0.0006, betas=(0.0, 0.99))
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0003, betas=(0.0, 0.99))

    fixed_noise = torch.randn(1, z_dim).to(device)

    max_steps = epochs * len(loader)
    #progress_bar = display(progress(0, 0, max_steps), display_id=True)
    gen_losses = []
    disc_losses = []
    steps = 0
    for epoch in range(epochs):
        for i, real in enumerate(loader):
            real = torch.unsqueeze(real, 1)
            real = real.to(device)
            batch_size = len(real)

            # random standard normal noise for generator
            noise = torch.randn(batch_size, z_dim).to(device)

            ### Train Discriminator ###
            # Generator generates a fake image
            fake = gen(noise)

            # Pass the fake and real image to the discriminator
            # Next don't forget to give a detached fake to the discriminator
            # since we do not want to backdrop to generator yet
            disc_fake_pred, disc_fake_pred_sigmoid, fm_fake = disc(fake.detach())
            disc_real_pred, disc_real_pred_sigmoid, fm_real = disc(real)

            # Calculate discriminator loss
            noise = torch.rand_like(disc_real_pred) / 10
            disc_loss_real = sigmoid_cross_entropy_with_logits(disc_real_pred, torch.ones_like(disc_real_pred)).mean()
            noise = torch.rand_like(disc_real_pred) / 10
            disc_loss_fake = sigmoid_cross_entropy_with_logits(disc_fake_pred, torch.zeros_like(disc_fake_pred)).mean()
            disc_loss = (disc_loss_real + disc_loss_fake) / 2

            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()

            ### Train Generator ###
            # for i in range(2): # Potentially train generator multiple times per discriminator train time
            # Get the discriminator's probability for the fake images
            disc_fake_pred, disc_fake_pred_sigmoid, fm_fake = disc(fake)

            # Calculate discriminator loss
            gen_loss = sigmoid_cross_entropy_with_logits(disc_fake_pred, torch.ones_like(disc_fake_pred)).mean()

            # Feature matching
            mse_loss = nn.MSELoss(reduction='mean')
            fm_g_loss1 = torch.mul(mse_loss(fake.mean(), real.mean()), 1)
            fm_g_loss2 = torch.mul(mse_loss(fm_fake.mean(), fm_real.mean()), 1)
            # print('gen loss: {}, fm_g_loss1: {}, fm_g_loss2: {}'.format(gen_loss, fm_g_loss1, fm_g_loss2))
            total_gen_loss = gen_loss + fm_g_loss1 + fm_g_loss2

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            print(progress(steps, (gen_losses[-1], disc_losses[-1]), max_steps))
            steps += 1

        ### Visualize the fake images
        if (epoch + 1) % 5 == 0:
            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111)
            # fake = gen(fixed_noise)
            # fake = fake.permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1).transpose(0, 1)
            # # fake = fake.view(1, -1, 360).squeeze(0).transpose(0, 1)
            # fake = fake.detach().cpu().tolist()
            # fake = np.array(fake)
            # ax.imshow(fake, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            # plt.title('Epoch {} Fake'.format(epoch))
            # plt.show()

            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111)
            # fake[fake < 0.7] = 0.0
            # ax.imshow(fake, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            # plt.title('Epoch {} Fake <0.8 set to 0'.format(epoch))
            # plt.show()
            #
            # fig = plt.figure(figsize=(6, 6))
            # ax = fig.add_subplot(111)
            # real = real[0:5, :, :, :]  # Subset only the first 20 samples, only piano part
            # real = real.permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1).transpose(0, 1)
            # real = real.detach().cpu().numpy()
            # ax.imshow(real, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            # # ax.set_aspect(5)
            # plt.title('Epoch {} Real'.format(epoch))
            # plt.show()

            print('Epoch {} at {}'.format(epoch, datetime.now()))
            print(gen_loss.item())
            print(disc_loss.item())

        # Save checkpoints
        #if (epoch + 1) % 500 == 0:
            # save_path = os.path.join(root_dir, 'GAN Checkpoints',
            #                          'gan_25apr_11_checkpoint_gen_{}'.format(epoch + start_epoch))
            # torch.save(gen.state_dict(), save_path)
            # save_path = os.path.join(root_dir, 'GAN Checkpoints',
            #                          'gan_25apr_11_checkpoint_disc_{}'.format(epoch + start_epoch))
            # torch.save(disc.state_dict(), save_path)
            #
            # with open(os.path.join(root_dir, 'GAN Checkpoints', 'gan_25apr_11_gen_loss'), 'w') as outfile:
            #     json.dump(gen_losses, outfile)
            # with open(os.path.join(root_dir, 'GAN Checkpoints', 'gan_25apr_11_disc_loss'), 'w') as outfile:
            #     json.dump(disc_losses, outfile)

    plt.plot(gen_losses, label='Generator loss')
    plt.plot(disc_losses, label='Discriminator loss')
    plt.xlabel('Batches')
    plt.ylabel('Training loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start_training()
