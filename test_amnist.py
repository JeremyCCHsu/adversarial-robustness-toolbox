from argparse import ArgumentParser

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import torch

from art.attacks import ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.config import ART_DATA_PATH
from art.utils import get_file
from art.attacks import FastGradientMethod

from dev.loaders import AudioMNISTDataset, PreprocessRaw
from dev.transforms import Preprocessor
from hparams import hp

# set seed
np.random.seed(123)
epsilon = .0005


def main(args):
    # load AudioMNIST test set
    audiomnist_test = AudioMNISTDataset(
        root_dir="data/audiomnist/test/",
        transform=PreprocessRaw(),
    )

    # load pretrained model
    model = torch.load(
        args.model_ckpt,
        map_location="cpu",
    )
    model.eval()
    model = model.to("cpu")

    # wrap model in a ART classifier
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, hp.sr],
        nb_classes=10,
    )

    sum_, correct, acc = 0, 0, 0
    n_iter = len(audiomnist_test)
    success = []
    for i in range(n_iter):
        sample = audiomnist_test[i]
        waveform = sample['input']
        label = sample['digit']

        power = waveform.pow(2)
        power = (power.mean() / 1000).sqrt()
        pgd = FastGradientMethod(classifier_art, eps=power.item())

        # craft adversarial example with PGD
        adv_waveform = pgd.generate(x=torch.unsqueeze(
            waveform, 0).numpy())  # , y=np.asarray([1]))
        noise = torch.from_numpy(adv_waveform) - waveform
        snr = 10 * (waveform.pow(2).mean() / noise.pow(2).mean()).log10()

        # evaluate the classifier on the adversarial example
        with torch.no_grad():
            _, pred = torch.max(model(torch.unsqueeze(waveform, 0)), 1)
            _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform)), 1)
        if pred.tolist()[0] == label:
            acc += 1
        if pred.tolist()[0] == pred_adv.tolist()[0]:
            correct += 1
        sum_ += 1
        print((
            f"Processing {i}/{n_iter}: [L={label}|P={pred.tolist()[0]}|A={pred_adv.tolist()[0]}], "
            f"SNR={snr:.2f} attack success rate={1 - correct/sum_:.4f}, acc={acc/sum_:.4f}"),
            end="\r"
        )
        if pred.tolist()[0] != pred_adv.tolist()[0]:
            success.append(i)

    print()

    # =====================================================
    # load a test sample
    sample = audiomnist_test[success[-1]]


    waveform = sample['input']
    label = sample['digit']

    # craft adversarial example with PGD
    power = waveform.pow(2)
    power = (power.mean() / 1000).sqrt()
    pgd = FastGradientMethod(classifier_art, eps=power.item())
    adv_waveform = pgd.generate(
        x=torch.unsqueeze(waveform, 0).numpy()
    )

    # evaluate the classifier on the adversarial example
    with torch.no_grad():
        _, pred = torch.max(model(torch.unsqueeze(waveform, 0)), 1)
        _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform)), 1)

    # print results
    print(f"Original prediction (ground truth):\t{pred.tolist()[0]} ({label})")
    print(f"Adversarial prediction:\t\t\t{pred_adv.tolist()[0]}")

    noise = adv_waveform[0] - waveform.numpy()
    noise = torch.from_numpy(noise)

    adv_waveform = torch.from_numpy(adv_waveform).squeeze(1)

    prep = Preprocessor()
    mel = prep(waveform)
    nel = prep(noise)
    ael = prep(adv_waveform)

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(
        f"Truth: {label}. Prediction: {pred.tolist()[0]}. Corrupted prediction: {pred_adv.tolist()[0]}\n"
        "Top: clean. Middle: noise. Bottom: noisy"
    )
    im = ax[0].imshow(mel[0].numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(nel[0].numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(ael[0].numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[2])
    fig.savefig("model/test-mel.png")

    fig, ax = plt.subplots(3, 1)
    im = ax[0].plot(waveform[0])
    im = ax[1].plot(noise[0])
    im = ax[2].plot(adv_waveform[0])
    fig.savefig("model/test.png")

    write("model/ori.wav", hp.sr, waveform[0].numpy())
    write("model/noise.wav", hp.sr, noise[0].numpy())
    write("model/adv.wav", hp.sr, adv_waveform[0].numpy())
    # =====================================================


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_ckpt", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
