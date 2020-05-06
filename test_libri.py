from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pathlib import Path

from art.classifiers import PyTorchClassifier
from art import attacks

from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.transforms import Preprocessor
from hparams import hp


# set seed
NUM_CLASS = 251
np.random.seed(123)
epsilon = .0005
Fs = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    resolver = LibriSpeechSpeakers("/data/speech")
    attacker = getattr(attacks, args.attack)

    # load AudioMNIST test set
    dataset = LibriSpeech4SpeakerRecognition(
        root="/data/speech",
        subset="test",
        train_speaker_ratio=1,      # NOTE: make sure it is consistent with training
        train_utterance_ratio=0.9,  # NOTE: make sure it is consistent with training
        project_fs=Fs,  # FIXME: unused
        wav_length=None,
        url='train-clean-100'
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # load pretrained model
    model = torch.load(
        args.model_ckpt,
        # map_location="cpu",
    )
    model.eval()
    model = model.to(device)

    # wrap model in a ART classifier
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, 5 * hp.sr],  # FIXME
        nb_classes=NUM_CLASS,
    )

    sum_, correct, acc = 0, 0, 0
    n_iter = len(loader)
    success = []

    meta = open(args.output_dir / "meta.tsv", "w")
    meta.write("Index\tTrueGender\tPrediction\tPredictedGender\tSource\n")
    
    vec = open(args.output_dir / "vec.tsv", "w")
    
    for i, (waveform, label) in enumerate(loader, 1):
        label = label.item()

        if args.epsilon is not None:
            eps = args.epsilon
        else:
            eps = (waveform.pow(2).mean() / np.power(10, args.snr / 10)).sqrt().item()

        pgd = attacker(classifier_art, eps=eps, eps_step=eps / 5)  # TODO ad-hoc

        # craft adversarial example with PGD
        adv_waveform = pgd.generate(waveform)
        adv_waveform = torch.from_numpy(adv_waveform)
        noise = adv_waveform - waveform
        snr = 10 * (waveform.pow(2).mean() / noise.pow(2).mean()).log10()


        # evaluate the classifier on the adversarial example
        with torch.no_grad():
            stem = f"{i:04d}-{label:03d}\t{resolver.get_gender(label)}"

            emb = model.encode(waveform.to(device))
            logits = model.predict_from_embeddings(emb)
            # _, pred = torch.max(model(waveform.to(device)), 1)
            pred = logits.argmax(-1)
            pred = pred.tolist()[0]
            vec.write("\t".join([f"{x.item():.8f}" for x in emb[0]]) + "\n")
            meta.write(
                f"{stem}\t{pred:03d}\t{resolver.get_gender(pred)}\toriginal\n")

            adv_emb = model.encode(adv_waveform.to(device))
            adv_logits = model.predict_from_embeddings(adv_emb)
            # _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform).to(device)), 1)
            pred_adv = adv_logits.argmax(-1)
            pred_adv = pred_adv.tolist()[0]
            vec.write("\t".join([f"{x.item()}" for x in adv_emb[0]]) + "\n")
            meta.write(
                f"{stem}\t{pred_adv:03d}\t{resolver.get_gender(pred_adv)}\tadversarial\n")


        if pred == label:
            acc += 1
        
        if pred == pred_adv:
            correct += 1

        if pred != pred_adv:
            success.append(i)

        sum_ += 1

        print((
            f"Processing {i}/{n_iter}: [L={label:3d}|P={pred:3d}|A={pred_adv:3d}], "
            f"SNR={snr:.2f} dB, Attack Success Rate={1 - correct/sum_:.4f}, acc={acc/sum_:.4f}"),
            end="\r"
        )
        
        # write("model/ori.wav", hp.sr, waveform[0, 0].numpy())
        # write("model/noise.wav", hp.sr, noise[0, 0].numpy())
        if args.output_dir is not None:
            write(
                args.output_dir / f"adv-{i:04d}-true-{label:03d}--prediction-{pred_adv:03d}.wav",
                Fs,
                adv_waveform[0, 0].detach().cpu().numpy()
            )

    print()

    meta.close()
    vec.close()


    # # =====================================================
    prep = Preprocessor()
    mel = prep(waveform[0])
    nel = prep(noise[0])
    ael = prep(adv_waveform[0])

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(
        f"Truth: {label}. Prediction: {pred}. Corrupted prediction: {pred_adv}\n"
        "Top: clean. Middle: noise. Bottom: noisy"
    )
    im = ax[0].imshow(mel[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(nel[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(ael[0].detach().cpu().numpy(), origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax[2])
    fig.savefig(args.output_dir / f"{stem}-mel.png")

    fig, ax = plt.subplots(3, 1)
    im = ax[0].plot(waveform[0, 0].detach().cpu().numpy())
    im = ax[1].plot(noise[0, 0].detach().cpu().numpy())
    im = ax[2].plot(adv_waveform[0, 0].detach().cpu().numpy())
    fig.savefig(args.output_dir / f"{stem}-wav.png")

    write(args.output_dir / f"{stem}-noise.wav", hp.sr, noise[0, 0].detach().cpu().numpy())
    # write(args.output_dir / "ori.wav", hp.sr, waveform[0].numpy())
    # write(args.output_dir / "adv.wav", hp.sr, adv_waveform[0].numpy())
    # # =====================================================


def parse_args():
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset")
    parser.add_argument("-m", "--model_ckpt", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None, required=True)
    parser.add_argument("-a", "--attack", choices=dir(attacks), default="FastGradientMethod")
    parser.add_argument("-e", "--epsilon", type=float, default=None)
    parser.add_argument("-s", "--snr", type=float, default=None, help="signal-to-noise ratio (in decibel)")
    args = parser.parse_args()

    assert not (args.epsilon is None and args.snr is None), "Set either `epsilon` or `snr`"

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    main(parse_args())
