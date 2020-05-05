from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pathlib import Path

from art.classifiers import PyTorchClassifier
from art.attacks import ProjectedGradientDescent, FastGradientMethod

from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.transforms import Preprocessor
from hparams import hp


# set seed
NUM_CLASS = 251
np.random.seed(123)
epsilon = .0005
Fs = 16000
noise_level = 1 / 1000  # 30 dB

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    resolver = LibriSpeechSpeakers("/data/speech")

    # load AudioMNIST test set
    dataset = LibriSpeech4SpeakerRecognition(
        root="/data/speech",
        subset="test",
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
        input_shape=[1, hp.sr],
        nb_classes=NUM_CLASS,
    )

    sum_, correct, acc = 0, 0, 0
    n_iter = len(loader)
    success = []

    meta = open(args.output_dir / "meta.tsv", "w")
    meta.write("Index\tTrueGender\tPrediction\tPredictedGender\tSource\n")
    vec = open(args.output_dir / "vec.tsv", "w")
    # for i in range(n_iter):
    for i, (waveform, label) in enumerate(loader, 1):
        label = label.item()
        power = (waveform.pow(2).mean() * noise_level).sqrt().item()
        pgd = ProjectedGradientDescent(classifier_art, eps=power, eps_step=power / 5)

        # craft adversarial example with PGD
        adv_waveform = pgd.generate(waveform)
        noise = torch.from_numpy(adv_waveform) - waveform
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

            adv_emb = model.encode(torch.from_numpy(adv_waveform).to(device))
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

        # viz vec = adv-emb, meta = (adv filename, label, pred, adv-pred, gender, adv-gen)
        
        

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
                adv_waveform[0, 0]
            )
    print()

    meta.close()
    vec.close()

    # # =====================================================
    # # load a test sample
    # # waveform, label = dataset[success[-1]]
    # # waveform, label = sample

    # # waveform = sample['input']
    # # label = sample['digit']

    # # craft adversarial example with PGD
    # power = waveform.pow(2)
    # power = (power.mean() / 1000).sqrt()
    # pgd = FastGradientMethod(classifier_art, eps=power.item())
    # adv_waveform = pgd.generate(
    #     x=torch.unsqueeze(waveform, 0).numpy()
    # )

    # # evaluate the classifier on the adversarial example
    # with torch.no_grad():
    #     _, pred = torch.max(model(torch.unsqueeze(waveform, 0)), 1)
    #     _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform)), 1)

    # # print results
    # print(f"Original prediction (ground truth):\t{pred.tolist()[0]} ({label})")
    # print(f"Adversarial prediction:\t\t\t{pred_adv.tolist()[0]}")

    # noise = adv_waveform[0] - waveform.numpy()
    # noise = torch.from_numpy(noise)

    # adv_waveform = torch.from_numpy(adv_waveform).squeeze(1)

    # prep = Preprocessor()
    # mel = prep(waveform)
    # nel = prep(noise)
    # ael = prep(adv_waveform)

    # fig, ax = plt.subplots(3, 1)
    # ax[0].set_title(
    #     f"Truth: {label}. Prediction: {pred.tolist()[0]}. Corrupted prediction: {pred_adv.tolist()[0]}\n"
    #     "Top: clean. Middle: noise. Bottom: noisy"
    # )
    # im = ax[0].imshow(mel[0].numpy(), origin="lower", aspect="auto")
    # fig.colorbar(im, ax=ax[0])
    # im = ax[1].imshow(nel[0].numpy(), origin="lower", aspect="auto")
    # fig.colorbar(im, ax=ax[1])
    # im = ax[2].imshow(ael[0].numpy(), origin="lower", aspect="auto")
    # fig.colorbar(im, ax=ax[2])
    # fig.savefig("model/test-mel.png")

    # fig, ax = plt.subplots(3, 1)
    # im = ax[0].plot(waveform[0])
    # im = ax[1].plot(noise[0])
    # im = ax[2].plot(adv_waveform[0])
    # fig.savefig("model/test.png")

    # write("model/ori.wav", hp.sr, waveform[0].numpy())
    # write("model/noise.wav", hp.sr, noise[0].numpy())
    # write("model/adv.wav", hp.sr, adv_waveform[0].numpy())
    # # =====================================================


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_ckpt", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    main(parse_args())
