import torch
import hashlib
import os
import urllib
import warnings
import yaml
from tqdm import tqdm
from model.clip.clip_model import build_model


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/"
    + "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/"
    + "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/"
    + "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/"
    + "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/"
    + "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/"
    + "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/"
    + "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/"
    + "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/"
    + "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/"
    + "ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, "
                + "but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def load_model(name: str, download_root: str = None):
    if name not in _MODELS:
        raise ValueError(
            f"Model {name} not available. Available models: {', '.join(_MODELS.keys())}"
        )

    if download_root is None:
        download_root = "pretrained_weights"

    model_path = _download(_MODELS[name], download_root)
    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location="cpu")
        except RuntimeError:
            raise NotImplementedError("Only JIT models are supported")

    model, config = build_model(model.state_dict())
    return model, config


def convert_clip_weights():
    for model_name in _MODELS.keys():
        print(f"Converting {model_name}")
        model, config = load_model(model_name)
        if "@" in model_name:
            model_name = model_name.replace("@", "-")
        if "/" in model_name:
            model_name = model_name.replace("/", "-")
        save_path = f"pretrained_weights/{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        config["pretrained_weights_path"] = save_path
        with open(f"model/model_configs/{model_name}.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"Saved {model_name}.pth")


def main():
    convert_clip_weights()


if __name__ == "__main__":
    main()
