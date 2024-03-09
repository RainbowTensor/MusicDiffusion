import yaml
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
import os

from diffusion.dataset import LakhPrmat2cLMDB
from diffusion.modules import Paella
from diffusion.music_diffusion import MusicDiffusion
from autoencoder.autoencoder import Autoencoder


with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.isdir("models"):
    os.mkdir("models")

autoencoder_config = config["autoencoder"]
diffusion_config = config["music_diffusion"]
train_config = diffusion_config["train"]
logger_kwargs = train_config["logger_kwargs"]

batch_size = train_config["batch_size"]
data_path = train_config["data_path"]
num_epochs = train_config["num_epochs"]
learning_rate = train_config["learning_rate"]
mixed_precision = train_config["mixed_precision"]
gradient_accumulation_steps = train_config["gradient_accumulation_steps"]
save_path = f'./{train_config["save_dir"]}/diffusion_model.pt'

train_dataset = LakhPrmat2cLMDB(data_path)
train_dataset[0]
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

autoencoder = Autoencoder(autoencoder_config)
autoencoder.eval().requires_grad_(False)
diffusion_model = Paella(**diffusion_config["model"])
model = MusicDiffusion(autoencoder, diffusion_model)

total_params = sum(p.numel() for p in model.diffusion_model.parameters())

print(f"Model has {total_params} params")

api = wandb.Api()
artifact = api.artifact(
    'rainbow_tensor/music_diffusion/model-ifzqp7oq:v116', type='model')
artifact.download("./models")
autoencoder_state_dict = torch.load(
    "./models/autoencoder.pt", map_location=torch.device('cpu'))
autoencoder.load_state_dict(autoencoder_state_dict)

loaded_state_dict = None
if train_config["resume"]:
    assert logger_kwargs["id"] is not None, \
        "When resuming training, WandB run ID should be set."

    run = wandb.init(
        config=config,
        project=logger_kwargs["project"],
        id=logger_kwargs["id"],
        resume="allow"
    )

    model_artifact_name = f'model-{run.id}:latest'
    artifact = run.use_artifact(
        f'{run.entity}/{run.project}/{model_artifact_name}',
        type="model"
    )

    artifact_dir = artifact.download(train_config["save_dir"])
    ckpt_path = f'{artifact_dir}/{train_config["save_name"]}.pt'

    loaded_state_dict = torch.load(ckpt_path)
    model.diffusion_model.load_state_dict(
        loaded_state_dict["diffusion_model"], strict=False
    )
    del loaded_state_dict
else:
    assert logger_kwargs["id"] is None, \
        "When creating new WandB run, ID should be empty."

    run = wandb.init(
        config=config,
        project=logger_kwargs["project"],
        id=logger_kwargs["id"],
        resume="allow"
    )

    # update config with new run id
    config["music_diffusion"]["train"]["resume"] = True
    config["music_diffusion"]["train"]["logger_kwargs"]["id"] = run.id

    with open("./config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

optimizer = bnb.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=train_config["lr_warmup_steps"],
#     num_training_steps=(len(train_dataloader) * num_epochs),
# )

# wandb.watch(model, log_freq=20)


def train_loop(model, optimizer, train_dataloader):
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    model.autoencoder.eval().requires_grad_(False)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = model.training_step(batch)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # lr_scheduler.step()

            optimizer.zero_grad()

            accelerator.print(f"Step: {step}, loss: {loss}")

            if step % train_config["checkpoint_every"] == 0 and step != 0:
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {
                    "diffusion_model": unwrapped_model.diffusion_model.state_dict(),
                }
                torch.save(state_dict, save_path)

                artifact = wandb.Artifact(f"model-{run.id}", "model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact, aliases=["best", "latest"])

            if step % 20 == 0:
                wandb.log({
                    "loss": loss
                })


train_loop(model, optimizer, train_dataloader)

if __name__ == "__main__":
    main()
