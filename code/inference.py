import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_url, cached_download, hf_hub_download
import shutil
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from typing_extensions import dataclass_transform
from transformers import CLIPProcessor, FlaxCLIPModel
from IPython.display import display

# TF_CPP_MIN_LOG_LEVEL=0
print(jax.local_device_count())
print(jax.devices())

dalle_mini_files_list = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'special_tokens_map.json', 'enwiki-words-frequency.txt', 'flax_model.msgpack']

vqgan_files_list = ['config.json',  'flax_model.msgpack']

for each_file in dalle_mini_files_list:
   downloaded_file = hf_hub_download("dalle-mini/dalle-mini", filename=each_file)
   target_path = '/home/ec2-user/SageMaker/huggingface-sagemaker/content/dalle-mini/' + each_file
   shutil.copy(downloaded_file, target_path)

for each_file in vqgan_files_list:
   downloaded_file = hf_hub_download("dalle-mini/vqgan_imagenet_f16_16384", filename=each_file)
   target_path = '/home/ec2-user/SageMaker/huggingface-sagemaker/content/dalle-mini/vqgan/' + each_file
   shutil.copy(downloaded_file, target_path)

DALLE_MODEL_LOCATION = '/home/ec2-user/huggingface-sagemaker/dalle_mini/content/dalle-mini'
DALLE_COMMIT_ID = None
model, params = DalleBart.from_pretrained(    
      DALLE_MODEL_LOCATION, revision=DALLE_COMMIT_ID, dtype=jnp.float32, _do_init=False,
)

VQGAN_LOCAL_REPO = '/home/ec2-user/SageMaker/dalle_mini/content/dalle-mini/vqgan'
VQGAN_LCOAL_COMMIT_ID = None
vqgan, vqgan_params = VQModel.from_pretrained(
     VQGAN_LOCAL_REPO, revision=VQGAN_LCOAL_COMMIT_ID, _do_init=False
)


print(model.config)
print(vqgan.config)

DALLE_MODEL_LOCATION = '/home/ec2-user/SageMaker/dalle_mini/content/dalle-mini'
DALLE_COMMIT_ID = None
processor = DalleBartProcessor.from_pretrained(
     DALLE_MODEL_LOCATION, 
     revision=DALLE_COMMIT_ID)

print(processor)

# # Works for all available devices to replicate the module
from flax.jax_utils import replicate
import random

params = replicate(params)
vqgan_params = replicate(vqgan_params)

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
  return model.generate(
      **tokenized_prompt,
      prng_key=key,
      params=params,
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      condition_scale=condition_scale,
  )

#decode the images
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)


# entering the prompts
prompts = [
    "sunset over a lake in the mountains",
    "the Eiffel tower landing on the moon",
]

tokenized_prompts = processor(prompts)
tokenized_prompt = replicate(tokenized_prompts)



# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)


n_predictions = 4

# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

print(f"Prompts: {prompts}\n")

images = []
for i in trange(max(n_predictions // jax.device_count(), 1)):
    # get a new key
    key, subkey = jax.random.split(key)
    # generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        images.append(img)
        display(img)