# This file is modified from https://github.com/artidoro/qlora/blob/main/qlora.py 
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np
import importlib
from packaging import version

import torch.nn as nn
import torch.optim as optim

import torch
import transformers
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    LlamaTokenizer
)


from datautils_block import test_ppl
from datautils_e2e import make_data_module
from bitsandbytes.optim import AdamW
import os
import utils
import math
from math import log

from quantize.int_linear_real import load_quantized_model
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import wandb
from quantize.quantizer import UniformAffineQuantizer, RoundUpQuantizer, RoundDownQuantizer
from quantize.int_linear_real import QuantLinear



def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    quant_model_path: Optional[str] = field(
        default="",
        metadata={"help": "path of the quantization model by Block-AP."}
    )
    model_family: Optional[str] = field(
        default="llama-2",
        metadata={"help": "for the saving of dataset cache for faster experiments"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    conv_temp: str = field(
        default='llama-2',
        metadata={"help": "Conversation template, only useful with deita datasets"}
    )
    mask_use: bool = field(
        default=True, metadata={"help": "mask the loss to role in dialogue datas"}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|redpajama]"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_ppl_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    pt_context_len: int = field(
        default=1024,
        metadata={"help": "language modeling length."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    wbits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    group_size: int = field(
        default=64,
        metadata={"help": "How many group size to use."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    resume_from_checkpoint: str = field(default=None, metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=0, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-5, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='cosine', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=5, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

global_device = "cuda:0"
# global_device = "cuda:1"

def get_accelerate_model(args, checkpoint_dir, quant_linear_type=QuantLinear.QuantType.REGULAR):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    print("Loading model from", args.quant_model_path)
    model, tokenizer = load_quantized_model(args.quant_model_path,args.wbits, args.group_size, quant_linear_type)
    tokenizer.model_max_length = args.pt_context_len
    
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))        
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.to(global_device)
    model.train()
        
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )    

    # TODO
    # if 'llama1' in args.model_name_or_path or 'llama2' in args.model_name_or_path or 'llama-1' in args.model_name_or_path or 'llama-2' in args.model_name_or_path:
    if isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })


    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)
            
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)    
            
        model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        # if isinstance(module, QuantLinear):
        #     # transfer trainable step size into float32
        #     module.scales.data = module.scales.data.to(torch.float32)
        if 'norm' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                    # module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*'*80)
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print('*'*80)
    if args.wbits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def copy_dataloader(dataloader):
    return DataLoader(
        dataset=dataloader.dataset,  # Same dataset
        batch_size=dataloader.batch_size,
        shuffle=dataloader.shuffle,
        sampler=dataloader.sampler,  # Ensures same sampling strategy
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context
    )

class EASGD():
    def __init__(self, model_list, alpha=0.001, beta=0.99, stickiness=5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.stickiness = stickiness
        
        self.num_workers = len(model_list) - 1

        self.main_model = model_list[0].to(global_device)
        
        params = []
        params.append({'params': [p for n, p in self.main_model.named_parameters() if 'scale' in n], 'weight_decay': 0.0, 'lr': 2e-5})
        self.main_optimizer = AdamW(params)

        self.models = model_list[1:]
        self.optimizers = []

        for model in self.models:
            model.load_state_dict(self.main_model.state_dict())
            
            params = []
            params.append({'params': [p for n, p in model.named_parameters() if 'scale' in n], 'weight_decay': 0.0, 'lr': 2e-5})
            optimizer = AdamW(params)

            self.optimizers.append(optimizer)
            
    def _initialize_weights(self):
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.main_model.apply(init_func)
        for model in self.models:
            model.load_state_dict(self.main_model.state_dict())

    def synchronize_models(self, batch):
        self.main_model.to(global_device)
        with torch.no_grad():
            for model in self.models:
                for local_param, central_param in zip(model.parameters(), self.main_model.parameters()):
                    diff = local_param.data - central_param.data
                    local_param.data -= self.alpha * diff
                    central_param.data += self.beta * diff
                    
                model.to("cpu", non_blocking=True)
                    
        for model in self.models:
            model.to("cpu")
        
        loss = self.main_model(**batch).loss
        wandb.log({"main_loss": loss.item()}, commit=False)
                    
        self.main_model.to("cpu", non_blocking=True)
        self.models[0].to(global_device)
        for model in self.models[1:]:
            model.to(global_device, non_blocking=True)

    def train_easgd(self, trainloader, testloader):
        
        train_losses = []
        test_losses = []
        model_choice = 0
        stickiness = self.stickiness
        sticky_count = 0

        
        config = {"epochs": 1, 
                  "batch_size": trainloader.batch_size,
                  "alpha": self.alpha,
                  "beta": self.beta,
                  "stickiness": self.stickiness,
                  "explanation": "actual_quantization_hopefully",
                  }
        run_name = f"fq_alpha_{self.alpha}_beta_{self.beta}_stickiness_{self.stickiness}"
        wandb.init(project="train_fq_easgd", name=run_name, config=config)
        
        self._initialize_weights()
                
        self.main_model.to("cpu")
        for model in self.models:
            model.to(global_device)
        
        # for model in self.models:
        #     wandb.watch(model, log="all", log_freq=10)

        for i, batch in enumerate(tqdm(trainloader)):
            model = self.models[model_choice]
            optimizer = self.optimizers[model_choice]

            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = loss.item()
            
            train_loss = train_loss / len(batch["input_ids"])
            train_losses.append(train_loss)
            
            # Free memory
            # del batch, outputs, loss
            # torch.cuda.empty_cache()
            # gc.collect()


            sticky_count += 1
            if sticky_count > stickiness:
                sticky_count = 0
                # self.models[model_choice].to("cpu", non_blocking=True)
                model_choice = (model_choice + 1) % self.num_workers

                if model_choice == 0:
                    self.synchronize_models(batch)            
        
            
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            # print(f"Memory allocated: {mem_allocated:.2f} GB")
            # print(f"Memory reserved: {mem_reserved:.2f} GB")
            
            last_elements = train_losses[-40:] if len(train_losses) >= 40 else train_losses
            avg_loss =  sum(last_elements) / len(last_elements)


            logs = {"loss": train_loss,
                    "avg_loss": avg_loss,
                    "batch": i,
                    "log_batch": log(i + 1, 10), 
                    }
            wandb.log(logs)
            
            # stop early so we can run experiments faster
            if i > 1250:
                break
        
        wandb.finish()

        return train_losses, train_losses
    
    def train_regular(self, trainloader, testloader):
        wandb.init(project="train_regular", config={"epochs": 1, "batch_size": trainloader.batch_size})
        
        self._initialize_weights()
        
        train_losses = []
        test_losses = []
        model_choice = 0
        
        # wandb.watch(self.main_model, log="all", log_freq=10)
        
        for model in self.models:
            model.to("cpu")
    
        model = self.main_model
        optimizer = self.main_optimizer

        for i, batch in enumerate(tqdm(trainloader)):

            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = loss.item()
        
            train_loss = train_loss / len(batch["input_ids"])
            train_losses.append(train_loss)
            
            wandb.log({"batch": i, "loss": train_loss})
        
        wandb.finish()

        return train_losses, train_losses
    
    
    def train_interleave(self, trainloader, regular_model):
        
        params = []
        params.append({'params': [p for n, p in regular_model.named_parameters() if 'scale' in n], 'weight_decay': 0.0, 'lr': 2e-5})
        regular_optimizer = AdamW(params)
        
        wandb.init(project="train_interleave", config={"epochs": 1, "batch_size": trainloader.batch_size})
        
        self._initialize_weights()
        
        train_losses = []
        test_losses = []
        model_choice = 0
        
        self.main_model.to("cpu")
        for model in self.models:
            model.to("cpu")
        self.models[model_choice].to("cuda")
        
        for i, batch in enumerate(tqdm(trainloader)):
            model = self.models[model_choice]
            optimizer = self.optimizers[model_choice]

            # Run and test EASGD model first
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            loss_easgd = loss.item()

            model_choice = (model_choice + 1) % self.num_workers

            model.to("cpu")
            if model_choice == 0:
                self.synchronize_models()
            self.models[model_choice].to("cuda", non_blocking=True)
                
                
            # Run and tes the regular model afterwards
            outputs = regular_model(**batch)
            loss = outputs.loss
            loss.backward()
            regular_optimizer.step()
            regular_optimizer.zero_grad()
            
            loss_reg = loss.item()
            torch.cuda.empty_cache()
        
            loss_easgd = loss_easgd / len(batch["input_ids"])
            loss_reg = loss_reg / len(batch["input_ids"])
            
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Memory allocated: {mem_allocated:.2f} GB")
            print(f"Memory reserved: {mem_reserved:.2f} GB")

                        
            wandb.log({"loss_easgd": loss_easgd, "loss_reg": loss_reg})        
        wandb.finish()

        return train_losses, train_losses
    
from quantize.utils import set_quant_state

def train(alpha=0.001, beta=0.99, stickiness=5):
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(args.output_dir)
    logger.info(args)
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')
    print("args: ", args)
    print("checkpoint_dir: ", checkpoint_dir)
    print("QuantLinear.QuantType.REGULAR: ", QuantLinear.QuantType.REGULAR)
    model1, tokenizer = get_accelerate_model(args, checkpoint_dir, QuantLinear.QuantType.REGULAR)
    model2, _         = get_accelerate_model(args, checkpoint_dir, QuantLinear.QuantType.UP)
    model3, _         = get_accelerate_model(args, checkpoint_dir, QuantLinear.QuantType.DOWN)
    # model4, _ = get_accelerate_model(args, checkpoint_dir)
    
    model2.to("cpu")
    model3.to("cpu")

    # Enabled gradient checkpointing
    model1.config.use_cache = False
    model2.config.use_cache = False
    model3.config.use_cache = False
    # model4.config.use_cache = False
        
    models = nn.ModuleList([
        model1,
        model2,
        model3,
    ])
    
    print('loaded models')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    for model in models:    
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear) and not 'head' in name:
                module.scales.requires_grad = True
        set_quant_state(model, True)


    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.append({'params': [p for n, p in model1.named_parameters() if 'scale' in n], 'weight_decay': 0.0, 'lr': args.learning_rate})
    optimizer = AdamW(optimizer_grouped_parameters)
    # optimizer = EASGDOptimizer()

    trainer = Seq2SeqTrainer(
        model=model1,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, None),
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
            
    train_dataloader = trainer.get_train_dataloader()
    # test_dataloader = trainer.get_test_dataloader()
    trainer.model.train()
    
    assert next(trainer.model.parameters()).is_cuda
    assert train_dataloader is not None
    
    easgd = EASGD(models, alpha, beta, stickiness)
    train_losses, test_losses = easgd.train_easgd(train_dataloader, None)
    # train_losses, test_losses = easgd.train_regular(train_dataloader, None)
    # train_losses, test_losses = easgd.train_interleave(train_dataloader, model4)
    
    # Verifying the datatypes and parameter counts before training.
    # print_trainable_parameters(args, model1)
    # dtypes = {}
    # for _, p in model1.named_parameters():
    #     dtype = p.dtype
    #     if dtype not in dtypes: dtypes[dtype] = 0
    #     dtypes[dtype] += p.numel()
    # total = 0
    # for k, v in dtypes.items(): total+= v
    # for k, v in dtypes.items():
    #     print(k, v, v/total)

    # all_metrics = {"run_name": args.run_name}
    del model1
    del model2
    del model3
    torch.cuda.empty_cache()
    gc.collect()
    print(args.output_dir)
    
    
import itertools
import logging

import smtplib, ssl

def send_email(message, subject=None):
    if subject is None:
        subject = "Crimson Update" 
    if message is None:
        message = "This message is sent from Python."
    
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "ritarka.samanta@gmail.com"  # Enter your address
    receiver_email = "ritarka.samanta@gmail.com"  # Enter receiver address
    password = ""
    message = (
        f"Subject: {subject}\n"
        f"\n"
        f"{message}"
    )

    print(message)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    
if __name__ == "__main__":

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
        
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

        # Cuda 0
        # alpha_list = [0, 0.00005, 0.0001]
        # beta_list = [0.95, 1]
        # stickiness_list = [5, 7, 10]
        
        # Cuda 1
    #alpha_list = [0, 0.000025, 0.000075, 0.0002]      # 4
    #beta_list = [0.95, 1.0, 1.1, 1.2]                 # 4
    #stickiness_list = [5, 9, 12]                      # 4
    alpha_list = [0.000075]      # 4
    beta_list = [0.95]                 # 4
    stickiness_list = [9]                      # 4
        # Total: 4 × 4 × 3 = 64 combinations


    for alpha, beta, stickiness in itertools.product(alpha_list, beta_list, stickiness_list):
        logging.info(f"Running alpha={alpha}, beta={beta}, stickiness={stickiness}")
            
        result = train(alpha=alpha, beta=beta, stickiness=stickiness)
            
        gc.collect()
        torch.cuda.empty_cache()

        logging.info(f"alpha={alpha}, beta={beta}, stickiness={stickiness}")
    
    # try:
    #     import gc
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()
        
    #     import torch.backends.cudnn as cudnn
    #     cudnn.benchmark = True

    #     # Cuda 0
    #     # alpha_list = [0, 0.00005, 0.0001]
    #     # beta_list = [0.95, 1]
    #     # stickiness_list = [5, 7, 10]
        
    #     # Cuda 1
    #     alpha_list = [0, 0.000025, 0.000075, 0.0002]      # 4
    #     beta_list = [0.95, 1.0, 1.1, 1.2]                 # 4
    #     stickiness_list = [5, 9, 12]                      # 4
    #     # Total: 4 × 4 × 3 = 64 combinations


    #     for alpha, beta, stickiness in itertools.product(alpha_list, beta_list, stickiness_list):
    #         logging.info(f"Running alpha={alpha}, beta={beta}, stickiness={stickiness}")
            
    #         result = train(alpha=alpha, beta=beta, stickiness=stickiness)
            
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #         logging.info(f"alpha={alpha}, beta={beta}, stickiness={stickiness}")

    # except Exception as e:
    #     print(f"Error during training: {e}")
    #     # send_email(
    #     #     message=f"Error during training: {e}",
    #     #     subject="Training Error"
    #     # )
