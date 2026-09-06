"""vLLM out-of-tree plugin: registers the native K2-Horizon-MoVA model."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "K2HorizonForCausalLM",
        "k2_horizon_vllm.model:K2HorizonForCausalLM",
    )
