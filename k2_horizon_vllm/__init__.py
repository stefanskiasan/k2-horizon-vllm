"""vLLM out-of-tree plugin: registers the native K2-Horizon-MoVA model
plus the IFM reasoning + tool parsers. Loaded in every vLLM process via the
`vllm.general_plugins` entry point, so the parsers are available in the engine
core as well as the API server."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "K2HorizonForCausalLM",
        "k2_horizon_vllm.model:K2HorizonForCausalLM",
    )

    # Register the IFM reasoning ("ifm") and tool ("ifm") parsers in this process.
    try:
        from k2_horizon_vllm import ifm_parsers  # noqa: F401
    except Exception:  # pragma: no cover - parsers are optional at model-load time
        pass
