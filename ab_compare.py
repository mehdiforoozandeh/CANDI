from dataclasses import asdict
from pathlib import Path
from sandbox.config import deep_merge, load_yaml
from sandbox.jepa_config import JEPAConfig
from sandbox.config_types import config_from_dict
from sandbox.train_jepa import _dotted_overrides_from_argv
from sandbox import SANDBOX_ASSAYS
from sandbox.jepa_model import JEPAModel, JEPAModelConfig as FreshJEPAModelConfig, JEPATransformerPredictor, MetadataEmbedding as FreshMetadataEmbedding
from sandbox.model import build_sandbox_candi
from sandbox.jepa import CANDIJepa

base = asdict(JEPAConfig())
base = deep_merge(base, load_yaml(Path("sandbox/configs/jepa_default.yaml")))

common = [
    "training.epochs=1","training.batch_size=2","training.amp=false","training.max_train_batches=1",
    "training.save_checkpoint=false","training.optimizer.name=adamw","training.optimizer.adamw.lr=5e-5",
    "training.optimizer.adamw.weight_decay=1e-2","training.schedule.name=cosine","training.schedule.warmup_frac=0.1",
    "training.schedule.min_lr_ratio=0.1","training.grad.clip_norm=1.0","training.masking.p_full_assay=1.0",
    "training.masking.p_full_loci=0.0","training.masking.p_chunks=0.0","training.masking.min_available_frac=0.3",
    "training.masking.preserve_assay_id=true","training.dsf.dsf_list=1,2,4","training.dsf.sampling=uniform",
    "jepa.proj_dim=0","jepa.proj_hidden_dim=256","jepa.pred_hidden_dim=0","jepa.pred_proj_hidden_dim=0",
    "jepa.lambda_sigreg=0.5","jepa.sigreg_num_proj=1024","jepa.sigreg_knots=17","jepa.predictor_type=fresh_transformer",
    "jepa.pred_cond_source=meta_tgt_embed","jepa.pred_meta_embed_dim=32","jepa.pred_meta_embed_layernorm=false",
    "jepa.predictor_layers=1","jepa.predictor_heads=4","jepa.predictor_dim_head=64","jepa.predictor_ff_mult=4",
    "model.n_cnn_layers=3","model.expansion_factor=2","model.n_transformer_layers=2","model.nhead=4",
    "model.dropout=0.1","model.metadata_embedding_dim_mult=4","model.mask_stem=true","model.encode_input_transform=log1p",
    "fresh.metadata_embed_dim=32","fresh.n_cnn_layers=3","fresh.expansion_factor=2","fresh.conv_kernel_size=3",
    "fresh.pool_size=2","fresh.dna_pool_size=5","fresh.n_transformer_layers=2","fresh.nhead=4","fresh.dropout=0.1",
    "fresh.d_model=0","fresh.proj_dim=0","fresh.proj_hidden_dim=256","fresh.pred_hidden_dim=0","fresh.pred_proj_hidden_dim=0",
    "fresh.predictor_layers=1","fresh.predictor_heads=4","fresh.predictor_dim_head=64","fresh.predictor_ff_mult=4",
    "fresh.predictor_type=transformer","fresh.cond_source=meta_tgt_embed","fresh.cond_embed_shared=separate",
    "fresh.meta_embed_layernorm=true","fresh.pred_meta_embed_layernorm=false","fresh.missing_data_mode=mask_stem",
    "fresh.film_mode=per_conv","fresh.conv_norm=layer","fresh.dna_pool_order=late","fresh.transformer_type=production_dual",
]

def resolve(model_type, run_name):
    cfg_dict = deep_merge(base, _dotted_overrides_from_argv(common + [f"model_type={model_type}", f"wandb.run_name={run_name}"]))
    return config_from_dict(JEPAConfig, cfg_dict)

cfg_c = resolve("candi", "jepa_candi_enc")
cfg_f = resolve("fresh", "jepa_fresh_enc")

signal_dim = len(SANDBOX_ASSAYS)
meta_dim = int(cfg_c.model.metadata_embedding_dim_mult) * signal_dim

candi_core = build_sandbox_candi(
    context_bins=int(cfg_c.data.context_length), signal_dim=signal_dim, metadata_embedding_dim=meta_dim,
    n_cnn_layers=int(cfg_c.model.n_cnn_layers), expansion_factor=int(cfg_c.model.expansion_factor),
    nhead=int(cfg_c.model.nhead), n_sab_layers=int(cfg_c.model.n_transformer_layers), dropout=float(cfg_c.model.dropout),
    separate_decoders=bool(cfg_c.model.separate_decoders), mask_stem=bool(cfg_c.model.mask_stem), dist_type="gaussian",
    signal_transform=str(cfg_c.model.encode_input_transform), linear_film=bool(cfg_c.model.linear_film),
    single_shot_decoder_film=bool(cfg_c.model.single_shot_decoder_film), gaussian_var_min=float(cfg_c.model.gaussian_var_min),
)
encoder_out_dim = int(candi_core.latent_projection[0].in_features)
proj_dim = int(cfg_c.jepa.proj_dim) if int(cfg_c.jepa.proj_dim) > 0 else encoder_out_dim
pred_hidden_dim = int(cfg_c.jepa.pred_hidden_dim) if int(cfg_c.jepa.pred_hidden_dim) > 0 else proj_dim
pred_meta_embedding = FreshMetadataEmbedding(num_assays=signal_dim, embed_dim=int(cfg_c.jepa.pred_meta_embed_dim), use_layernorm=bool(cfg_c.jepa.pred_meta_embed_layernorm))
inj_pred = JEPATransformerPredictor(
    proj_dim=proj_dim, hidden_dim=pred_hidden_dim, cond_dim=(signal_dim + 1) * int(cfg_c.jepa.pred_meta_embed_dim),
    depth=int(cfg_c.jepa.predictor_layers), heads=int(cfg_c.jepa.predictor_heads), dim_head=int(cfg_c.jepa.predictor_dim_head),
    ff_mult=int(cfg_c.jepa.predictor_ff_mult), dropout=float(cfg_c.model.dropout),
)
model_c = CANDIJepa(
    candi_core, proj_dim=int(cfg_c.jepa.proj_dim), proj_hidden_dim=int(cfg_c.jepa.proj_hidden_dim),
    pred_hidden_dim=int(cfg_c.jepa.pred_hidden_dim), pred_proj_hidden_dim=int(cfg_c.jepa.pred_proj_hidden_dim),
    num_assays=signal_dim, use_mask_cond=bool(cfg_c.jepa.pred_use_mask_cond), pred_mask_cond_type=str(cfg_c.jepa.pred_mask_cond_type),
    lambda_sigreg=float(cfg_c.jepa.lambda_sigreg), sigreg_num_proj=int(cfg_c.jepa.sigreg_num_proj),
    sigreg_knots=int(cfg_c.jepa.sigreg_knots), target_dsf=str(cfg_c.jepa.target_dsf), predictor=inj_pred,
    pred_metadata_embedding=pred_meta_embedding,
)

fresh_cfg = FreshJEPAModelConfig(
    num_assays=signal_dim, context_length=int(cfg_f.data.context_length), metadata_embed_dim=int(cfg_f.fresh.metadata_embed_dim),
    n_cnn_layers=int(cfg_f.fresh.n_cnn_layers), expansion_factor=int(cfg_f.fresh.expansion_factor), conv_kernel_size=int(cfg_f.fresh.conv_kernel_size),
    pool_size=int(cfg_f.fresh.pool_size), dna_pool_size=int(cfg_f.fresh.dna_pool_size), n_transformer_layers=int(cfg_f.fresh.n_transformer_layers),
    nhead=int(cfg_f.fresh.nhead), dropout=float(cfg_f.fresh.dropout), d_model=int(cfg_f.fresh.d_model), proj_dim=int(cfg_f.fresh.proj_dim),
    proj_hidden_dim=int(cfg_f.fresh.proj_hidden_dim), pred_hidden_dim=int(cfg_f.fresh.pred_hidden_dim), pred_proj_hidden_dim=int(cfg_f.fresh.pred_proj_hidden_dim),
    predictor_layers=int(cfg_f.fresh.predictor_layers), predictor_heads=int(cfg_f.fresh.predictor_heads), predictor_dim_head=int(cfg_f.fresh.predictor_dim_head),
    predictor_ff_mult=int(cfg_f.fresh.predictor_ff_mult), predictor_type=str(cfg_f.fresh.predictor_type), cond_source=str(cfg_f.fresh.cond_source),
    cond_embed_shared=str(cfg_f.fresh.cond_embed_shared), lambda_sigreg=float(cfg_f.fresh.lambda_sigreg), sigreg_num_proj=int(cfg_f.fresh.sigreg_num_proj),
    sigreg_knots=int(cfg_f.fresh.sigreg_knots), signal_transform=str(cfg_f.model.encode_input_transform), meta_embed_layernorm=bool(cfg_f.fresh.meta_embed_layernorm),
    pred_meta_embed_layernorm=bool(cfg_f.fresh.pred_meta_embed_layernorm), missing_data_mode=str(cfg_f.fresh.missing_data_mode),
    film_mode=str(cfg_f.fresh.film_mode), conv_norm=str(cfg_f.fresh.conv_norm), dna_pool_order=str(cfg_f.fresh.dna_pool_order), transformer_type=str(cfg_f.fresh.transformer_type),
)
model_f = JEPAModel(fresh_cfg)

print("=== RUN NAMES ===")
print(cfg_c.wandb.run_name, cfg_f.wandb.run_name)
print("\n=== MATCHED EFFECTIVE SETTINGS ===")
for k, a, b in [
    ("predictor_class", type(model_c.jepa_predictor).__name__, type(model_f.jepa_predictor).__name__),
    ("predictor_layers", cfg_c.jepa.predictor_layers, cfg_f.fresh.predictor_layers),
    ("predictor_heads", cfg_c.jepa.predictor_heads, cfg_f.fresh.predictor_heads),
    ("predictor_dim_head", cfg_c.jepa.predictor_dim_head, cfg_f.fresh.predictor_dim_head),
    ("predictor_ff_mult", cfg_c.jepa.predictor_ff_mult, cfg_f.fresh.predictor_ff_mult),
    ("encoder_n_cnn_layers", cfg_c.model.n_cnn_layers, cfg_f.fresh.n_cnn_layers),
    ("encoder_n_transformer_layers", cfg_c.model.n_transformer_layers, cfg_f.fresh.n_transformer_layers),
    ("encoder_nhead", cfg_c.model.nhead, cfg_f.fresh.nhead),
    ("encoder_dropout", cfg_c.model.dropout, cfg_f.fresh.dropout),
    ("encoder_meta_embed_dim", int(cfg_c.model.metadata_embedding_dim_mult) * signal_dim, cfg_f.fresh.metadata_embed_dim),
    ("input_transform", cfg_c.model.encode_input_transform, cfg_f.model.encode_input_transform),
    ("sigreg_lambda", cfg_c.jepa.lambda_sigreg, cfg_f.fresh.lambda_sigreg),
    ("sigreg_num_proj", cfg_c.jepa.sigreg_num_proj, cfg_f.fresh.sigreg_num_proj),
    ("sigreg_knots", cfg_c.jepa.sigreg_knots, cfg_f.fresh.sigreg_knots),
    ("missing_mode", "mask_stem" if cfg_c.model.mask_stem else "none", cfg_f.fresh.missing_data_mode),
]:
    print(f"{k}: {a} | {b}")

print("\n=== STILL-DIFFERENT BY DESIGN ===")
print("model_type:", cfg_c.model_type, "|", cfg_f.model_type)
print("encoder_class:", type(model_c.candi.encoder).__name__, "|", type(model_f.encoder).__name__)
print("candi_attention_backend: dual (fixed in build_sandbox_candi)")
print("fresh_transformer_type:", cfg_f.fresh.transformer_type)
print("candi_film: per-conv only (fixed)")
print("fresh_film_mode:", cfg_f.fresh.film_mode)
print("candi_pred_mask_cond_type:", cfg_c.jepa.pred_mask_cond_type, "(used)")
print("fresh_mask_cond: ignored in JEPAModel.forward")
print("fresh_cond_source:", model_f.cond_source)
print("fresh_cond_embed_shared:", model_f.cond_embed_shared)
print("fresh_has_separate_pred_meta_embedding:", model_f.pred_metadata_embedding is not None)
