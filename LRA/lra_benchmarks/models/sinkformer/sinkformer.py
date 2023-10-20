# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sinkformer model."""
from flax import nn
from flax.nn.activation import softmax
from flax.nn.stochastic import make_rng
# import flax.linen as nn

import jax
import jax.numpy as jnp
from jax import random
from jax import lax
from jax.experimental.host_callback import call, id_print

from lra_benchmarks.models.layers import common_layers
import numpy as onp
from collections.abc import Iterable

def M(C, u, v):
    return -C + lax.expand_dims(u, dimensions=(-1,)) + lax.expand_dims(v, dimensions=(-2,))

def SinkhornDistance(c, max_iter=0):
    C = -c
    x_points = C.shape[-2]
    y_points = C.shape[-1]
    batch_size = C.shape[0]
    mu = jnp.full((batch_size, x_points), 1.0 / x_points, dtype=float)
    nu = jnp.full((batch_size, y_points), 1.0 / y_points, dtype=float)

    # if mu.dim() < 2:
    #     mu = mu.view(-1, 1)
    # if nu.dim() < 2:
    #     nu = nu.view(-1, 1)

    u = jnp.zeros_like(mu)
    v = jnp.zeros_like(nu)

    # Sinkhorn iterations
    for i in range(max_iter):
        if i % 2 == 0:
            u = lax.log(mu) - jax.nn.logsumexp(M(C, u, v), axis=-1) + u
        else:
            v = lax.log(nu) - jax.nn.logsumexp(M(C, u, v).transpose((0, 2, 1)), axis=-1) + v

    return jnp.exp(M(C, u, v))


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          broadcast_dropout=True,
                          dropout_rng=None,
                          dropout_rate=0.,
                          deterministic=False,
                          precision=None):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights. This
    function supports multi-dimensional inputs.


    Args:
      query: queries for calculating attention with shape of `[batch_size, dim1,
        dim2, ..., dimN, num_heads, mem_channels]`.
      key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
        ..., dimN, num_heads, mem_channels]`.
      value: values to be used in attention with shape of `[batch_size, dim1,
        dim2,..., dimN, num_heads, value_channels]`.
      dtype: the dtype of the computation (default: float32)
      bias: bias for the attention weights. This can be used for incorporating
        autoregressive mask, padding mask, proximity bias.
      axis: axises over which the attention is applied.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
    """
    assert key.shape[:-1] == value.shape[:-1]
    assert (query.shape[0:1] == key.shape[0:1] and
            query.shape[-1] == key.shape[-1])

    if axis is None:
        axis = tuple(range(1, key.ndim - 2))
    if not isinstance(axis, Iterable):
        axis = (axis,)
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim
    for ax in axis:
        if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
            raise ValueError('Attention axis must be between the batch '
                             'axis and the last-two axes.')
    depth = query.shape[-1]
    n = key.ndim
    # batch_dims is  <bs, <non-attention dims>, num_heads>
    batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
    # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    qk_perm = batch_dims + axis + (n - 1,)
    key = key.transpose(qk_perm)
    query = query.transpose(qk_perm)
    # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
    v_perm = batch_dims + (n - 1,) + axis
    value = value.transpose(v_perm)

    query = query / jnp.sqrt(depth).astype(dtype)
    batch_dims_t = tuple(range(len(batch_dims)))
    attn_weights = lax.dot_general(
        query,
        key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
        precision=precision)

    # apply attention bias: masking, droput, proximity bias, ect.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
    attn_weights = softmax(attn_weights, axis=norm_dims)
    attn_weights = attn_weights.astype(dtype)

    # apply dropout
    if not deterministic and dropout_rate > 0.:
        if dropout_rng is None:
            dropout_rng = make_rng()
        keep_prob = jax.lax.tie_in(attn_weights, 1.0 - dropout_rate)
        if broadcast_dropout:
            # dropout is broadcast across the batch+head+non-attention dimension
            dropout_dims = attn_weights.shape[-(2 * len(axis)):]
            dropout_shape = (tuple([1] * len(batch_dims_t)) + dropout_dims)
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = (keep.astype(attn_weights.dtype) /
                      jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier
    # print(query.shape, key.shape, value.shape)
    # print(attn_weights.shape)

    # Sinkhorn
    # attn_weights_shape = attn_weights.shape
    # attn_weights = attn_weights.reshape(-1, attn_weights_shape[-2], attn_weights_shape[-1])
    # # print(attn_weights.shape)
    # attn_weights = SinkhornDistance(attn_weights)
    # attn_weights = attn_weights * attn_weights.shape[-2]
    # attn_weights = attn_weights.reshape(attn_weights_shape)

    # compute the new values given the attention weights
    wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
    y = lax.dot_general(
        attn_weights,
        value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
        precision=precision)

    # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
    perm_inv = _invert_perm(qk_perm)
    y = y.transpose(perm_inv)
    return y

def _invert_perm(perm):
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
        perm_inv[j] = i
    return tuple(perm_inv)


class SinkformerBlock(nn.Module):
  """Sinkformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            cache=None):
    """Applies SinkformerBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      deterministic: bool, deterministic or not (to apply dropout)
      cache: flax autoregressive cache for fast decoding.

    Returns:
      output after sinkformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs)
    x = nn.SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache,
        attention_fn=dot_product_attention,
    )
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(x)
    y = common_layers.MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    return x + y


class SinkformerEncoder(nn.Module):
  """Sinkformer Model Encoder."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            dtype=jnp.float32,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            learn_pos_emb=False,
            classifier=False,
            classifier_pool='CLS',
            num_classes=10,
            tied_weights=False):
    """Applies Sinkformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      dtype: the dtype of the computation (default: float32)
      num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      learn_pos_emb: boolean, if learn the positional embedding or use the
        sinusoidal positional embedding.
      classifier: boolean, for classification mode (output N-class logits)
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      tied_weights: bool, to tie weights or not.

    Returns:
      output of a sinkformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)

    if classifier and classifier_pool == 'CLS':
      cls = self.param('cls', (1, 1, emb_dim), nn.initializers.zeros)
      cls = jnp.tile(cls, [x.shape[0], 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      max_len += 1
      src_padding_mask = jnp.concatenate(
          [src_padding_mask[:, :1], src_padding_mask], axis=1)

    pe_init = nn.initializers.normal(stddev=0.02) if learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        posemb_init=pe_init,
        max_len=max_len,
        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    if tied_weights:
      encoder = SinkformerBlock.shared(
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          name='encoderblock')
      for _ in range(num_layers):
        x = encoder(x)
    else:
      for lyr in range(num_layers):
        x = SinkformerBlock(
            x,
            qkv_dim=qkv_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dtype=dtype,
            padding_mask=src_padding_mask,
            inputs_segmentation=inputs_segmentation,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            deterministic=not train,
            name=f'encoderblock_{lyr}')

    encoded = nn.LayerNorm(x, dtype=dtype, name='encoder_norm')

    if classifier:
      encoded = common_layers.classifier_head(
          encoded, num_classes, mlp_dim, pooling_mode=classifier_pool)
    return encoded


class SinkformerDualEncoder(nn.Module):
  """Sinkformer Model for Matching (dual encoding) tasks."""

  def apply(self,
            inputs1,
            inputs2,
            vocab_size=None,
            inputs1_positions=None,
            inputs2_positions=None,
            inputs1_segmentation=None,
            inputs2_segmentation=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            classifier=True,
            classifier_pool='CLS',
            num_classes=2,
            interaction=None):
    """Applies Sinkformer model on text similarity.

    A deliberate choice to distinguish this from NLI because
    we may want to do different things to the model later. Dual Encoding
    mode enforces that we do not do cross attention between pairs.

    Args:
      inputs1: input data.
      inputs2: target data.
      vocab_size: size of the input vocabulary.
      inputs1_positions: input subsequence positions for packed examples.
      inputs2_positions: target subsequence positions for packed examples.
      inputs1_segmentation: input segmentation info for packed examples.
      inputs2_segmentation: target segmentation info for packed examples.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      classifier: boolean, to use classifier.
      classifier_pool: str, supports "MEAN", "MAX" pooling.
      num_classes: int, number of classification classes.
      interaction: str, supports "NLI"

    Returns:
      output of a sinkformer decoder.
    """

    encoder = SinkformerEncoder.shared(
        vocab_size=vocab_size,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name='encoder')
    inputs1_encoded = encoder(
        inputs=inputs1,
        inputs_positions=inputs1_positions,
        inputs_segmentation=inputs1_segmentation)
    inputs2_encoded = encoder(
        inputs=inputs2,
        inputs_positions=inputs2_positions,
        inputs_segmentation=inputs2_segmentation)

    encoded = common_layers.classifier_head_dual(
        inputs1_encoded,
        inputs2_encoded,
        num_classes,
        mlp_dim,
        pooling_mode=classifier_pool,
        interaction=interaction)

    return encoded
