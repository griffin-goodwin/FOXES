"""Tests for the configurable attention masking in model.py (ViTLocal).

Covers the three mask modes ('inverted' = original released behaviour, 'local',
'none') and, most importantly, that the change does NOT break existing
checkpoints:

  * default mode reproduces the original inverted mask bit-for-bit;
  * old checkpoints that saved the 'attention_mask' buffer still load strictly
    and produce identical outputs;
  * new checkpoints no longer carry the mask and round-trip cleanly.

Run with:
    python -m unittest forecasting.model_test -v
"""
import os
# Work around the duplicate-OpenMP runtime warning on this machine; must be set
# before torch is imported.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import math
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (  # noqa: E402
    InvertedAttentionBlock,
    VisionTransformerLocal,
    ViTLocal,
)


def original_inverted_mask(num_patches, local_window):
    """Reference implementation: the exact pre-change double-loop mask."""
    grid_size = int(math.sqrt(num_patches))
    mask = torch.zeros(num_patches, num_patches)
    for i in range(num_patches):
        row_i, col_i = i // grid_size, i % grid_size
        for j in range(num_patches):
            row_j, col_j = j // grid_size, j % grid_size
            if abs(row_i - row_j) <= local_window // 2 and abs(col_i - col_j) <= local_window // 2:
                mask[i, j] = 1
    return mask.bool()


def make_block(num_patches=36, local_window=3, mask_mode='inverted', embed_dim=16, heads=4):
    return InvertedAttentionBlock(embed_dim, embed_dim * 2, heads, num_patches,
                                  dropout=0.0, local_window=local_window, mask_mode=mask_mode)


def make_vit(mask_mode='inverted', num_patches=16, local_window=3, **over):
    kwargs = dict(embed_dim=16, hidden_dim=32, num_channels=2, num_heads=4, num_layers=2,
                  patch_size=4, num_patches=num_patches, dropout=0.0,
                  mask_mode=mask_mode, local_window=local_window)
    kwargs.update(over)
    return VisionTransformerLocal(**kwargs)


def dummy_image(vit, batch=2):
    grid = int(math.sqrt(vit.transformer_blocks[0].num_patches))
    side = grid * vit.patch_size
    n_channels = vit.input_layer.in_features // (vit.patch_size ** 2)
    return torch.randn(batch, side, side, n_channels)


SXR_NORM = torch.tensor([-6.0, 1.0])


class TestMaskConstruction(unittest.TestCase):
    def test_inverted_matches_original_loop_exactly(self):
        """The new vectorised 'inverted' mask must equal the old loop bit-for-bit."""
        for num_patches, lw in [(36, 3), (64, 5), (49, 3)]:
            block = make_block(num_patches=num_patches, local_window=lw, mask_mode='inverted')
            ref = original_inverted_mask(num_patches, lw)
            self.assertTrue(torch.equal(block.attention_mask, ref),
                            'inverted mask differs for num_patches=%d lw=%d' % (num_patches, lw))

    def test_local_is_complement_of_inverted(self):
        inv = make_block(mask_mode='inverted').attention_mask
        loc = make_block(mask_mode='local').attention_mask
        self.assertTrue(torch.equal(loc, ~inv))

    def test_self_attention_blocked_only_in_inverted(self):
        # In the original (inverted) mask True==blocked and the diagonal is "local",
        # so a patch cannot attend to itself; in 'local' it can.
        inv = make_block(mask_mode='inverted').attention_mask
        loc = make_block(mask_mode='local').attention_mask
        self.assertTrue(bool(inv.diagonal().all()))        # all diagonal True -> blocked
        self.assertFalse(bool(loc.diagonal().any()))       # all diagonal False -> allowed

    def test_none_mode_has_no_mask(self):
        block = make_block(mask_mode='none')
        self.assertIsNone(block.attention_mask)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            make_block(mask_mode='global')


class TestForward(unittest.TestCase):
    def test_all_modes_run_and_shape_is_stable(self):
        for mode in ('inverted', 'local', 'none'):
            vit = make_vit(mask_mode=mode)
            x = dummy_image(vit)
            g, patch = vit(x, SXR_NORM, return_attention=False)
            self.assertEqual(g.shape, (2, 1))
            self.assertEqual(patch.shape, (2, vit.transformer_blocks[0].num_patches))
            self.assertTrue(torch.isfinite(g).all(), 'non-finite output in mode %s' % mode)
            # return_attention path
            g2, attn, patch2 = vit(x, SXR_NORM, return_attention=True)
            self.assertEqual(len(attn), len(vit.transformer_blocks))

    def test_mask_mode_propagates_to_blocks(self):
        vit = make_vit(mask_mode='local')
        for blk in vit.transformer_blocks:
            self.assertEqual(blk.mask_mode, 'local')
            self.assertTrue(torch.equal(blk.attention_mask, ~original_inverted_mask(
                blk.num_patches, blk.local_window)))


class TestCheckpointCompatibility(unittest.TestCase):
    def test_mask_is_persisted_for_masked_modes(self):
        """A checkpoint must carry its mask so it reproduces what it trained with."""
        inv = make_vit(mask_mode='inverted')
        self.assertTrue(any(k.endswith('attention_mask') for k in inv.state_dict()))
        loc = make_vit(mask_mode='local')
        self.assertTrue(any(k.endswith('attention_mask') for k in loc.state_dict()))
        # 'none' has no mask, so nothing to persist.
        non = make_vit(mask_mode='none')
        self.assertEqual([k for k in non.state_dict() if k.endswith('attention_mask')], [])

    def test_handedited_mask_checkpoint_reproduces_exactly(self):
        """The user's case: a checkpoint trained with a manually changed mask must
        load and use THAT mask, not the default — with identical output."""
        torch.manual_seed(0)
        ref = make_vit(mask_mode='inverted')
        P = ref.transformer_blocks[0].num_patches
        # A custom mask that matches none of the built-in modes.
        custom = torch.triu(torch.ones(P, P), diagonal=1).bool()
        for blk in ref.transformer_blocks:
            blk.attention_mask = custom.clone()
        ref.eval()
        x = dummy_image(ref)
        with torch.no_grad():
            expected, _ = ref(x, SXR_NORM)

        ckpt = ref.state_dict()  # mask is included (persistent)
        # Reload into a model constructed with the DEFAULT mode.
        fresh = make_vit(mask_mode='inverted')
        missing, unexpected = fresh.load_state_dict(ckpt, strict=True)
        self.assertEqual((list(missing), list(unexpected)), ([], []))
        for blk in fresh.transformer_blocks:
            self.assertTrue(torch.equal(blk.attention_mask, custom),
                            'saved hand-edited mask was not restored')
        fresh.eval()
        with torch.no_grad():
            got, _ = fresh(x, SXR_NORM)
        self.assertTrue(torch.allclose(expected, got, atol=1e-6))

    def test_saved_mask_wins_over_constructed_mode(self):
        """Constructing as 'inverted' but loading a 'local'-trained checkpoint must
        end up using the saved (local) mask."""
        local_ckpt = make_vit(mask_mode='local').state_dict()
        model = make_vit(mask_mode='inverted')
        model.load_state_dict(local_ckpt, strict=True)
        for blk in model.transformer_blocks:
            self.assertTrue(torch.equal(
                blk.attention_mask, ~original_inverted_mask(blk.num_patches, blk.local_window)))

    def test_set_mask_mode_overrides_loaded_mask(self):
        """Explicit ablation hook: override a loaded checkpoint's mask on purpose."""
        model = make_vit(mask_mode='inverted')
        model.load_state_dict(make_vit(mask_mode='inverted').state_dict(), strict=True)
        model.set_mask_mode('none')
        for blk in model.transformer_blocks:
            self.assertEqual(blk.mask_mode, 'none')
            self.assertIsNone(blk.attention_mask)
        model.eval()
        with torch.no_grad():
            g, _ = model(dummy_image(model), SXR_NORM)
        self.assertTrue(torch.isfinite(g).all())

    def test_state_dict_roundtrip(self):
        a = make_vit(mask_mode='local')
        b = make_vit(mask_mode='local')
        b.load_state_dict(a.state_dict(), strict=True)
        a.eval(); b.eval()
        x = dummy_image(a)
        with torch.no_grad():
            ga, _ = a(x, SXR_NORM)
            gb, _ = b(x, SXR_NORM)
        self.assertTrue(torch.allclose(ga, gb, atol=1e-6))


class TestConfigFlow(unittest.TestCase):
    BASE_KWARGS = dict(embed_dim=16, hidden_dim=32, num_channels=2, num_heads=4,
                       num_layers=2, patch_size=4, num_patches=16, dropout=0.0,
                       learning_rate=1e-4, num_classes=1)

    def test_missing_mask_mode_defaults_to_inverted(self):
        """An old config (no mask_mode key) must give the original behaviour."""
        model = ViTLocal(model_kwargs=dict(self.BASE_KWARGS), sxr_norm=SXR_NORM)
        for blk in model.model.transformer_blocks:
            self.assertEqual(blk.mask_mode, 'inverted')

    def test_mask_mode_from_config_reaches_blocks_and_hparams(self):
        kwargs = dict(self.BASE_KWARGS, mask_mode='none')
        model = ViTLocal(model_kwargs=kwargs, sxr_norm=SXR_NORM)
        for blk in model.model.transformer_blocks:
            self.assertEqual(blk.mask_mode, 'none')
            self.assertIsNone(blk.attention_mask)
        # save_hyperparameters stores model_kwargs, so load_from_checkpoint will
        # reconstruct the right mode.
        self.assertEqual(model.hparams['model_kwargs']['mask_mode'], 'none')


if __name__ == '__main__':
    unittest.main(verbosity=2)
