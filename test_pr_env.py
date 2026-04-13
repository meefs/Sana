#!/usr/bin/env python3
"""
Test script to validate PR #368 environment and dependencies
"""

import sys
import traceback


def test_basic_imports():
    """Test basic package imports"""
    print("=== Testing basic imports ===")
    packages = [
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("mmcv", "mmcv"),
    ]

    for pkg_name, import_name in packages:
        try:
            exec(f"import {import_name}")
            print(f"✓ {pkg_name} import OK")
        except ImportError as e:
            print(f"✗ {pkg_name} import FAILED: {e}")
            return False
    return True


def test_ltx2_vae_import():
    """Test the problematic LTX2 VAE import"""
    print("\n=== Testing LTX2VAE import ===")
    try:
        from diffusers.models.autoencoders import AutoencoderKLLTX2Video

        print("✓ AutoencoderKLLTX2Video import OK")
        return True
    except ImportError as e:
        print(f"✗ AutoencoderKLLTX2Video import FAILED: {e}")
        return False


def test_available_autoencoders():
    """List all available autoencoder classes"""
    print("\n=== Available Autoencoder classes ===")
    try:
        import diffusers.models.autoencoders as ae

        autoencoders = [name for name in dir(ae) if "Autoencoder" in name]
        for cls in autoencoders:
            print(f"  - {cls}")
        return True
    except Exception as e:
        print(f"✗ Failed to list autoencoders: {e}")
        return False


def test_builder_import():
    """Test if diffusion.model.builder imports successfully"""
    print("\n=== Testing builder.py import ===")
    try:
        from diffusion.model.builder import get_vae, vae_decode, vae_encode

        print("✓ diffusion.model.builder import OK")
        return True
    except ImportError as e:
        print(f"✗ diffusion.model.builder import FAILED: {e}")
        traceback.print_exc()
        return False


def test_package_versions():
    """Print relevant package versions"""
    print("\n=== Package Versions ===")
    try:
        import diffusers

        print(f"diffusers: {diffusers.__version__}")
    except:
        print("diffusers version unknown")

    try:
        import torch

        print(f"torch: {torch.__version__}")
    except:
        print("torch version unknown")

    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
    except:
        print("transformers version unknown")


def main():
    """Run all tests"""
    print("Testing PR #368 environment compatibility\n")

    test_package_versions()

    tests = [
        test_basic_imports,
        test_available_autoencoders,
        test_ltx2_vae_import,
        test_builder_import,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
            results.append(False)

    print(f"\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("🎉 All tests passed! PR environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
