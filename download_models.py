"""
Pre-download Models and Dependencies for Offline Use
This script downloads all necessary models and saves them locally
"""
import os
import sys
import torch
import torchvision
from pathlib import Path

def download_pretrained_models():
    """Download pretrained backbone models"""
    print("=" * 60)
    print("Downloading Pre-trained Models for Offline Use")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("./models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Downloading Wide ResNet-50-2...")
    try:
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        torch.save(model.state_dict(), models_dir / "wide_resnet50_2.pth")
        print("✓ Wide ResNet-50-2 downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading Wide ResNet-50-2: {str(e)}")
    
    print("\n[2/3] Downloading ResNet-18...")
    try:
        model = torchvision.models.resnet18(pretrained=True)
        torch.save(model.state_dict(), models_dir / "resnet18.pth")
        print("✓ ResNet-18 downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading ResNet-18: {str(e)}")
    
    print("\n[3/3] Downloading ResNet-50...")
    try:
        model = torchvision.models.resnet50(pretrained=True)
        torch.save(model.state_dict(), models_dir / "resnet50.pth")
        print("✓ ResNet-50 downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading ResNet-50: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Model Download Summary")
    print("=" * 60)
    print(f"Models saved to: {models_dir.absolute()}")
    print("\nDownloaded models:")
    for model_file in models_dir.glob("*.pth"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name}: {size_mb:.2f} MB")
    
    return models_dir


def setup_torch_hub_cache():
    """Configure torch hub to use local cache"""
    print("\n" + "=" * 60)
    print("Configuring PyTorch Hub Cache")
    print("=" * 60)
    
    cache_dir = Path("./models/torch_hub")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable for torch hub
    os.environ['TORCH_HOME'] = str(cache_dir.absolute())
    print(f"PyTorch Hub cache set to: {cache_dir.absolute()}")
    
    return cache_dir


def verify_installation():
    """Verify all required packages are installed"""
    print("\n" + "=" * 60)
    print("Verifying Installation")
    print("=" * 60)
    
    packages = {
        'Flask': 'flask',
        'Anomalib': 'anomalib',
        'PyTorch': 'torch',
        'TorchVision': 'torchvision',
        'NumPy': 'numpy',
        'OpenCV': 'cv2',
        'PyTorch Lightning': 'pytorch_lightning'
    }
    
    all_installed = True
    for name, module in packages.items():
        try:
            __import__(module)
            version = __import__(module).__version__
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: Not installed")
            all_installed = False
        except AttributeError:
            print(f"✓ {name}: Installed (version unknown)")
    
    # Check NumPy version
    try:
        import numpy as np
        numpy_version = np.__version__
        if numpy_version.startswith('2.'):
            print(f"\n⚠ Warning: NumPy {numpy_version} detected. Anomalib 0.7.0 requires NumPy < 2.0")
            print("  Please downgrade: pip install numpy<2.0")
            all_installed = False
    except:
        pass
    
    return all_installed


def create_directory_structure():
    """Create necessary directory structure"""
    print("\n" + "=" * 60)
    print("Creating Directory Structure")
    print("=" * 60)
    
    directories = [
        './uploads',
        './models',
        './models/pretrained',
        './models/torch_hub',
        './configs',
        './results',
        './templates',
        './static/css',
        './static/js'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("PatchCore Training App - Offline Setup")
    print("Anomalib 0.7.0")
    print("=" * 60)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠ Please install missing packages before proceeding:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    create_directory_structure()
    
    # Setup torch hub cache
    setup_torch_hub_cache()
    
    # Download models
    try:
        models_dir = download_pretrained_models()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nYou can now run the application offline.")
        print("To start the server, run:")
        print("  python app.py")
        print("\nNote: On first training, PyTorch may still download some")
        print("additional model components if not cached.")
        
    except Exception as e:
        print(f"\n✗ Error during setup: {str(e)}")
        print("\nPlease check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
